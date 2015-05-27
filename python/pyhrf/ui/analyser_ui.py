# -*- coding: utf-8 -*-

import os.path as op
import os
import sys
import traceback
import StringIO
import logging

import numpy as np

from pyhrf import xmlio, FmriData, FmriGroupData
from pyhrf.ndarray import MRI3Daxes
from pyhrf.tools import stack_trees, add_prefix
from pyhrf.tools._io import read_volume, read_texture
from pyhrf.ndarray import stack_cuboids


logger = logging.getLogger(__name__)


class FMRIAnalyser(xmlio.XmlInitable):

    P_OUTPUT_PREFIX = 'outputPrefix'
    P_ROI_AVERAGE = 'averageRoiBold'

    parametersToShow = [P_ROI_AVERAGE, P_OUTPUT_PREFIX]

    parametersComments = {
        P_ROI_AVERAGE: 'Average BOLD signals within each ROI before analysis.',
        P_OUTPUT_PREFIX: 'Tag to prefix every output name',
    }

    def __init__(self, outputPrefix='', roiAverage=False, pass_error=True,
                 gzip_outputs=False):
        xmlio.XmlInitable.__init__(self)
        if len(outputPrefix) == 0:
            outputPrefix = 'pyhrf_'
        self.outPrefix = outputPrefix
        self.outFile = add_prefix('outputs.xml', self.outPrefix)
        self.roiAverage = roiAverage
        self.pass_error = pass_error
        self.gzip_outputs = gzip_outputs

    def get_label(self):
        return 'pyhrf_fmri_analysis'

    def set_pass_errors(self, pass_error):
        self.pass_error = pass_error

    def set_gzip_outputs(self, gzip_outputs):
        self.gzip_outputs = gzip_outputs

    def __call__(self, *args, **kargs):
        return self.analyse_roi_wrap(*args, **kargs)

    def split_data(self, fdata, output_dir=None):

        if self.roiAverage:
            logger.info('Averaging ROI ...')
            fdata.average()  # TODO : debug sampling in JDE when averaging ...
        else:
            logger.info('Explode data ...')

        fdata.build_graphs()
        return fdata.roi_split()

    def analyse_roi_wrap_bak(self, roiData):

        report = 'ok'
        try:
            res = self.analyse_roi(roiData)
        except Exception:
            logger.error('!! Sampling crashed !!')
            logger.error('Exception traceback :')
            sio = StringIO.StringIO()
            traceback.print_exc(file=sio)
            # Make sure that the traceback is well garbage collected and
            # won't keep hidden references to objects :
            sys.traceback = None
            sys.exc_traceback = None
            report = sio.getvalue()
            sio.close()
            del sio
            logger.info(report)
            res = None

        return (roiData, res, report)

    def analyse_roi_wrap(self, roiData):
        """
        Wrap the analyse_roi method to catch potential exception
        """
        report = 'ok'
        if self.pass_error:
            try:
                res = self.analyse_roi(roiData)
            except Exception:
                logger.error('!! Sampling crashed !!')
                logger.error('Exception traceback :')
                sio = StringIO.StringIO()
                traceback.print_exc(file=sio)
                # Make sure that the traceback is well garbage collected and
                # won't keep hidden references to objects :
                sys.traceback = None
                sys.exc_traceback = None
                report = sio.getvalue()
                sio.close()
                del sio
                logger.info(report)
                res = None
        else:
            res = self.analyse_roi(roiData)

        return (roiData, res, report)

    def analyse_roi(self, roiData):
        raise NotImplementedError('%s does not implement roi analysis.'
                                  % self.__class__)

    def analyse(self, data, output_dir=None):
        """
        Launch the wrapped analyser onto the given data

        Args:
            - data (pyhrf.core.FmriData): the input fMRI data set (there may be
                                          multi parcels)
            - output_dir (str): the path where to store parcel-specific fMRI data
                                sets (after splitting according to the
                                parcellation mask)

        Return:
            a list of analysis results
               ->  (list of tuple(FmriData, None|output of analyse_roi, str))
               =   (list of tuple(parcel data, analysis results, analysis report))
            See method analyse_roi_wrap
        """
        logger.info("Split data ...")
        explodedData = self.split_data(data, output_dir)
        logger.info("Data splitting returned %d rois", len(explodedData))
        return [self.analyse_roi_wrap(d) for d in explodedData]

    def filter_crashed_results(self, results):
        to_pop = []
        for i, r in enumerate(results[:]):
            roi_data, result, report = r
            roi_id = roi_data.get_roi_id()
            if report != 'ok':
                logger.error('-> Sampling crashed, roi %d!', roi_id)
                logger.error(report)
                to_pop.insert(0, i)

            elif result is None:
                logger.error('-> Sampling crashed (result is None), roi %d!',
                            roi_id)
                to_pop.insert(0, i)

        for i in to_pop:
            results.pop(i)

        return results

    def make_outputs_single_subject(self, data_rois, irois, all_outputs,
                                    targetAxes, ext, meta_data, output_dir):

        coutputs = {}
        output_fns = []

        roi_masks = [(data_roi.roiMask != data_roi.backgroundLabel)
                     for data_roi in data_rois]
        np_roi_masks = [np.where(roi_mask) for roi_mask in roi_masks]

        for output_name, roi_outputs in all_outputs.iteritems():
            logger.info('Merge output %s ...', output_name)
            try:
                if roi_outputs[0].has_axis('voxel'):
                    logger.debug('Merge as expansion ...')
                    dest_c = None
                    for i_roi, c in enumerate(roi_outputs):
                        dest_c = c.expand(roi_masks[i_roi],
                                          'voxel', targetAxes,
                                          dest=dest_c, do_checks=False,
                                          m=np_roi_masks[i_roi])
                else:
                    logger.debug('Merge as stack (%d elements)...', len(irois))
                    c_to_stack = [roi_outputs[i] for i in np.argsort(irois)]
                    # print 'c_to_stack:', c_to_stack
                    # print 'sorted(irois):', sorted(irois)
                    dest_c = stack_cuboids(c_to_stack, domain=sorted(irois),
                                           axis='ROI')
            except Exception, e:
                print "Could not merge outputs for %s" % output_name
                print "Exception was:"
                print e
                # raise e #stop here
            if 0 and dest_c is not None:
                print '-> ', dest_c.data.shape
            output_fn = op.join(output_dir, output_name + ext)
            output_fn = add_prefix(output_fn, self.outPrefix)
            output_fns.append(output_fn)
            logger.debug('Save output %s to %s', output_name, output_fn)
            try:
                dest_c.save(output_fn, meta_data=meta_data,
                            set_MRI_orientation=True)
            except Exception:
                print 'Could not save output "%s", error stack was:' \
                    % output_name
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_exception(exc_type, exc_value, exc_traceback,
                                          limit=4, file=sys.stdout)
            coutputs[output_name] = dest_c

        return coutputs, output_fns

    def make_outputs_multi_subjects(self, data_rois, irois, all_outputs,
                                    targetAxes, ext, meta_data, output_dir):

        coutputs = {}
        output_fns = []

        roi_masks = [[(ds.roiMask != ds.backgroundLabel)
                      for ds in dr.data_subjects]
                     for dr in data_rois]
        #-> roi_masks[roi_id][subject]

        np_roi_masks = [[np.where(roi_subj_mask)
                         for roi_subj_mask in roi_subj_masks]
                        for roi_subj_masks in roi_masks]
        #-> np_roi_masks[roi_id][subject]

        for output_name, roi_outputs in all_outputs.iteritems():
            logger.info('Merge output %s ...', output_name)
            dest_c = {}
            try:
                if roi_outputs[0].has_axis('voxel'):
                    if not roi_outputs[0].has_axis('subject'):
                        raise Exception('Voxel-mapped output "%s" does not'
                                        'have a subject axis')
                    logger.debug('Merge as expansion ...')
                    dest_c_tmp = None
                    for isubj in roi_outputs[0].get_domain('subject'):
                        for i_roi, c in enumerate(roi_outputs):
                            m = np_roi_masks[i_roi][isubj]
                            dest_c_tmp = c.expand(roi_masks[i_roi][isubj],
                                                  'voxel', targetAxes,
                                                  dest=dest_c_tmp,
                                                  do_checks=False,
                                                  m=m)
                    dest_c[output_name] = dest_c_tmp
                else:
                    logger.debug('Merge as stack (%d elements)...', len(irois))
                    c_to_stack = [roi_outputs[i] for i in np.argsort(irois)]
                    dest_c[output_name] = stack_cuboids(c_to_stack,
                                                        domain=sorted(irois),
                                                        axis='ROI')
            except Exception, e:
                logger.error("Could not merge outputs for %s", output_name)
                logger.error("Exception was:")
                logger.error(e)

            for output_name, c in dest_c.iteritems():
                output_fn = op.join(output_dir, output_name + ext)
                output_fn = add_prefix(output_fn, self.outPrefix)
                output_fns.append(output_fn)
                logger.debug('Save output %s to %s', output_name, output_fn)
                try:
                    c.save(output_fn, meta_data=meta_data,
                           set_MRI_orientation=True)
                except Exception:
                    print 'Could not save output "%s", error stack was:' \
                        % output_name
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    traceback.print_exception(exc_type, exc_value,
                                              exc_traceback,
                                              limit=4, file=sys.stdout)
                coutputs[output_name] = c

        return coutputs, output_fns

    def outputResults(self, results, output_dir, filter='.\A',):
        """
        Return: a tuple (dictionary of outputs, output file names)
        """
        if output_dir is None:
            return {}, []
        # print results
        if not isinstance(results[0][0], (FmriData, FmriGroupData)):
            self.outputResults_back_compat(results, output_dir, filter)
            return {}, []

        logger.info('Building outputs from %d results ...', len(results))
        logger.debug('results :')
        logger.debug(results)

        # Handle analyses that crashed
        results = self.filter_crashed_results(results)

        if len(results) == 0:
            logger.warning('No more result to treat. Did everything crash ?')
            return {}, []

        # Merge all the analysis outputs
        target_shape = results[0][0].spatial_shape
        meta_data = results[0][0].meta_obj

        if len(target_shape) == 3:  # Volumic data:
            targetAxes = MRI3Daxes  # ['axial','coronal', 'sagittal']
            ext = '.nii'
        else:  # surfacic
            targetAxes = ['voxel']
            ext = '.gii'

        if self.gzip_outputs:
            ext += '.gz'

        if hasattr(results[0][1], 'getOutputs'):
            all_outputs = stack_trees([r[1].getOutputs() for r in results])
        else:
            all_outputs = stack_trees([r[1] for r in results])

        data_rois = [r[0] for r in results]
        irois = [d.get_roi_id() for d in data_rois]

        if isinstance(results[0][0], FmriData):
            return self.make_outputs_single_subject(data_rois, irois, all_outputs,
                                                    targetAxes, ext, meta_data,
                                                    output_dir)
        else:
            return self.make_outputs_multi_subjects(data_rois, irois, all_outputs,
                                                    targetAxes, ext, meta_data,
                                                    output_dir)

    def enable_draft_testing(self):
        raise NotImplementedError(
            'Enabling of draft testing is not implemented')

    def outputResults_back_compat(self, results, output_dir, filter='.\A',):

        logger.warning('Content of result.pck seems outdated, consider '
                       'running the analysis again to update it.')

        if output_dir is None:
            return

        logger.info('Building outputs ...')
        logger.debug('results :')
        logger.debug(results)

        to_pop = []
        for i, r in enumerate(results[:]):
            roi_id, result, report = r
            if report != 'ok':
                logger.info('-> Sampling crashed, roi %d!', roi_id)
                logger.info(report)
                to_pop.insert(0, i)

            elif result is None:
                logger.info('-> Sampling crashed (result is None), roi %d!',
                            roi_id)
                to_pop.insert(0, i)

        for i in to_pop:
            results.pop(i)

        if len(results) == 0:
            logger.info('No more result to treat. Did everything crash ?')
            return

        def load_any(fns, load_func):
            for f in fns:
                try:
                    return load_func(f)
                except Exception:
                    pass
            return None

        r = load_any(['roiMask.tex', 'roi_mask.tex', 'jde_roi_mask.tex'],
                     read_texture)
        if r is None:
            r = load_any(['roi_mask.nii', 'jde_roi_mask.nii'], read_volume)

        if r is None:
            raise Exception('Can not find mask data file in current dir')

        all_rois_mask, meta_data = r

        target_shape = all_rois_mask.shape

        if len(target_shape) == 3:  # Volumic data:
            targetAxes = MRI3Daxes  # ['axial','coronal', 'sagittal']
            ext = '.nii'
        else:  # surfacic
            targetAxes = ['voxel']
            ext = '.gii'

        def genzip(gens):
            while True:
                # TODO: handle dict ...
                yield [g.next() for g in gens]

        logger.info('Get each ROI output ...')
        # print 'roi outputs:'
        if hasattr(results[0][1], 'getOutputs'):
            gen_outputs = [r[1].getOutputs() for r in results]
        else:
            gen_outputs = [r[1].iteritems() for r in results]
        irois = [r[0] for r in results]

        for roi_outputs in genzip(gen_outputs):
            output_name = roi_outputs[0][0]
            logger.info('Merge output %s ...', output_name)
            if roi_outputs[0][1].has_axis('voxel'):
                logger.info('Merge as expansion ...')
                dest_c = None
                for iroi, output in zip(irois, roi_outputs):
                    _, c = output
                    if output_name == 'ehrf':
                        print 'Before expansion:'
                        print c.descrip()

                        hrfs = c.data[0, :, :]
                        print 'ehrf for cond 0', hrfs.shape
                        for ih in xrange(hrfs.shape[1]):
                            print np.array2string(hrfs[:, ih])

                        # print np.array2string(hrfs, precision=2)

                    roi_mask = (all_rois_mask == iroi)
                    dest_c = c.expand(roi_mask, 'voxel', targetAxes,
                                      dest=dest_c)
                    if output_name == 'ehrf':
                        print 'After expansion:'
                        print dest_c.descrip()
                        m = np.where(roi_mask)
                        hrfs = dest_c.data[0, :, m[0], m[1], m[2]]
                        print 'ehrf for cond 0', hrfs.shape
                        for ih in xrange(hrfs.shape[0]):
                            print np.array2string(hrfs[ih, :])
            else:
                logger.info('Merge as stack (%d elements)...', len(irois))
                c_to_stack = [roi_outputs[i][1] for i in np.argsort(irois)]
                dest_c = stack_cuboids(c_to_stack, domain=sorted(irois),
                                       axis='ROI')

            output_fn = op.join(output_dir, output_name + ext)
            logger.info('Save output %s to %s', output_name, output_fn)
            try:
                dest_c.save(output_fn, meta_data=meta_data,
                            set_MRI_orientation=True)
            except Exception:
                print 'Could not save output "%s", error stack was:' \
                    % output_name
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_exception(exc_type, exc_value, exc_traceback,
                                          limit=4, file=sys.stdout)

    def joinOutputs(self, cuboids, roiIds, mappers):

        mapper0 = mappers[roiIds[0]]
        if mapper0.isExpendable(cuboids[0]):
            logger.info('Expanding ...')
            expandedxndarray = mapper0.expandxndarray(cuboids[0])
            for c, roi in zip(cuboids[1:], roiIds[1:]):
                mappers[roi].expandxndarray(c, dest=expandedxndarray)
            return expandedxndarray
        else:
            return stack_cuboids(cuboids, 'ROI', roiIds)

    def clean_output_files(self, output_dir):
        # Clean all formatted outputs
        if self.outFile is not None and output_dir is not None:
            out_file = op.join(output_dir, self.outFile)
            if op.exists(self.outFile):
                allOuts = xmlio.from_xml(open(out_file).read())
                for c in allOuts.itervalues():
                    c.cleanFiles()
                os.remove(out_file)
