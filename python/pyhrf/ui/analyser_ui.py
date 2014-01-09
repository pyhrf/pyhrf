# -*- coding: utf-8 -*-


# -*- coding: utf-8 -*-
import re
import os.path as op
import os
import sys
import warnings
import traceback, StringIO
import numpy as np
#from glob import glob

import pyhrf
from pyhrf import xmlio, FmriData, FmriGroupData
from pyhrf.ndarray import MRI3Daxes

from pyhrf.boldsynth.spatialconfig import maskToMapping

from pyhrf.tools import stack_trees, set_leaf, add_prefix
from pyhrf.tools.io import read_volume, writexndarrayToTex, \
    read_texture

from pyhrf.ndarray import stack_cuboids

class FMRIAnalyser(xmlio.XmlInitable):

    P_OUTPUT_PREFIX = 'outputPrefix'
    P_ROI_AVERAGE = 'averageRoiBold'

    parametersToShow = [P_ROI_AVERAGE, P_OUTPUT_PREFIX]

    parametersComments = {
        P_ROI_AVERAGE : 'Average BOLD signals within each ROI before analysis.',
        P_OUTPUT_PREFIX : 'Tag to prefix every output name',
        }

    def __init__(self, outputPrefix='', roiAverage=False, pass_error=True):
        xmlio.XmlInitable.__init__(self)
        if len(outputPrefix) == 0: outputPrefix = 'pyhrf_'
        self.outPrefix = outputPrefix
        self.outFile = add_prefix('outputs.xml', self.outPrefix)
        self.roiAverage = roiAverage
        self.pass_error = pass_error
        #self.roi_ids = roi_ids

    def get_label(self):
        return 'pyhrf_fmri_analysis'

    def set_pass_errors(self, pass_error):
        self.pass_error = pass_error

    def __call__(self, *args, **kargs):
        return self.analyse_roi_wrap(*args,**kargs)

    def split_data(self, fdata, output_dir=None):

        if self.roiAverage:
            pyhrf.verbose(1, 'Averaging ROI ...')
            fdata.average() #TODO : debug sampling in JDE when averaging ...
        else:
            pyhrf.verbose(1, 'Explode data ...')

        # if len(self.roi_ids) > 0:
        #     pyhrf.verbose(1, 'Analysis limited to some ROIs: %s' \
        #                       %str(self.roi_ids))

        #     m0 = fdata.roiMask.copy()
        #     m = np.zeros_like(m0) + fdata.backgroundLabel
        #     for roi_id in self.roi_ids:
        #         m[np.where(m0==roi_id)] = roi_id
        # else:
        #     m = None
        fdata.build_graphs()
        return fdata.roi_split()


    def analyse_roi_wrap_bak(self, roiData):

        report = 'ok'
        # res = self.analyse_roi(roiData)
        try:
            res = self.analyse_roi(roiData)
        except Exception:
            pyhrf.verbose(1, '!! Sampling crashed !!')
            pyhrf.verbose(1,'Exception traceback :')
            sio = StringIO.StringIO()
            traceback.print_exc(file=sio)
            # Make sure that the traceback is well garbage collected and
            # won't keep hidden references to objects :
            sys.traceback = None
            sys.exc_traceback = None
            report = sio.getvalue()
            sio.close()
            del sio
            pyhrf.verbose(1,report)
            res = None

        return (roiData, res, report)


    def analyse_roi_wrap(self, roiData):

        report = 'ok'
        if self.pass_error:
            try:
                res = self.analyse_roi(roiData)
            except Exception:
                pyhrf.verbose(1, '!! Sampling crashed !!')
                pyhrf.verbose(1,'Exception traceback :')
                sio = StringIO.StringIO()
                traceback.print_exc(file=sio)
                # Make sure that the traceback is well garbage collected and
                # won't keep hidden references to objects :
                sys.traceback = None
                sys.exc_traceback = None
                report = sio.getvalue()
                sio.close()
                del sio
                pyhrf.verbose(1,report)
                res = None
        else:
            res = self.analyse_roi(roiData)

        return (roiData, res, report)

    def analyse_roi(self, roiData):
        raise NotImplementedError('%s does not implement roi analysis.' \
                                      %self.__class__)

    def analyse(self, data, output_dir=None):
        pyhrf.verbose(3, "Split data ...")
        explodedData = self.split_data(data, output_dir)
        pyhrf.verbose(3, "Data splitting returned %d rois" %len(explodedData))
        return [self.analyse_roi_wrap(d) for d in explodedData]


    def filter_crashed_results(self, results):
        to_pop = []
        for i,r in enumerate(results[:]):
            roi_data,result,report = r
            roi_id = roi_data.get_roi_id()
            if report != 'ok':
                pyhrf.verbose(1, '-> Sampling crashed, roi %d!' \
                                  %roi_id)
                pyhrf.verbose(2, report)
                to_pop.insert(0,i)

            elif result is None:
                pyhrf.verbose(1, '-> Sampling crashed (result is None), '
                              'roi %d!' %roi_id)
                to_pop.insert(0,i)

        for i in to_pop:
            results.pop(i)

        return results


    def make_outputs_single_subject(self, data_rois, irois, all_outputs,
                                    targetAxes, ext, meta_data, output_dir):

            coutputs = {}
            output_fns = []

            roi_masks = [(data_roi.roiMask != data_roi.backgroundLabel) \
                         for data_roi in data_rois]
            np_roi_masks = [np.where(roi_mask) for roi_mask in roi_masks]

            for output_name, roi_outputs in all_outputs.iteritems():
                pyhrf.verbose(3,'Merge output %s ...' %output_name)
                try:
                    if roi_outputs[0].has_axis('voxel'):
                        pyhrf.verbose(5,'Merge as expansion ...')
                        dest_c = None
                        for i_roi,c in enumerate(roi_outputs):
                            dest_c = c.expand(roi_masks[i_roi],
                                              'voxel', targetAxes,
                                              dest=dest_c, do_checks=False,
                                              m=np_roi_masks[i_roi])
                    else:
                        pyhrf.verbose(5,'Merge as stack (%d elements)...' \
                                          %len(irois))
                        c_to_stack = [roi_outputs[i] for i in np.argsort(irois)]
                        # print 'c_to_stack:', c_to_stack
                        # print 'sorted(irois):', sorted(irois)
                        dest_c = stack_cuboids(c_to_stack, domain=sorted(irois),
                                               axis='ROI')
                except Exception, e:
                    print "Could not merge outputs for %s" %output_name
                    print "Exception was:"
                    print e
                    #raise e #stop here
                if 0 and dest_c is not None:
                    print '-> ', dest_c.data.shape
                output_fn = op.join(output_dir,output_name + ext)
                output_fn = add_prefix(output_fn, self.outPrefix)
                output_fns.append(output_fn)
                pyhrf.verbose(5,'Save output %s to %s' %(output_name, output_fn))
                try:
                    dest_c.save(output_fn, meta_data=meta_data,
                                set_MRI_orientation=True)
                except Exception:
                    print 'Could not save output "%s", error stack was:' \
                        %output_name
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    traceback.print_exception(exc_type, exc_value, exc_traceback,
                                              limit=4, file=sys.stdout)
                coutputs[output_name] = dest_c

            return coutputs, output_fns



    def make_outputs_multi_subjects(self, data_rois, irois, all_outputs,
                                    targetAxes, ext, meta_data, output_dir):

            coutputs = {}
            output_fns = []

            roi_masks = [[(ds.roiMask != ds.backgroundLabel) \
                          for ds in dr.data_subjects] \
                          for dr in data_rois]
            #-> roi_masks[roi_id][subject]

            np_roi_masks = [[np.where(roi_subj_mask) \
                             for roi_subj_mask in roi_subj_masks] \
                             for roi_subj_masks in roi_masks]
            #-> np_roi_masks[roi_id][subject]

            for output_name, roi_outputs in all_outputs.iteritems():
                pyhrf.verbose(3,'Merge output %s ...' %output_name)
                dest_c = {}
                try:
                    if roi_outputs[0].has_axis('voxel'):
                        if not roi_outputs[0].has_axis('subject'):
                            raise Exception('Voxel-mapped output "%s" does not'\
                                            'have a subject axis')
                        pyhrf.verbose(5,'Merge as expansion ...')
                        dest_c_tmp = None
                        for isubj in roi_outputs[0].get_domain('subject'):
                            for i_roi,c in enumerate(roi_outputs):
                                m = np_roi_masks[i_roi][isubj]
                                dest_c_tmp = c.expand(roi_masks[i_roi][isubj],
                                                      'voxel', targetAxes,
                                                      dest=dest_c_tmp,
                                                      do_checks=False,
                                                      m=m)
                        dest_c[output_name] = dest_c_tmp
                    else:
                        pyhrf.verbose(5,'Merge as stack (%d elements)...' \
                                          %len(irois))
                        c_to_stack = [roi_outputs[i] for i in np.argsort(irois)]
                        # print 'c_to_stack:', c_to_stack
                        # print 'sorted(irois):', sorted(irois)
                        dest_c[output_name] = stack_cuboids(c_to_stack,
                                                            domain=sorted(irois),
                                                            axis='ROI')
                except Exception, e:
                    print "Could not merge outputs for %s" %output_name
                    print "Exception was:"
                    print e

                for output_name, c in dest_c.iteritems():
                    output_fn = op.join(output_dir,output_name + ext)
                    output_fn = add_prefix(output_fn, self.outPrefix)
                    output_fns.append(output_fn)
                    pyhrf.verbose(5,'Save output %s to %s' \
                                  %(output_name, output_fn))
                    try:
                        c.save(output_fn, meta_data=meta_data,
                               set_MRI_orientation=True)
                    except Exception:
                        print 'Could not save output "%s", error stack was:' \
                            %output_name
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
        #print results
        if not isinstance(results[0][0], (FmriData, FmriGroupData)):
            self.outputResults_back_compat(results, output_dir, filter)
            return {}, []

        pyhrf.verbose(1,'Building outputs from %d results ...' %len(results))
        pyhrf.verbose(6, 'results :')
        pyhrf.verbose.printDict(6, results, exclude=['xmlHandler'])

        results = self.filter_crashed_results(results)

        if len(results) == 0:
            pyhrf.verbose(1, 'No more result to treat. Did everything crash ?')
            return {}, []


        target_shape = results[0][0].spatial_shape
        meta_data = results[0][0].meta_obj

        if len(target_shape) == 3: #Volumic data:
            targetAxes = MRI3Daxes #['axial','coronal', 'sagittal']
            ext = '.nii'
        else: #surfacic
            targetAxes = ['voxel']
            ext = '.gii'

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



    def outputResults_old2(self, results, output_dir, filter='.\A',):
        """
        Return: a tuple (dictionary of outputs, output file names)
        """
        if output_dir is None:
            return {}, []

        if not isinstance(results[0][0], FmriData, FmriGroupData):
            self.outputResults_back_compat(results, output_dir, filter)
            return {}, []

        pyhrf.verbose(1,'Building outputs from %d results ...' %len(results))
        pyhrf.verbose(6, 'results :')
        pyhrf.verbose.printDict(6, results, exclude=['xmlHandler'])

        to_pop = []
        for  i,r in enumerate(results[:]):
            roi_data,result,report = r
            roi_id = roi_data.get_roi_id()
            if report != 'ok':
                pyhrf.verbose(1, '-> Sampling crashed, roi %d!' \
                                  %roi_id)
                pyhrf.verbose(2, report)
                to_pop.insert(0,i)

            elif result is None:
                pyhrf.verbose(1, '-> Sampling crashed (result is None), '
                              'roi %d!' %roi_id)
                to_pop.insert(0,i)

        for i in to_pop:
            results.pop(i)

        if len(results) == 0:
            pyhrf.verbose(1, 'No more result to treat. Did everything crash ?')
            return {}, []

        target_shape = results[0][0].spatial_shape
        meta_data = results[0][0].meta_obj

        if len(target_shape) == 3: #Volumic data:
            targetAxes = MRI3Daxes #['axial','coronal', 'sagittal']
            ext = '.nii'
        else: #surfacic
            targetAxes = ['voxel']
            ext = '.gii'

        def genzip(gens):
            while True:
                yield [g.next() for g in gens]

        coutputs = {}
        output_fns = []

        pyhrf.verbose(1,'Get each ROI output ...')
        #print 'roi outputs:'
        if hasattr(results[0][1], 'getOutputs'):
            gen_outputs = [r[1].getOutputs() for r in results]
        else:
            gen_outputs = [r[1].iteritems() for r in results]

        data_rois = [r[0] for r in results]
        irois = [d.get_roi_id() for d in data_rois]
        for roi_outputs in genzip(gen_outputs):
            output_name = roi_outputs[0][0]
            pyhrf.verbose(3,'Merge output %s ...' %output_name)
            try:
                if roi_outputs[0][1].has_axis('voxel'):
                    pyhrf.verbose(5,'Merge as expansion ...')
                    dest_c = None
                    for data_roi,output in zip(data_rois,roi_outputs):
                        _, c = output
                        roi_mask = (data_roi.roiMask != data_roi.backgroundLabel)

                        dest_c = c.expand(roi_mask, 'voxel', targetAxes,
                                          dest=dest_c)
                else:
                    pyhrf.verbose(5,'Merge as stack (%d elements)...' \
                                      %len(irois))
                    c_to_stack = [roi_outputs[i][1] for i in np.argsort(irois)]
                    # print 'c_to_stack:', c_to_stack
                    # print 'sorted(irois):', sorted(irois)
                    dest_c = stack_cuboids(c_to_stack, domain=sorted(irois),
                                           axis='ROI')
            except Exception, e:
                print "Could not merge outputs for %s" %output_name
                print "Exception was:"
                print e
                #raise e #stop here
            if 0 and dest_c is not None:
                print '-> ', dest_c.data.shape
            output_fn = op.join(output_dir,output_name + ext)
            output_fn = add_prefix(output_fn, self.outPrefix)
            output_fns.append(output_fn)
            pyhrf.verbose(5,'Save output %s to %s' %(output_name, output_fn))
            try:
                dest_c.save(output_fn, meta_data=meta_data,
                            set_MRI_orientation=True)
            except Exception:
                print 'Could not save output "%s", error stack was:' \
                    %output_name
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_exception(exc_type, exc_value, exc_traceback,
                                          limit=4, file=sys.stdout)
            coutputs[output_name] = dest_c

        return coutputs, output_fns



    def enable_draft_testing(self):
        raise NotImplementedError('Enabling of draft testing is not implemented')

    def outputResults_back_compat(self, results, output_dir, filter='.\A',):

        warnings.warn('Content of result.pck seems outdated, consider ' \
                          'running the analysis again to update it.',
                      DeprecationWarning)

        if output_dir is None:
            return

        pyhrf.verbose(1,'Building outputs ...')
        pyhrf.verbose(6, 'results :')
        pyhrf.verbose.printDict(6, results, exclude=['xmlHandler'])

        to_pop = []
        for  i,r in enumerate(results[:]):
            roi_id,result,report = r
            if report != 'ok':
                pyhrf.verbose(1, '-> Sampling crashed, roi %d!' %roi_id)
                pyhrf.verbose(2, report)
                to_pop.insert(0,i)

            elif result is None:
                pyhrf.verbose(1, '-> Sampling crashed (result is None), '
                              'roi %d!' %roi_id)
                to_pop.insert(0,i)

        for i in to_pop:
            results.pop(i)

        if len(results) == 0:
            pyhrf.verbose(1, 'No more result to treat. Did everything crash ?')
            return


        def load_any(fns, load_func):
            for f in fns:
                try:
                    return load_func(f)
                except Exception:
                    pass
            return None

        r = load_any(['roiMask.tex', 'roi_mask.tex',  'jde_roi_mask.tex'],
                     read_texture)
        if r is None:
            r = load_any(['roi_mask.nii', 'jde_roi_mask.nii'], read_volume)

        if r is None:
            raise Exception('Can not find mask data file in current dir')

        all_rois_mask, meta_data = r

        target_shape = all_rois_mask.shape

        if len(target_shape) == 3: #Volumic data:
            targetAxes = MRI3Daxes#['axial','coronal', 'sagittal']
            ext = '.nii'
        else: #surfacic
            targetAxes = ['voxel']
            ext = '.gii'

        def genzip(gens):
            while True:
                #TODO: handle dict ...
                yield [g.next() for g in gens]

        pyhrf.verbose(1,'Get each ROI output ...')
        #print 'roi outputs:'
        if hasattr(results[0][1], 'getOutputs'):
            gen_outputs = [r[1].getOutputs() for r in results]
        else:
            gen_outputs = [r[1].iteritems() for r in results]
        irois = [r[0] for r in results]

        for roi_outputs in genzip(gen_outputs):
            output_name = roi_outputs[0][0]
            pyhrf.verbose(3,'Merge output %s ...' %output_name)
            if roi_outputs[0][1].has_axis('voxel'):
                pyhrf.verbose(3,'Merge as expansion ...')
                dest_c = None
                for iroi,output in zip(irois,roi_outputs):
                    _, c = output
                    if output_name == 'ehrf':
                        print 'Before expansion:'
                        print c.descrip()

                        hrfs = c.data[0,:,:]
                        print 'ehrf for cond 0', hrfs.shape
                        for ih in xrange(hrfs.shape[1]):
                            print np.array2string(hrfs[:,ih])

                        #print np.array2string(hrfs, precision=2)

                    roi_mask = (all_rois_mask==iroi)
                    dest_c = c.expand(roi_mask, 'voxel', targetAxes,
                                      dest=dest_c)
                    if output_name == 'ehrf':
                        print 'After expansion:'
                        print dest_c.descrip()
                        m = np.where(roi_mask)
                        hrfs = dest_c.data[0,:,m[0],m[1],m[2]]
                        print 'ehrf for cond 0', hrfs.shape
                        for ih in xrange(hrfs.shape[0]):
                            print np.array2string(hrfs[ih,:])
                        #print np.array2string(hrfs, precision=2)
            else:
                pyhrf.verbose(3,'Merge as stack (%d elements)...' \
                                  %len(irois))
                c_to_stack = [roi_outputs[i][1] for i in np.argsort(irois)]
                dest_c = stack_cuboids(c_to_stack, domain=sorted(irois),
                                       axis='ROI')

            output_fn = op.join(output_dir,output_name + ext)
            pyhrf.verbose(3,'Save output %s to %s' %(output_name, output_fn))
            try:
                dest_c.save(output_fn, meta_data=meta_data,
                            set_MRI_orientation=True)
            except Exception:
                print 'Could not save output "%s", error stack was:' \
                    %output_name
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_exception(exc_type, exc_value, exc_traceback,
                              limit=4, file=sys.stdout)


    def outputResults_old(self, results, output_dir, filter='.\A',
                      historyFlag=False):
        """ Return a dictionnary mapping output names to cuboid objects
        """
        from pyhrf.boldsynth.spatialconfig import xndarrayMapper1D

        filter = re.compile(filter)

        pyhrf.verbose(1,'Building outputs ...')
        pyhrf.verbose(6, 'results :')
        pyhrf.verbose.printDict(6, results, exclude=['xmlHandler'])

        for i,r in enumerate(results):
            # print 'i:', i
            # print 'r:', r
            if r[1] is None:
                results.pop(i)

        nbROI = len(results)
        if nbROI == 0: #Nothing ...
            print 'No result found, everything crashed?'
            return None

        pyhrf.verbose(1, '%s ROI(s)'  %(nbROI))
        resultTree = {}
        roiMapperTree = {}

        target_shape = results[0][0].spatial_shape
        meta_data = results[0][0].meta_obj

        if len(target_shape) == 3: #Volumic data:
            targetAxes = MRI3Daxes#['axial','coronal', 'sagittal']
        else: #surfacic
            targetAxes = ['voxel']

        pyhrf.verbose(1,'Get each ROI output ...')
        for roiData,result,report in results:
            roiId = roiData.get_roi_id()
            roiMapper = xndarrayMapper1D(maskToMapping(roiData.roiMask==roiId),
                                       target_shape, targetAxes, 'voxel')

            if report == 'ok':
                if not isinstance(result, dict):
                    outs = result.getOutputs()
                else:
                    outs = result
                set_leaf(resultTree,[roiId], outs)
                set_leaf(roiMapperTree,[roiId], roiMapper)
            else:
                pyhrf.verbose(1, '-> Sampling crashed, roi %d!' %roiId)
                pyhrf.verbose(2, report)

        if len(resultTree) == 0:
            return None

        pyhrf.verbose(1,'Joining outputs ...')
        outputs = {}
        outputs = stack_trees(resultTree.values())
        roiList = resultTree.keys()
        topop = []
        for on, cubList in outputs.iteritems():
            pyhrf.verbose(4, "Treating output: " + on + "...")
            if not filter.match(on):
                try:
                    outputs[on] = self.joinOutputs(cubList, roiList,
                                                   roiMapperTree)
                except Exception, e:
                    print 'Could not join outputs for', on
                    print 'Exception was:'
                    print e
                    topop.append(on)
            else:
                print on, 'filtered!'
        for on in topop:
            outputs.pop(on)

        pyhrf.verbose(5, 'output keys:'+ str(outputs.keys()))

        pyhrf.verbose(6, 'tree of results :')
        pyhrf.verbose.printDict(6, resultTree, exclude=['xmlHandler'])

        # Apply prefix:
        newOutputs = {}
        for on,ov in outputs.items():
            pname = self.outPrefix + on
            # make sure we don't overwrite anything:
            assert (pname==on or not outputs.has_key(pname))
            newOutputs[pname] = ov

        outputs = newOutputs

        if output_dir is not None:
            from pyhrf.tools.io import xndarrayXml2

            pyhrf.verbose(1,'Writing outputs ...')
            pyhrf.verbose(1,'Output directory: ' + output_dir)
            oxml = {}
            for on,ov in outputs.iteritems():
                oxml[on] = xndarrayXml2.fromxndarray(ov, label=on,
                                                 outDir=output_dir,
                                                 relativePath=False,
                                                 meta_data=meta_data)

            if len(target_shape) == 3: #volumic data
                so = xmlio.to_xml(oxml, handler=NumpyXMLHandler())
                out_file = op.join(output_dir,self.outFile)
                f = open(out_file,'w')
                f.write(so)
                f.close()
            else: #surfacic data #TODO: xml output ... ?
                for name, out in outputs.items():
                    fn = op.join(output_dir, name + '.gii')
                    writexndarrayToTex(out, fn)

        else:
            pyhrf.verbose(1,"No output (file not specified)")
        return outputs


    def joinOutputs(self, cuboids, roiIds, mappers):

        mapper0 = mappers[roiIds[0]]
        if mapper0.isExpendable(cuboids[0]):
            pyhrf.verbose(4, 'Expanding ...')
            expandedxndarray = mapper0.expandxndarray(cuboids[0])
            for c,roi in zip(cuboids[1:],roiIds[1:]):
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








