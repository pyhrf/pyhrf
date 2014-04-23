"""
Import data obtained from a Shanoir query into a hierarchical folder structure.

This script should be called in the targer data directory
"""
import os
import os.path as op
import pyhrf
from pyhrf.tools.io import rx_copy, read_volume, write_volume

def filter_fn(src, dest):
    if 'anat' in dest :
        if dest[-5] == '2': #filter session 2
                            #-> keep only session 3
            return None    
        else:
            print 'dest:', dest
            print 'new dest:', '_'.join(dest.split('_')[:4]) + '-0001.nii'
            return  '_'.join(dest.split('_')[:4]) + '_anat-0001.nii'
    if 'PASL' in src:
        if ' MB ' not in src:
            if dest[-6:-4] != '04': #keep only 1st session for ASLF data
                return None
        else:
            if dest[-6:-4] != '08': #only 8th session seems right for this subj
                return None
        if 0: #filter not-4D data files
            try:
                i,h = read_volume(src)
                if i.ndim == 3:
                    return None
            except Exception, e:
                print 'Discard because of error reading it: %s'%src
                print e
                return None

    return dest

def main():
    data_dir = './'
    shanoir_dir = op.join(data_dir, 'FromShanoir')
    
    # regexp to catch source files and define tags:
    rx_nii = r'ASL[f]* mt (?P<subject>[A-Z]+)\ *\^'           \
              '(?P<modality>3DMPRAGE|PASLFONCTTR38COUPES7MM)' \
              's(?P<session>[0-9]{3})a[0-9]{3,4}.nii'
    src_folder = shanoir_dir
    # define targets (tag between brackets {} will be replaced with group 
    # values caught bby rx_nii)
    archive_dir = op.join(data_dir, 'gin_struct', 'archives_origin')
    dest_folder = (archive_dir, '{subject}', '{subject}_{modality}')
    dest_basename = '{subject}_{modality}_session_{session}.nii'
    
    # string replacements in output file names:
    replacements = [('PASLFONCTTR38COUPES7MM', 'ASLf'),
                    ('3DMPRAGE' , 'anat')]
    
    
    if 0: # test the importation
        verb_level = 4
        dry=True
    else: # do the importation
        verb_level = 4
        dry=False

    pyhrf.verbose.set_verbosity(verb_level)
    rx_copy(rx_nii, src_folder, dest_basename, dest_folder, dry=dry, 
            replacements=replacements, callback=filter_fn)

    for subj in os.listdir(archive_dir):
        asl_base_fn = op.join(archive_dir, subj, "%s_ASLf"%subj, "%s_ASLf"%subj)
        if subj != 'MB':
            asl_src = asl_base_fn + "_session_004.nii"
        else:
            asl_src = asl_base_fn + "_session_008.nii"
        asl_dest = asl_base_fn + "-%04d.nii"
        copy_4D_to_3D_vols(asl_src, asl_dest, dry)

def copy_4D_to_3D_vols(src, dest, dry):
    print '4D -> 3D'
    print src, '->', dest
    print ''
    input_img, h = read_volume(src)
    if input_img.ndim != 4:
        raise Exception('%s should be splitted into 3D volumes but is not 4D'\
                        %src)

    if not dry:
        for i in xrange(input_img.shape[3]):
            write_volume(input_img[:,:,:,i], dest%i, h)

if __name__ == "__main__":
    main()
