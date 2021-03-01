"""
extract time series given brain parcelaiton
"""

import os
import sys
import numpy as np
from nilearn.input_data import NiftiMapsMasker, NiftiLabelsMasker

def extract_ts(func_file, atlas='msdl', output_dir='../output'):
    """extract time series given functional files and atlas"""
    # define atlas location
    atlas_file = atlas_path(atlas)
    # masker extract
    if atlas == 'msdl':
        masker = NiftiMapsMasker(maps_img=atlas_file, standardize=True, memory='nilearn_cache', verbose=5)
    else:
        masker = NiftiLabelsMasker(labels_img=atlas_file, standardize=True, memory='nilearn_cache', verbose=5)

    time_series = masker.fit_transform(func_file)
    # save
    save_dir = os.path.join(output_dir, atlas)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    file_name = func_file.split('/')[-1].split('.')[0]
    output_file = os.path.join(save_dir, file_name+'.npy')
    np.save(output_file, time_series)
    print(f'output saved to {output_file}')

def atlas_path(atlas_name):
    """define atlas location on cluster"""
    base_dir = '/home/fs0/syzhang/'
    if atlas_name == 'msdl':
        return base_dir + 'nilearn_data/msdl_atlas/MSDL_rois/msdl_rois.nii'
    elif atlas_name == 'fan':
        return '../atlas/Fan_et_al_atlas_r279_MNI_2mm.nii'
    elif atlas_name == 'yeo':
        return base_dir + 'nilearn_data/yeo_2011/Yeo_JNeurophysiol11_MNI152/Yeo2011_17Networks_MNI152_FreeSurferConformed1mm_LiberalMask.nii.gz'
    else:
        raise ValueError('Atlas does not exist')

if __name__=="__main__":
    # run code
    func_file = sys.argv[1]
    # extract_ts(func_file, atlas='msdl', output_dir='../output')
    # extract_ts(func_file, atlas='fan', output_dir='../output')
    extract_ts(func_file, atlas='yeo', output_dir='../output')