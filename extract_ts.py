"""
extract time series given brain parcelaiton
"""

import os
import sys
import numpy as np
from nilearn.input_data import NiftiMapsMasker

def extract_ts(func_file, atlas='msdl', output_dir='../output'):
    """extract time series given functional files and atlas"""
    # define atlas location
    atlas_file = atlas_path(atlas)
    # masker extract
    masker = NiftiMapsMasker(maps_img=atlas_file, standardize=True,
                         memory='nilearn_cache', verbose=5)
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
    if atlas_name == 'msdl':
        return '/home/fs0/syzhang/nilearn_data/msdl_atlas/MSDL_rois/msdl_rois.nii'
    else:
        raise ValueError('Atlas does not exist')

if __name__=="__main__":
    # run code
    func_file = sys.argv[1]
    extract_ts(func_file, atlas='msdl', output_dir='../output')