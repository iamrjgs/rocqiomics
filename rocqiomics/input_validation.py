import os
import numpy as np
import SimpleITK as sitk

def key_existence_tests(case, id_col, **kwargs):
    keys = list(case.keys())
    key_tests = {
        'case_id key exists' : id_col in keys,
        'image key exists' : 'image' in keys,
        'mask key exists' : 'mask' in keys,
    }
    return [key for key, value in key_tests.items() if not value]

def file_existence_tests(case, **kwargs):
    file_existence_tests = {
        'image file exists' : os.path.exists(case['image']),
        'mask file exists' : os.path.exists(case['mask']),
    }
    return [key for key, value in file_existence_tests.items() if not value]

def file_validity_tests(case, label, **kwargs):
    mask = sitk.GetArrayFromImage(sitk.ReadImage(case['mask']))
    file_validity_tests = {
        'mask contains labels' : len(np.unique(mask)) > 1,
        'mask has multiple voxels' : len(mask[mask == label]) > 1
    }
    return [key for key, value in file_validity_tests.items() if not value]

def get_input_validation_tests():
    return {
        'key_existence' : key_existence_tests,
        'file_existence' : file_existence_tests,
        'file_validity' : file_validity_tests
    }