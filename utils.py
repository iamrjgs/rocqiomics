import pandas as pd
import numpy as np

from monai.data import itk_torch_bridge

import itk
import SimpleITK as sitk

def get_sitk_image_metadata(image):
     keys = image.GetMetaDataKeys()
     return {key : image.GetMetaData(key) for key in keys}

def tensor_to_sitk(image=None, dtype=np.float32):
        image = itk_torch_bridge.metatensor_to_itk_image(image, dtype=dtype)
        return itk_to_sitk(image)

def itk_to_sitk(itk_image):
    new_sitk_image = sitk.GetImageFromArray(itk.GetArrayFromImage(itk_image), isVector=itk_image.GetNumberOfComponentsPerPixel()>1)
    new_sitk_image.SetOrigin(tuple(itk_image.GetOrigin()))
    new_sitk_image.SetSpacing(tuple(itk_image.GetSpacing()))
    new_sitk_image.SetDirection(itk.GetArrayFromMatrix(itk_image.GetDirection()).flatten()) 
    
    return new_sitk_image

def resample_to_target_image(img, target, is_mask=False):
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(target)
        resampler.SetDefaultPixelValue(0)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear)
        return resampler.Execute(img)
               
def split_dataframe_by_unique_values_in_columns(df, columns):
    # Function to split dataframe into multiple dataframes separated by unique values in specified columns
    # E.g. if 'timepoint' in columns, then separates into dataframes with unique timepoints
    unique_values = df[columns].drop_duplicates()
    unique_vals_result_dicts = []

    for _, row in unique_values.iterrows():
        result_dict = {col : val for col, val in zip(columns, list(row))}

        result_df = df[(df[columns] == row).all(axis=1)]
        result_dict['df'] = result_df

        unique_vals_result_dicts.append(result_dict)
    
    return unique_vals_result_dicts

