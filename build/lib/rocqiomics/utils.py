from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pydicom
import numpy as np

from monai.data import itk_torch_bridge
import itk
import SimpleITK as sitk

def read_dicom_series_as_nrrd(dcm_dirpath):
    dcm_names = []
    
    reader = sitk.ImageSeriesReader()
    series_IDs = reader.GetGDCMSeriesIDs(str(dcm_dirpath))

    if series_IDs:
        series_file_lists = [reader.GetGDCMSeriesFileNames(str(dcm_dirpath), sid) for sid in series_IDs]
        sid = series_IDs[max(range(len(series_file_lists)), key=lambda i: len(series_file_lists[i]))]
        dcm_names = reader.GetGDCMSeriesFileNames(str(dcm_dirpath), sid)
    else:
        dcm_dirpath_obj = Path(dcm_dirpath)
        for f in sorted(dcm_dirpath_obj.glob("*")):
            if f.is_file():
                try:
                    ds = pydicom.dcmread(f, stop_before_pixels=True)
                    uid = getattr(ds, "SeriesInstanceUID", None)
                    dcm_names.append(str(f))
                except Exception as e:
                    pass

    reader.SetFileNames(dcm_names)
    image = reader.Execute()

    return image

def get_dataset_info(data_dicts, id_col='case_id', include_mask_data=True, limit=None):
    metadata_keys = list(dict(data_dicts[0]['metadata']).keys())
    ddicts = data_dicts if limit is None else data_dicts[0:limit]

    ids = []
    full_data = []
    unique_values = {}

    for d in ddicts:
        case_dict = {}
        ids.append(d[id_col])

        for k in metadata_keys:
            case_dict[k] = dict(d['metadata'])[k]
        
        img = sitk.ReadImage(d['image'])
        img_array = sitk.GetArrayFromImage(img)

        case_dict['img_spacing'] = tuple(np.round(img.GetSpacing(), 3))
        case_dict['img_size'] = tuple(img.GetSize())
        case_dict['img_direction'] = tuple(np.round(img.GetDirection(), 3))
        case_dict['img_origin'] = tuple(np.round(img.GetOrigin(), 3))
        case_dict['img_dtype'] = img_array.dtype
        case_dict['img_intensity_min'] = np.min(img_array)
        case_dict['img_intensity_max'] = np.max(img_array)
        case_dict['img_intensity_mean'] = np.mean(img_array)

        if include_mask_data:
            mask = sitk.ReadImage(d['mask'])
            mask_array = sitk.GetArrayFromImage(mask)

            case_dict['mask_spacing'] = tuple(np.round(mask.GetSpacing(), 3))
            case_dict['mask_size'] = tuple(mask.GetSize())
            case_dict['mask_direction'] = tuple(mask.GetDirection())
            case_dict['mask_origin'] = tuple(np.round(mask.GetOrigin(), 3))
            case_dict['mask_dtype'] = mask_array.dtype
            case_dict['mask_unique_values'] = tuple((np.unique(mask_array)))

        full_data.append(case_dict)
    
    full_df = pd.DataFrame(full_data)
    full_df.index = ids

    for col in full_df.columns:
        unique_values[col] = full_df[col].unique()
    
    return full_df, unique_values

def plot_data_dict(data_dict, rows=2, slice='auto', slice_step=3, id_col='case_id'):

    img = sitk.GetArrayFromImage(sitk.ReadImage(data_dict['image']))
    mask = sitk.GetArrayFromImage(sitk.ReadImage(data_dict['mask']))

    fig, ax = plt.subplots(rows, 2)

    sl = slice 
    if slice == 'auto':
        # Choose slice where mask is known to have non-zero value
        slice_indices = np.where(mask.any(axis=(1, 2)))[0]
        sl = slice_indices[len(slice_indices) // 2]

    for i in range(rows):

        im = img[sl,:,:]
        ma = mask[sl,:,:]

        if rows > 1:
            ax[i, 0].imshow(im, cmap='gray')
            ax[i, 1].imshow(im, cmap='gray')
            ax[i, 1].imshow(ma, cmap='inferno', alpha=0.5)
        else:
            ax[0].imshow(im, cmap='gray')
            ax[1].imshow(im, cmap='gray')
            ax[1].imshow(ma, cmap='inferno', alpha=0.5)

        sl += slice_step

    for a in ax.flatten():
        a.axis('off')

    fig.tight_layout()

def get_sitk_image_metadata(image):
     keys = image.GetMetaDataKeys()
     return {key : image.GetMetaData(key) for key in keys}

def copy_sitk_metadata(img, target):
    for key in target.GetMetaDataKeys():
        img.SetMetaData(key, target.GetMetaData(key))
    return img

def tensor_to_sitk(image=None, dtype=np.float32):
    image = itk_torch_bridge.metatensor_to_itk_image(image, dtype=dtype)
    return itk_to_sitk(image)

def itk_to_sitk(itk_image):
    new_sitk_image = sitk.GetImageFromArray(itk.GetArrayFromImage(itk_image), isVector=itk_image.GetNumberOfComponentsPerPixel()>1)
    new_sitk_image.SetOrigin(tuple(itk_image.GetOrigin()))
    new_sitk_image.SetSpacing(tuple(itk_image.GetSpacing()))
    new_sitk_image.SetDirection(itk.GetArrayFromMatrix(itk_image.GetDirection()).flatten()) 
    return new_sitk_image

def resample_to_target_image(img, target, is_mask=False, interpolator=None):
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(target)
    resampler.SetDefaultPixelValue(0)

    if is_mask:
        interpolator = sitk.sitkNearestNeighbor
    else:
        if interpolator is None:
            interpolator = sitk.sitkBSpline3
            
    resampler.SetInterpolator(interpolator)
    img = resampler.Execute(img)
    img.CopyInformation(target)
    img = copy_sitk_metadata(img, target)
    return img
               
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

def mask_nonzero_slices_per_dimension(mask):
    arr = sitk.GetArrayFromImage(mask)
    slices_z = [i for i in range(arr.shape[0]) if np.any(arr[i, :, :])]
    slices_y = [j for j in range(arr.shape[1]) if np.any(arr[:, j, :])]
    slices_x = [k for k in range(arr.shape[2]) if np.any(arr[:, :, k])]
    return [slices_x, slices_y, slices_z]

def is2D(image):
    return sum(dim > 1 for dim in image.GetSize()) == 2

def mask_is2D(mask):
    slices = mask_nonzero_slices_per_dimension(mask)
    return any(len(s) == 1 for s in slices)

def stringify_value(v):
    if isinstance(v, (int, float, str, bool, type(None))):
        return repr(v)
    if isinstance(v, (list, tuple)):
        inner = ", ".join(stringify_value(x) for x in v)
        return f"[{inner}]" if isinstance(v, list) else f"({inner})"
    if isinstance(v, dict):
        inner = ", ".join(f"{k}={stringify_value(val)}"
                        for k, val in v.items()
                        if not k.startswith("_"))
        return f"{{{inner}}}"
    if hasattr(v, "__dict__"):
        cls = v.__class__.__name__
        params = ", ".join(
            f"{k}={stringify_value(val)}"
            for k, val in v.__dict__.items()
            if not k.startswith("_")
        )
        return f"{cls}({params})"
    return repr(v)

def stringify_transforms(obj):
    if hasattr(obj, "transforms"):
        return [stringify_value(t) for t in obj.transforms]
    if isinstance(obj, (list, tuple)):
        return [stringify_value(t) for t in obj]
    return [stringify_value(obj)]