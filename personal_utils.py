from datetime import datetime
import json
import os
import re
import shutil

import pandas as pd
import pingouin as pg
import numpy as np

from monai.data import itk_torch_bridge

import itk
import SimpleITK as sitk

from rocqiomics.constants import CONSTANTS

def sort_by_numerical_suffix(case_ids):
    if isinstance(case_ids, list):
        return sorted(case_ids, key=lambda x: int(re.search(r"\d+$", x).group()))
    else:
        return case_ids.loc[sort_by_numerical_suffix(case_ids.index.to_list())]

def prepare_data_dicts(
        base_dirpath='',
        case_ids=None,
        mask_names=None,
        timepoints=None,
        modalities=None,
        image_extension='nrrd',
        mask_extension='seg.nrrd',
        ):

        data_dicts = []

        imaging_dirpath = os.path.join(base_dirpath, 'Images')
        masks_dirpath = os.path.join(base_dirpath, 'Masks')

        if case_ids is None:
            case_ids = os.listdir(imaging_dirpath)
            case_ids = [cid.split('_')[2:] for cid in case_ids]
            case_ids = ['_'.join(cid) for cid in case_ids]
            case_ids = [cid.replace(f'.{image_extension}', '') for cid in case_ids]
            case_ids = list(set(case_ids))
        case_ids = sort_by_numerical_suffix(case_ids)
    
        if timepoints is None:
            timepoints = os.listdir(imaging_dirpath)
            timepoints = [tp.split('_')[0] for tp in timepoints]
            timepoints = list(set(timepoints))

        if modalities is None:
            modalities = os.listdir(imaging_dirpath)
            modalities = [md.split('_')[1] for md in modalities]
            modalities = list(set(modalities))
        
        if mask_names is None:
            mask_names = os.listdir(masks_dirpath)
        
        for case_id in case_ids:
            for mask in mask_names:
                mask_dirpath = os.path.join(masks_dirpath, mask)

                for md in modalities:
                    for tp in timepoints:

                        image_filename = f'{tp}_{md}_{case_id}.{image_extension}'
                        image_path = os.path.join(imaging_dirpath, image_filename)

                        mask_filename = f'{tp}_{md}_{case_id}.{mask_extension}'
                        mask_path = os.path.join(mask_dirpath, mask_filename)

                        if os.path.exists(image_path) and os.path.exists(mask_path):
                            data_dicts.append({
                                'case_id' : case_id,
                                'image' : image_path,
                                'mask' : mask_path,
                                'metadata' : {
                                    'mask_name' : mask,
                                    'timepoint' : tp,
                                    'modality' : md
                                }
                            })
        
        return data_dicts

def prepare_data_dicts_from_old_format(
        imaging_dirpath='',
        subject_list_filepath=None,
        id_col='PTID',
        mask_names=[],
        timepoints=[],
        modalities=[],
        image_extension='nrrd',
        segmentation_extension='seg.nrrd',
        segmentations_folder='',
        ):
    """
    Prepare data_dicts for Rocqiomics from old directory format:
        case_folder >> timepoint >> modality >> images and segmentations
    """ 

    data_dicts = []
    missing_images = []
    missing_segmentations = []

    if subject_list_filepath is not None:
        case_ids = pd.read_excel(subject_list_filepath).set_index(id_col).index.to_list()
    else:
        case_ids = os.listdir(imaging_dirpath)

    # Iterate over each case
    for case in case_ids:
        timepoints_path = os.path.join(imaging_dirpath, case)
        case_timepoints = [tp for tp in os.listdir(timepoints_path) if tp in timepoints]

        # Iterate over each timepoint
        for tp in case_timepoints:
            modalities_path = os.path.join(timepoints_path, tp)
            case_modalities = [md for md in os.listdir(modalities_path) if md in modalities]

            # Iterate over each modality
            for md in case_modalities:
                image_data_path = os.path.join(modalities_path, md)
                image_filepath = os.path.join(image_data_path, f'{md}.{image_extension}')

                if not os.path.exists(image_filepath):
                    missing_images.append({
                        'case_id' : case,
                        'timepoint' : tp,
                        'modality' : md
                    })

                # If image exists
                else:
                    for mask in mask_names:
                        
                        # Get all masks in segmentation folder containing mask_name
                        seg_folderpath = os.path.join(image_data_path, segmentations_folder) if segmentations_folder else image_data_path
                        segmentations = [seg for seg in os.listdir(seg_folderpath) if (mask in seg and segmentation_extension in seg)]

                        # Check if segmentations exist for given mask_name
                        if len(segmentations) == 0:
                            missing_segmentations.append({
                                'case_id' : case,
                                'timepoint' : tp,
                                'modality' : md,
                                'mask_name' : mask
                            })

                        # If segmentation(s) exist
                        else:
                            for seg in segmentations:
                                seg_filepath = os.path.join(seg_folderpath, seg)

                                # Prepare data dicts
                                data_dicts.append({
                                    'case_id' : case,
                                    'image' : image_filepath,
                                    'mask' : seg_filepath,
                                    'metadata' : {
                                        'mask_name' : mask,
                                        'modality' : md,
                                        'timepoint' : tp
                                    }
                                })

    return data_dicts, missing_images, missing_segmentations

def unpack_old_directory_format(
    old_imaging_dirpath='',
    new_imaging_dirpath='',
    new_masks_dirpath='',
    mask_names=[],
    image_extension='nrrd',
    segmentation_extension='seg.nrrd',
    segmentations_folder='',
    copy_images=True,
    copy_masks=True,
    ):
    """
    Copies a folder of imaging structured in the old format into the new format used.

    Old directory format was case >> timepoint >> modality >> images and masks
    New directory format contains images and masks folders and timepoints/modality info goes in filenames
    
    Time has shown it is more convenient to have all images in the same folder (can more easily load multiple images simultaneously into 3D Slicer, for instance)
    """

    if not os.path.exists(new_imaging_dirpath):
        os.makedirs(new_imaging_dirpath)
    
    if not os.path.exists(new_masks_dirpath):
        os.makedirs(new_masks_dirpath)

    for case_id in os.listdir(old_imaging_dirpath):
        print(f'Running case {case_id}...')

        timepoints_path = os.path.join(old_imaging_dirpath, case_id)
        timepoints = os.listdir(timepoints_path)

        for tp in timepoints:
            modalities_path = os.path.join(timepoints_path, tp)
            modalities = os.listdir(modalities_path)

            for md in modalities:
                old_image_path = os.path.join(modalities_path, md, f'{md}.{image_extension}')

                new_image_name = f'{tp}_{md}_{case_id}.{image_extension}'
                new_image_path = os.path.join(new_imaging_dirpath, new_image_name)
                
                if copy_images:
                    shutil.copy(old_image_path, new_image_path)

                masks_folder = os.path.join(modalities_path, md)
                if len(segmentations_folder) > 0:
                    masks_folder = os.path.join(modalities_path, md, segmentations_folder)

                for mask in mask_names:

                    new_mask_dir = os.path.join(new_masks_dirpath, mask)
                    if not os.path.exists(new_mask_dir):
                        os.makedirs(new_mask_dir)

                    old_mask_path = os.path.join(masks_folder, f'{mask}.{segmentation_extension}')
                    
                    if os.path.exists(old_mask_path):
                        new_mask_name = f'{tp}_{md}_{mask}_{case_id}.{segmentation_extension}'
                        new_mask_path = os.path.join(new_mask_dir, new_mask_name)

                        if copy_masks:
                            shutil.copy(old_mask_path, new_mask_path)

def copy_images_from_analyze_hdr_folder(
        hdr_folder='',
        destination_folder='',
        export_extension='.nrrd',
        timepoints=None,
        modality_keywords=None,
        modality_names=None,
):  
    """
    modality_keywords are the search keyword to identify images of a given modality (e.g. T2_rare for T2-weighted images)
    modality_names are the names you want to want to give for the modalities on the new images (e.g. T2, ADC)
    """

    info_json = {
        'conversion_datetime' : datetime.now().strftime("%I:%M%p on %B %d, %Y"),
        'original_folder' : hdr_folder,
        'destination_folder' : destination_folder,
        'cases' : []
    }

    os.makedirs(destination_folder, exist_ok=True)
    os.makedirs(os.path.join(destination_folder, 'Images'), exist_ok=True)

    if modality_keywords is None or modality_names is None:
        raise ValueError ("modality_keywords and modality_names must be lists of the same length")

    if timepoints is None:
        timepoints = []
        for case in os.listdir(hdr_folder):
            timepoints.extend(os.listdir(os.path.join(hdr_folder, case)))
        timepoints = list(set(timepoints))

    for case in os.listdir(hdr_folder):

        case_info_json = {}

        for tp in timepoints:
            image_folder = os.path.join(hdr_folder, case, tp)
            
            if os.path.exists(image_folder):

                print(f'Case {case}, timepoint {tp}')

                image_names = {}
                for mdkw in modality_keywords:
                    for file in os.listdir(image_folder):
                        if mdkw in file and '.img' in file:
                            image_names[mdkw] = file
                
                for mdkw, img in image_names.items():
                    md = modality_names[modality_keywords.index(mdkw)]
                    new_image_name = f'{tp}_{md}_{case}{export_extension}'

                    image = sitk.ReadImage(os.path.join(image_folder, img)) 
                    sitk.WriteImage(image, os.path.join(destination_folder, 'Images', new_image_name))

                    case_info_json['image_source'] = os.path.join(image_folder, img)

                    info_json['cases'].append(case_info_json)
    
    with open(os.path.join(destination_folder, 'sourceinfo.json'), 'w') as json_file:
        json.dump(info_json, json_file, indent=0)

def transpose_sitk_image(image, transpose=(1,2,0)):
    origin = image.GetOrigin()
    spacing = np.array(image.GetSpacing())
        
    array = np.transpose(sitk.GetArrayFromImage(image), transpose)
    new_image = sitk.GetImageFromArray(array)
    idx = np.array(transpose, dtype=np.uint8)
    
    new_image.SetSpacing(spacing[idx])
    new_image.SetOrigin(origin)

    return new_image

def embed_feature_map_in_image_geometry(reference_image, feature_map):
    
    # Initialize blank version of reference image geometry
    embedded_fmap = sitk.Cast(0 * reference_image, sitk.sitkFloat32)

    # Hack to fix Pyradiomics bug where spacing and size don't match on feature maps
    transposed_fmap = transpose_sitk_image(feature_map, transpose=(2,0,1))
    transposed_fmap.SetSpacing(reference_image.GetSpacing())

    # Define the region within the reference image where feature map should be pasted
    fmap_origin_idx = reference_image.TransformPhysicalPointToIndex(feature_map.GetOrigin())
    fmap_size = transposed_fmap.GetSize()
    x0, x1 = fmap_origin_idx[1], fmap_origin_idx[1] + fmap_size[0]
    y0, y1 = fmap_origin_idx[2], fmap_origin_idx[2] + fmap_size[1]
    z0, z1 = fmap_origin_idx[0], fmap_origin_idx[0] + fmap_size[2]

    # Paste the feature map into the reference image geometry at the right positions
    embedded_fmap[x0:x1,y0:y1,z0:z1] = sitk.Cast(transposed_fmap, sitk.sitkFloat32)

    # Transpose one more time to fit image geometry
    embedded_fmap = transpose_sitk_image(embedded_fmap, transpose=(0,2,1))

    return embedded_fmap

def load_feature_set(
        features_dirpath='',
        mask_name='',
        timepoint='',
        modality='',
        suffix='',
        id_col='case_id',
        exclude_augmented_cases=False,
        drop_diagnostic_cols=True,
        drop_metadata_cols=False,
):
    feature_set_filename = '_'.join(filter(None, [timepoint, modality, mask_name, suffix])) + '.xlsx'
    feature_set_filepath = os.path.join(features_dirpath, feature_set_filename)

    df = pd.read_excel(feature_set_filepath)
    if id_col in df.columns:
        df = df.set_index(id_col)

    if exclude_augmented_cases:
        df = df[df['augmentation'] == 'aug_0']

    if drop_diagnostic_cols:
        df = df.loc[:, ~df.columns.astype(str).str.contains('diagnostics')]
    
    if drop_metadata_cols:
        df = df.drop(columns=CONSTANTS['metadata_columns'])

    return df

def load_feature_sets(
        features_dirpath='',
        mask_names=None,
        timepoints=None,
        modalities=None,
        suffix='',
        id_col='case_id',
        exclude_augmented_cases=False,
        drop_diagnostic_cols=True,
        drop_metadata_cols=False,
):
    df = pd.DataFrame()
    
    feature_sets = os.listdir(features_dirpath)

    if mask_names is None:
        mask_names = [f.split('_')[2] for f in feature_sets]
        mask_names = [m.replace('.xlsx', '') for m in mask_names]
        mask_names = sorted(list(set(mask_names)))

    if modalities is None:
        modalities = [f.split('_')[1] for f in feature_sets]
        modalities = [m.replace('.xlsx', '') for m in modalities]
        modalities = sorted(list(set(modalities)))

    if timepoints is None:
        timepoints = [f.split('_')[0] for f in feature_sets]
        timepoints = [m.replace('.xlsx', '') for m in timepoints]
        timepoints = sorted(list(set(timepoints)))

    for mask in mask_names:
        for tp in timepoints:
            for md in modalities:
                
                new_df = load_feature_set(
                    features_dirpath=features_dirpath,
                    mask_name=mask,
                    timepoint=tp,
                    modality=md,
                    suffix=suffix,
                    id_col=id_col,
                    exclude_augmented_cases=exclude_augmented_cases,
                    drop_diagnostic_cols=drop_diagnostic_cols,
                    drop_metadata_cols=drop_metadata_cols
                )

                df = pd.concat([df, new_df], axis=0)
    
    return df

def augmentation_average(df, col_id='case_id'):
    
    groupby_cols = [col_id]

    if len(np.unique(df['timepoint'])) > 1:
        groupby_cols.append('timepoint')
    
    if len(np.unique(df['modality'])) > 1:
        groupby_cols.append('modality')

    return df.groupby(groupby_cols).apply(lambda x: x.mean(numeric_only=True))

def select_feature_classes_and_filters(df,
                                       feature_classes=None,
                                       filter_types=None,
                                       additional_columns=[]
                                       ):
    
    feature_classes = feature_classes or ["shape", "firstorder", "glcm", "gldm", "glrlm", "glszm", "ngtdm"]
    feature_classes.extend(additional_columns)
    feature_classes_str =  '|'.join(feature_classes)

    filter_types = filter_types or ["original"]
    filter_types.extend(additional_columns)
    filter_types_str =  '|'.join(filter_types)
    
    df = df.filter(regex=feature_classes_str)
    df = df.filter(regex=filter_types_str)

    return df

def get_feature_set_statistics(df):
    df_num = df.select_dtypes(include=[np.number])
    means, stds, covs = df_num, df_num, df_num

    means = means.apply(lambda feature: np.mean(feature, numeric_only=True))
    stds = stds.apply(lambda feature: np.std(feature, ddof=1, numeric_only=True))
    covs = covs.apply(lambda feature: np.std(feature, ddof=1, numeric_only=True) / np.mean(feature) * 100, numeric_only=True)
    return {
        'mean' : means,
        'std' : stds,
        'cov' : covs
    }

def get_longitudinal_cov(df,
                        mask_name='',
                        modality='',
                        ):
        
        new_df = df
    
        if mask_name in new_df.columns:
            new_df = new_df[new_df['mask_name'] == mask_name]
        
        if modality in new_df.columns:
            new_df = new_df[new_df['modality'] == modality]

        return new_df.groupby(new_df.index).apply(lambda x : 100 * x.std(numeric_only=True) / abs(x.mean(numeric_only=True)))

def get_features_icc(df,
                     raters='timepoint',
                     mask_name='',
                     modality='',
                     icc_metric='icc',
                     id_col='case_id'
                     ):
    new_df = df
    
    if mask_name in new_df.columns:
        new_df = new_df[new_df['mask_name'] == mask_name]
    
    if modality in new_df.columns:
        new_df = new_df[new_df['modality'] == modality]

    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    keep_columns = numeric_columns + [raters]

    new_df = df
    new_df = new_df.loc[:, keep_columns]
    new_df = new_df.reset_index()
    
    ICCs = {}
    for feature in numeric_columns:
        icc_df = pg.intraclass_corr(data=new_df,
                                    targets=id_col,
                                    raters=raters,
                                    ratings=feature,
                                    nan_policy='omit'
                                    )
        
        if icc_metric == 'ci_lower_bound':
            icc_value = icc_df.loc[:,'CI95%'][0][0]
        else:
            icc_value = icc_df.loc[:,'ICC'][1]

        ICCs[feature] = abs(round(icc_value, 4))

    return ICCs    

def get_features_with_icc_above_cutoff(df,
                                       icc_cutoff,
                                       raters='timepoint',
                                       mask_name='',
                                       modality='',
                                       icc_metric='icc',
                                       id_col='case_id'
                                       ):
    iccs = get_features_icc(df, raters, mask_name, modality, 
                            icc_metric, id_col
                            )
    
    passing_features = [feature for feature, icc in iccs.items() if icc > icc_cutoff]
    return passing_features