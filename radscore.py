import os
import json 

import pandas as pd
import numpy as np
import SimpleITK as sitk

import rocqiomics as rq

def align_df_to_model(df, pipe):
    _, first_step = list(pipe.named_steps.items())[0]

    if hasattr(first_step, "get_feature_names_out"):
        expected_cols = first_step.get_feature_names_out()
    elif hasattr(first_step, "feature_names_in_"):
        expected_cols = first_step.feature_names_in_
    else:
        raise ValueError("First pipeline step does not expose feature names")

    df = df.copy()
    for col in expected_cols:
        if col not in df:
            df[col] = 0
    return df[expected_cols]

def calculate_radscores(df, model):
    radscores = {}
    df = align_df_to_model(df, model)
    for ptid in df.index.tolist():
        X = df.loc[[ptid]]
        radscores[ptid] = model.decision_function(X)[0]
    radscores = pd.DataFrame(radscores.items(), columns=['case_id', 'radscore']).set_index('case_id')
    return radscores

def generate_radscore_map(
        data_dict,
        coefficients_dict,
        scaling_data_dict,
        preprocessing=None,
        augmentations=[],
        bin_width=1.0,
        save_dirpath=None,
        extract_maps=False,
        resample_to_image=False,
        save_radscore_map=True
        ):
    
    image = sitk.ReadImage(data_dict['image'])
    metadata = data_dict['metadata']
    timepoint = metadata['timepoint']
    modality = metadata['modality']
    mask_name = metadata['mask_name']
    fmaps_save_dirpath = os.path.join(save_dirpath, 'Feature Maps', data_dict['case_id'])
    
    features = list(coefficients_dict.keys())
    coefficients = list(coefficients_dict.values())

    feature_names = ['_'.join(f.split('_')[2:]) for f in features]
    feature_names.remove('')

    if extract_maps:
        extraction = rq.Rocqiomics(
            data_dicts=[data_dict],
            preprocessing=preprocessing,
            augmentations=augmentations,
            features=feature_names,
            bin_width=bin_width,
            save_results=True,
            voxel_based=True,
            save_by_columns=['mask_name', 'modality', 'timepoint'],
            save_dirpath=save_dirpath
        )
        extraction.run_pipeline()

    fmap_arrays = []
    for feature in [f for f in features if 'intercept' not in f]:
        fmap_filename = f'{timepoint}_{modality}_{mask_name}_{feature}.nrrd'
        fmap_filepath = os.path.join(fmaps_save_dirpath, fmap_filename)
        fmap = sitk.ReadImage(fmap_filepath)
        fmap_arrays.append(sitk.GetArrayFromImage(fmap))

    means = scaling_data_dict['means'].reshape(len(scaling_data_dict['means']), 1, 1, 1)
    scales = scaling_data_dict['scales'].reshape(len(scaling_data_dict['scales']), 1, 1, 1)

    v = (np.array(fmap_arrays) - means) / scales
    dot = np.tensordot(coefficients[:-1], v, axes=(0,0))

    signature_map = sitk.GetImageFromArray(dot + coefficients[-1])
    signature_map.CopyInformation(fmap)
    signature_map.SetMetaData('coefficients', json.dumps(coefficients_dict))
    signature_map.SetMetaData('scaling_values' , json.dumps(scaling_data_dict))

    if resample_to_image:
        signature_map = rq.utils.resample_to_target_image(signature_map, image, is_mask=False)

    if save_radscore_map:
        sitk.WriteImage(os.path.join(fmaps_save_dirpath, 'Radscore.nrrd'))

    return signature_map, image
