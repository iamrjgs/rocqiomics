import os
import logging

import numpy as np
import radiomics

import SimpleITK as sitk

class FeatureExtractor:
    def __init__(self,
                engine='pyradiomics',
                label=1,
                voxel_based=False,
                voxel_based_settings={},
                feature_classes=['shape', 'firstorder', 'glcm', 'gldm', 'glrlm', 'glszm', 'ngtdm'],
                filter_types=['Original'],
                filter_settings_by_type={},
                extraction_settings_yaml_filepath=None,
                logging_level=logging.INFO,
                normalize=False,
                remove_outliers=None,
                bin_width=None,
                bin_count=None,
                resampled_pixel_size=None,
                interpolator='sitkBSpline',
                normalize_scale=1.0,
                correct_mask=False,
                pre_crop=False,
                pad_distance=5,
                force_2D=False,
                force_2D_dimension=0,
                resegment_range=None,
                additional_info=True,
                ):
        
        self.engine = engine

        if self.engine == 'pyradiomics':
            self.prepare_pyradiomics_extractor(
                label=label,
                voxel_based=voxel_based,
                voxel_based_settings=voxel_based_settings,
                feature_classes=feature_classes,
                filter_types=filter_types,
                filter_settings_by_type=filter_settings_by_type,
                extraction_settings_yaml_filepath=extraction_settings_yaml_filepath,
                normalize=normalize,
                remove_outliers=remove_outliers,
                bin_width=bin_width,
                bin_count=bin_count,
                resampled_pixel_size=resampled_pixel_size,
                interpolator=interpolator,
                normalize_scale=normalize_scale,
                correct_mask=correct_mask,
                pre_crop=pre_crop,
                pad_distance=pad_distance,
                force_2D=force_2D,
                force_2D_dimension=force_2D_dimension,
                resegment_range=resegment_range,
                additional_info=additional_info,
            )

    def transform(self, image=None, mask=None):
        if self.engine == 'pyradiomics':
            return self.extractor.execute(image, mask, voxelBased=self.voxel_based)
        return
    
    def prepare_pyradiomics_extractor(self,
                                    label=1,
                                    voxel_based=False,
                                    voxel_based_settings={},
                                    feature_classes=['shape', 'firstorder', 'glcm', 'gldm', 'glrlm', 'glszm', 'ngtdm'],
                                    filter_types=['Original'],
                                    filter_settings_by_type={},
                                    extraction_settings_yaml_filepath=None,
                                    normalize=False,
                                    remove_outliers=None,
                                    bin_width=None,
                                    bin_count=None,
                                    resampled_pixel_size=None,
                                    interpolator='sitkBSpline',
                                    normalize_scale=1.0,
                                    correct_mask=False,
                                    pre_crop=False,
                                    pad_distance=5,
                                    force_2D=False,
                                    force_2D_dimension=0,
                                    resegment_range=None,
                                    additional_info=True,
                                    ):
        
        self.voxel_based = voxel_based

        self.extractor = radiomics.featureextractor.RadiomicsFeatureExtractor()

        if extraction_settings_yaml_filepath is not None:
            self.extractor.loadParams(paramsFile=extraction_settings_yaml_filepath)
        
        else:
            feature_class_params_dict = {fc : [] for fc in feature_classes}
            filter_params_dict = {ft : {} for ft in filter_types}

            for filter, filter_settings in filter_settings_by_type.items():
                filter_params_dict[filter] = filter_settings

            params_dict = {
                'featureClass' : feature_class_params_dict,
                'imageType' : filter_params_dict,
                'setting': {
                    'label' : label,
                    'normalize' : normalize,
                    'normalizeScale' : normalize_scale,
                    'removeOutliers' : remove_outliers,
                    'resampledPixelSpacing' : resampled_pixel_size,
                    'interpolator' : interpolator,
                    'preCrop' : pre_crop,
                    'correctMask' : correct_mask,
                    'padDistance' : pad_distance,
                    'force2D' : force_2D,
                    'force2Ddimension' : force_2D_dimension,
                    'resegmentRange' : resegment_range,
                    'additionalInfo' : additional_info,
                },
            }

            if bin_width is not None:
                params_dict['setting']['binWidth'] = bin_width
            if bin_count is not None:
                params_dict['setting']['binCount'] = bin_count

            if voxel_based:
                params_dict['voxelSetting'] = voxel_based_settings

            self.extractor._applyParams(paramsDict=params_dict)