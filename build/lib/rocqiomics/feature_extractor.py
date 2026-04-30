import os
import logging

import numpy as np
import radiomics
from radiomics import getFeatureClasses

import SimpleITK as sitk

class FeatureExtractor:
    def __init__(self,
                engine='pyradiomics',
                label=1,
                voxel_based=False,
                voxel_based_settings={},
                feature_classes=['shape', 'firstorder', 'glcm', 'gldm', 'glrlm', 'glszm', 'ngtdm'],
                features=None,
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
        
        self.engine = engine

        if self.engine == 'pyradiomics':
            self.prepare_pyradiomics_extractor(
                label=label,
                voxel_based=voxel_based,
                voxel_based_settings=voxel_based_settings,
                feature_classes=feature_classes,
                features=features,
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
        extraction_functions = {
            'pyradiomics' : self.extract_pyradiomics_features
        }
        return extraction_functions[self.engine](image, mask)
    
    def prepare_pyradiomics_extractor(self,
                                    label=1,
                                    voxel_based=False,
                                    voxel_based_settings={},
                                    feature_classes=['shape', 'firstorder', 'glcm', 'gldm', 'glrlm', 'glszm', 'ngtdm'],
                                    features=None,
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
        self.all_features = self.get_all_pyradiomics_features()
        self.all_feature_classes = self.get_all_pyradiomics_feature_classes()

        """
        If the usual Pyradiomics yaml file is provided, those are the settings that will be used.
        This makes it easy to reproduce previous Pyradiomics workflows using this library.
        
        NOTE: Remember that we turn off all Pyradiomics preprocessing steps (e.g. intensity standardization) by default
        and instead use Monai transforms to implement these.
        If the YAML settings include image preprocessing steps such as normalization, these will be performed internally by Pyradiomics,
        so you should not include the Monai transforms.
        """
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

            # if voxel_based:
            #     params_dict['voxelBased'] = voxel_based_settings

            self.extractor._applyParams(paramsDict=params_dict)

            if features is not None:
                self.extractor.disableAllFeatures()
                enabled_features = {cl : [] for cl in self.all_feature_classes}
        
                for feature in features:
                    f = [x for x in self.all_features if feature in x][0]
                    fclass = f.split('_')[0]
                    enabled_features[fclass].append(feature)

                enabled_features = {k:v for k,v in enabled_features.items() if v}

                self.extractor.enableFeaturesByName(**enabled_features)

    def extract_pyradiomics_features(self, image, mask):
        return self.extractor.execute(image, mask, voxelBased=self.voxel_based)
    
    def get_all_pyradiomics_features(self):
        all_features = []
        all_classes = getFeatureClasses()

        for name, cl in all_classes.items():
            feature_names = list(cl.getFeatureNames().keys())
            feature_names = [f'{name}_{f}' for f in feature_names]
            all_features.extend(feature_names)

        return all_features      

    def get_all_pyradiomics_feature_classes(self):
        return list(getFeatureClasses().keys())      