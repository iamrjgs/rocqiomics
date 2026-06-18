from abc import ABC, abstractmethod
import os
import logging

import numpy as np
import SimpleITK as sitk

import torch
from monai.data import MetaTensor as MonaiMetaTensor
from monai.utils.type_conversion import convert_to_tensor

try:
    import radiomics
    import radiomics.featureextractor
    from radiomics import getFeatureClasses
except:
    pass
try:
    import fastrad
    from fastrad import FeatureSettings, FeatureExtractor, VoxelFeatureExtractor, DenseFeatureExtractor
    from fastrad.image import MedicalImage, Mask
    from fastrad.io import _sitk_to_tensor
except:
    pass

from rocqiomics.utils import tensor_to_sitk

def MAP_ENGINE(name):
    engine_map = {
        'pyradiomics' : PyradiomicsExtractor,
        'fastrad' : FastradExtractor
    }
    if name.lower() not in list(engine_map.keys()):
        raise ValueError(f"Engine must be one of: {list(engine_map.keys())}. Value provided: {name}")
    return engine_map[name]

class FeatureExtractionEngine(ABC):
    def __init__(self,
                name=None,
                voxel_based=False,
                **kwargs
                ):
        self.name = name
        self.voxel_based = voxel_based
        self.extractor = self.prepare_extractor(**kwargs)

    @abstractmethod
    def preprocess(self, **kwargs):
        pass

    @abstractmethod
    def extract(self, **kwargs):
        pass

    @abstractmethod
    def postprocess(self, **kwargs):
        pass

    @abstractmethod
    def prepare_extractor(self, **kwargs):
        pass

class PyradiomicsExtractor(FeatureExtractionEngine):
    def __init__(self, name="pyradiomics", voxel_based=False, **kwargs):
        super().__init__(name=name, voxel_based=voxel_based, **kwargs)

    def preprocess(self, image, mask, **kwargs):
        if isinstance(image, MonaiMetaTensor) and isinstance(mask, MonaiMetaTensor):
            return tensor_to_sitk(image), tensor_to_sitk(mask)
        if isinstance(image, sitk.Image) and isinstance(mask, sitk.Image):
            return image, mask
        else:
            raise TypeError

    def prepare_extractor(
        self,
        label=1,
        voxel_based_settings=None,
        feature_classes=None,
        features=None,
        filter_types=None,
        filter_settings={"Original" : {}},
        extraction_settings_yaml_filepath=None,
        normalize=False,
        remove_outliers=None,
        bin_width=None,
        bin_count=None,
        resampled_pixel_size=None,
        interpolator="sitkBSpline",
        normalize_scale=1.0,
        correct_mask=False,
        pre_crop=False,
        pad_distance=5,
        force_2D=False,
        force_2D_dimension=0,
        resegment_range=None,
        additional_info=True,
        **kwargs,
    ):
        extractor = radiomics.featureextractor.RadiomicsFeatureExtractor()
        self.all_features = self.get_all_pyradiomics_features()
        self.all_feature_classes = self.get_all_pyradiomics_feature_classes()

        if extraction_settings_yaml_filepath is not None:
            extractor.loadParams(paramsFile=extraction_settings_yaml_filepath)
            return extractor

        feature_classes = self.all_feature_classes if feature_classes is None else feature_classes
        filter_types = ["Original"] if filter_types is None else filter_types

        feature_class_params = {fc: [] for fc in feature_classes}
        image_type_params = {ft: filter_settings.get(ft, {}) for ft in filter_types}

        params_dict = {
            "featureClass": feature_class_params,
            "imageType": image_type_params,
            "setting": {
                "label": label,
                "normalize": normalize,
                "normalizeScale": normalize_scale,
                "removeOutliers": remove_outliers,
                "resampledPixelSpacing": resampled_pixel_size,
                "interpolator": interpolator,
                "preCrop": pre_crop,
                "correctMask": correct_mask,
                "padDistance": pad_distance,
                "force2D": force_2D,
                "force2Ddimension": force_2D_dimension,
                "resegmentRange": resegment_range,
                "additionalInfo": additional_info,
            },
        }

        voxel_based_settings = voxel_based_settings if voxel_based_settings is not None else {}
        params_dict['voxelSetting'] = voxel_based_settings

        if bin_width is not None:
            params_dict["setting"]["binWidth"] = bin_width
        else:
            if bin_count is not None:
                params_dict["setting"]["binCount"] = bin_count

        extractor._applyParams(paramsDict=params_dict)

        if features is not None:
            extractor.disableAllFeatures()
            enabled = {cl: [] for cl in self.all_feature_classes}
            for feat in features:
                full = [f for f in self.all_features if feat in f][0]
                fclass = full.split("_")[0]
                enabled[fclass].append(feat)
            enabled = {k: v for k, v in enabled.items() if v}
            extractor.enableFeaturesByName(**enabled)

        return extractor

    def extract(self, image, mask, **kwargs):
        image, mask = self.preprocess(image, mask)
        results = self.extractor.execute(image, mask, voxelBased=self.voxel_based)
        return self.postprocess(results)

    def postprocess(self, results):
        return results

    @staticmethod
    def get_all_pyradiomics_features():
        out = []
        for name, cl in getFeatureClasses().items():
            for f in cl.getFeatureNames().keys():
                out.append(f"{name}_{f}")
        return out

    @staticmethod
    def get_all_pyradiomics_feature_classes():
        return list(getFeatureClasses().keys())

class FastradExtractor(FeatureExtractionEngine):
    def __init__(self, name="fastrad", voxel_based=False, **kwargs):
        super().__init__(name=name, voxel_based=voxel_based, **kwargs)

    def preprocess(self, image, mask, **kwargs):
        if isinstance(image, MonaiMetaTensor) and isinstance(mask, MonaiMetaTensor):
            image_t = convert_to_tensor(image)
            mask_t = convert_to_tensor(mask)

            xyz_to_zxy_reindex = [2,0,1]
    
            image_t = image_t.squeeze(0).permute(*xyz_to_zxy_reindex).contiguous()
            mask_t = mask_t.squeeze(0).permute(*xyz_to_zxy_reindex).contiguous()

            im_sp = image.pixdim[xyz_to_zxy_reindex]
            mask_sp = mask.pixdim[xyz_to_zxy_reindex]
      
            return MedicalImage(tensor=image_t, spacing=im_sp), Mask(tensor=mask_t, spacing=mask_sp)
        
        if isinstance(image, sitk.Image) and isinstance(mask, sitk.Image):
            image_t, sp_im = _sitk_to_tensor(image)
            mask_t, sp_mask = _sitk_to_tensor(mask)
            return MedicalImage(tensor=image_t, spacing=sp_im), Mask(tensor=mask_t, spacing=sp_mask)
        else:
            raise TypeError

    def prepare_extractor(
        self,
        label=1,
        voxel_based_settings={},
        feature_classes=None,
        features=None,
        filter_settings={"Original" : {}},
        extraction_settings_yaml_filepath=None,
        bin_width=None,
        resampled_pixel_size=None,
        interpolator='sitkBSpline',
        pad_distance=5,
        kernel_size=3,
        stride=1,
        force_2D=False,
        force_2D_dimension=0,
        compile=False,
        compile_mode='reduce_overhead',
        amp=False,
        differentiable=False,
        additional_info=True,
        **kwargs,
    ):  
        self.filter_settings = filter_settings
        self.kernel_size = kernel_size
        self.stride = stride

        settings = FeatureSettings(
            feature_classes=feature_classes,
            bin_width=bin_width,
            device="cuda",
            force2D=force_2D,
            force2Ddimension=force_2D_dimension,
            amp=amp,
            differentiable=differentiable,
        )
        return DenseFeatureExtractor(settings)

    def extract(self, image, mask, **kwargs):
        image, mask = self.preprocess(image, mask)
        filtered_images_dict = fastrad.apply_builtin_filters(image, self.filter_settings)
        
        results = {}
        for fname, fimg in filtered_images_dict.items():
            if self.voxel_based:
                filter_results = self.extractor.extract_dense(fimg, mask, kernel_size=self.kernel_size, stride=self.stride)
            else:
                filter_results =  self.extractor.extract(fimg, mask)

            filter_results = {f'{fname}:{k}':v for k,v in filter_results.items()}
            results = results | filter_results

        return self.postprocess(results)

    def postprocess(self, results):
        def cap(s):
            return s[:1].upper() + s[1:] if s[:1].isalpha() else s
            
        def pyradiomics_map_col(col):
            parts = col.split(':')
            filter_name = "original" if len(parts) == 2 else parts[0]
            feature = parts[-1]
            words = feature.split('_')
            words = [cap(w) for w in words]
            return f"{filter_name}_{parts[-2]}_{''.join(words)}"

        def map_result(v):
            if isinstance(v, (int, float)):
                return float(v)
            if isinstance(v, torch.Tensor):
                # return tensor_to_sitk(v)
                return v
            raise TypeError

        return {pyradiomics_map_col(k): map_result(v) for k, v in results.items()}
