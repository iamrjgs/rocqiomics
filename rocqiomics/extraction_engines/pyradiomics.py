import numpy as np
import SimpleITK as sitk

from monai.data import MetaTensor as MonaiMetaTensor

import radiomics
import radiomics.featureextractor
from radiomics import getFeatureClasses

from rocqiomics.utils import tensor_to_sitk
from .engine_base_class import FeatureExtractionEngine

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

