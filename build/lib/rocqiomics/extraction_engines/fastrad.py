import numpy as np
import SimpleITK as sitk

import torch
from monai.data import MetaTensor as MonaiMetaTensor
from monai.utils.type_conversion import convert_to_tensor

import fastrad
from fastrad import FeatureSettings, FeatureExtractor, VoxelFeatureExtractor, DenseFeatureExtractor
from fastrad.image import MedicalImage, Mask
from fastrad.io import _sitk_to_tensor

from .engine_base_class import FeatureExtractionEngine

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
