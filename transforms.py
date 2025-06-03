import itk
import numpy as np
import torch

import monai
from monai.data import itk_torch_bridge

class N4ITKBiasFieldCorrection(monai.transforms.MapTransform):
    def __init__(self,
                 image_key='image', 
                 mask_key='mask',
                 max_iterations=50,
                 number_of_fitting_levels=1,
                 number_of_histogram_bins=None,
                 convergence_threshold=None,
                 spline_order=None
                 ):
        self.image_key=image_key
        self.mask_key=mask_key
        self.number_of_fitting_levels = number_of_fitting_levels
        self.number_of_histogram_bins = number_of_histogram_bins
        self.convergence_threshold = convergence_threshold
        self.spline_order = spline_order
        
        self.max_iterations = [max_iterations for i in range(number_of_fitting_levels)]

    def __call__(self, data):
        img = data[self.image_key]
        mask = data[self.mask_key]

        img = itk_torch_bridge.metatensor_to_itk_image(img, dtype=np.float32)
        itk_mask = itk_torch_bridge.metatensor_to_itk_image(mask, dtype=np.uint8)

        ImgType = type(img)
        MaskType = type(itk_mask)

        n4 = itk.N4BiasFieldCorrectionImageFilter[ImgType, MaskType, ImgType].New(img, itk_mask)
   
        # Set correction settings for each setting with a non-null value
        n4_config_params = {
            "SetNumberOfFittingLevels": self.number_of_fitting_levels,
            "SetNumberOfHistogramBins": self.number_of_histogram_bins,
            "SetConvergenceThreshold": self.convergence_threshold,
            "SetSplineOrder": self.spline_order,
            "SetMaximumNumberOfIterations" : self.max_iterations
        }
        for setter_name, setting_value in n4_config_params.items():
            if setting_value is not None:
                getattr(n4, setter_name)(setting_value)
        
        n4.Update()
        corrected_img = n4.GetOutput()
       
        data[self.image_key] = itk_torch_bridge.itk_image_to_metatensor(corrected_img)

        return data
    

class IntensityDiscretizationBinWidth(monai.transforms.MapTransform):
    def __init__(self,
                 image_key='image', 
                 mask_key=None, 
                 bin_width=1.0, 
                 min_intensity=None,
                 ):
        self.image_key = image_key
        self.mask_key = mask_key
        self.bin_width = float(bin_width)
        self.min_intensity = min_intensity

    def __call__(self, data):
        image = data[self.image_key]
        image_array = image.numpy()
        
        mask_array = np.ones(image_array.shape, dtype=bool) if self.mask_key is None else data[self.mask_key].numpy().astype(bool)
        min_intensity = np.min(image_array[mask_array]) if self.min_intensity is None else self.min_intensity
        min_intensity -= min_intensity % self.bin_width

        image_array = np.floor((image_array - min_intensity) / self.bin_width) + 1.0

        image_array[image_array <= 1.0] = 1.0
        max_disc_intensity = np.max(image_array[mask_array])
        image_array[image_array >= max_disc_intensity] = max_disc_intensity
        
        data[self.image_key] = monai.data.MetaTensor(image_array, meta=image.meta, dtype=torch.float32)
        
        return data
    

class IntensityDiscretizationBinCount(monai.transforms.MapTransform):
    def __init__(self,
                 image_key='image', 
                 mask_key='mask', 
                 bin_count=100, 
                 use_mask_intensity_range=True,
                 ):

        self.image_key = image_key
        self.mask_key = mask_key
        self.bin_count = float(bin_count)
        self.use_mask_intensity_range = use_mask_intensity_range

    def __call__(self, data):
        image = data[self.image_key]
        mask = data[self.mask_key]

        image_array = image.numpy()
        mask_array = mask.numpy()

        a = image_array[mask_array] if self.use_mask_intensity_range else image_array
        min_intensity, max_intensity = np.min(a), np.max(a)

        image_array = np.floor( self.bin_count * (image_array - min_intensity) / (max_intensity - min_intensity) ) + 1.0
        image_array[image_array < 1.0] = 1.0
        image_array[image_array >= self.bin_count] = self.bin_count
        
        data[self.image_key] = monai.data.MetaTensor(image_array, meta=image.meta, dtype=torch.float32)

        return data