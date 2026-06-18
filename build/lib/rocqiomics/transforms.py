import itk
import numpy as np

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