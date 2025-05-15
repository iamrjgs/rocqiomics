import os
import copy
import json

import numpy as np
import SimpleITK as sitk

from .obj_parser import AnalyzeObjectMap


class MaskProcesser:
    def __init__(self):
        pass

    def extract_binary_masks(self, mask, save_path='', metadata_format='mitk'):
        binary_masks = []

        if metadata_format == 'mitk':
            metadata_key = 'org.mitk.multilabel.segmentation.labelgroups'
            metadata = json.loads(mask.GetMetaData(metadata_key))
            labels = metadata[0]['labels']

            for label in labels:
                name = label['name']
                value = label['value']

                binary_mask = copy.deepcopy(mask)
                
                binary_mask[mask == value] = 1
                binary_mask[mask != value] = 0

                label['value'] = 1
                new_metadata = [{'labels':[label]}]

                [binary_mask.SetMetaData(key, mask.GetMetaData(key)) for key in mask.GetMetaDataKeys()]
                binary_mask.SetMetaData(metadata_key, json.dumps(new_metadata))

                binary_masks.append(binary_mask)

                if save_path:
                    sitk.WriteImage(binary_mask, os.path.join(save_path, f'{str(name)}.seg.nrrd'), imageIO='NrrdImageIO')

        return binary_masks
    
    def create_dilated_mask(self, mask, dilation_radius=1):
        dilate = sitk.BinaryDilateImageFilter()
        dilate.SetKernelRadius((dilation_radius, dilation_radius, 0))
        return dilate.Execute(mask)
    
    def create_eroded_mask(self, mask, erosion_radius=1):
        erode = sitk.BinaryErodeImageFilter()
        erode.SetKernelRadius((erosion_radius, erosion_radius, 0))
        return erode.Execute(mask)

    def create_peritumoral_mask(self, mask, dilation_radius=1, erosion_radius=1):
        mask = sitk.Cast(mask, sitk.sitkUInt8)

        dilate = sitk.BinaryDilateImageFilter()
        erode = sitk.BinaryErodeImageFilter()

        dilate.SetKernelRadius((dilation_radius, dilation_radius, 0))
        erode.SetKernelRadius((erosion_radius, erosion_radius, 0))

        dilated_mask = dilate.Execute(mask)
        eroded_mask = erode.Execute(mask)

        outer_rim = sitk.MaskImageFilter().Execute(dilated_mask, sitk.BinaryNot(mask))
        inner_rim = sitk.MaskImageFilter().Execute(mask, sitk.BinaryNot(eroded_mask))

        peritumoral_mask = outer_rim | inner_rim

        return sitk.Cast(peritumoral_mask, sitk.sitkUInt8)

    def obj_to_mask(self, obj_path, index=0, reference_image=None):
        obj_map = AnalyzeObjectMap(obj_path)
        mask_arr = obj_map.get_data(index)
        mask_arr = mask_arr.astype(np.uint8)
        mask_arr = np.transpose(mask_arr, (2, 0, 1))
        
        mask = sitk.GetImageFromArray(mask_arr)

        if reference_image is not None:
            mask.CopyInformation(reference_image)

        return mask