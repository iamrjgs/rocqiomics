import os
import copy
import json

import numpy as np
import radiomics
import SimpleITK as sitk

from .obj_parser import AnalyzeObjectMap


class MaskProcessor:
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

    def create_peritumoral_mask(self, mask, dilation_radius=1, erosion_radius=1, dilation_mm=None, erosion_mm=None):
        mask = sitk.Cast(mask, sitk.sitkUInt8)

        dilate = sitk.BinaryDilateImageFilter()
        erode = sitk.BinaryErodeImageFilter()

        if dilation_mm is not None:
            dilation_radius = int(dilation_mm / np.min(mask.GetSpacing()))
        if erosion_mm is not None:
            erosion_radius = int(erosion_mm / np.min(mask.GetSpacing()))

        dilate.SetKernelRadius((dilation_radius, dilation_radius, 0))
        erode.SetKernelRadius((erosion_radius, erosion_radius, 0))

        dilated_mask = dilate.Execute(mask)
        eroded_mask = erode.Execute(mask)

        outer_rim = sitk.MaskImageFilter().Execute(dilated_mask, sitk.BinaryNot(mask))
        inner_rim = sitk.MaskImageFilter().Execute(mask, sitk.BinaryNot(eroded_mask))

        peritumoral_mask = outer_rim | inner_rim

        return sitk.Cast(peritumoral_mask, sitk.sitkUInt8)
    
    def get_centroid_coordinates(self, mask):
        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(mask)
        return stats.GetCentroid(1)

    def get_largest_connected_component(self, mask, label=1, return_as_binary=False):
        mask = mask == label
        cc = sitk.ConnectedComponent(mask)
        
        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(cc)
        
        largest_label = max(stats.GetLabels(), key=lambda l: stats.GetNumberOfPixels(l))
        largest = sitk.Cast(cc == largest_label, sitk.sitkUInt8)
        largest = largest * label if not return_as_binary else largest
        
        return largest
    
    def calculate_volume(self, mask, volume_type='MeshVolume'):
        dummy_image = sitk.Image(mask.GetSize(), sitk.sitkUInt8)
        dummy_image.CopyInformation(mask)
        shape_extractor = radiomics.shape.RadiomicsShape(dummy_image, mask)
        shape_extractor.enableFeatureByName(volume_type)
        result = shape_extractor.execute()
        return result[volume_type]
    
    def obj_to_mask(self, obj_path, index=0, reference_image=None):
        obj_map = AnalyzeObjectMap(obj_path)
        mask_arr = obj_map.get_data(index)
        mask_arr = mask_arr.astype(np.uint8)
        mask_arr = np.transpose(mask_arr, (2, 0, 1))
        
        mask = sitk.GetImageFromArray(mask_arr)

        if reference_image is not None:
            mask.CopyInformation(reference_image)

        return mask