from datetime import datetime
import os
import json
import logging
import time
from typing import List, Dict, Optional
import sys

import numpy as np
import pandas as pd
import SimpleITK as sitk

import monai
import torch

from rocqiomics.package_constants import PACKAGE_NAME
from rocqiomics.utils import (
    is2D,
    tensor_to_sitk,
    split_dataframe_by_unique_values_in_columns,
    resample_to_target_image
)


class Rocqiomics:

    def __init__(self,
                data_dicts: Optional[List[Dict]]=None,
                load_transform: Optional[monai.transforms.Transform]=None,
                preprocessing: Optional[monai.transforms.Transform]=None,
                augmentations: List[Optional[monai.transforms.Transform]]=[],
                voxel_based: bool=False,
                voxel_based_settings: Optional[Dict]=None,
                force_2D: bool=False,
                force_2D_dimension: int=0,
                bin_count: Optional[int]=None,
                bin_width: Optional[float]=25.0,
                feature_classes: List[str]=None,
                features: List[str]=None,
                filter_types: List[str]=None,
                filter_settings_by_type: Optional[Dict]=None,
                extraction_settings_yaml_filepath: Optional[str]=None,
                case_ids: Optional[List[str]]=None,
                case_limit: Optional[int]=None,
                engine: str="pyradiomics",
                device: str="cpu",
                logging_levels: Optional[Dict]=None,
                label: int=1,
                validate_inputs: bool=True,
                id_col: str="case_id",
                reader: str="ITKReader",
                save_results: bool=True,
                save_dirpath: Optional[str]=None,
                save_suffix: str="",
                save_results_to_existing_file: bool=False,
                save_by_columns: Optional[List[str]]=None,
                ):   
        
        """
        Pyradiomics-based medical image radiomics feature extraction with Monai-enabled preprocessing and augmentation.

        *** IMPORTANT ***
        This package disables all Pyradiomics preprocessing steps (intensity normalization, resampling, etc.) by default.
        We recommend these and any other preprocessing steps be implemented with monai transforms in the preprocessing keyword.
        The exception to this is the gray-level discretization step (whether fixed bin width or fixed bin size), which is 
        unavoidably performed by Pyradiomics at extraction time prior to extracting texture features. You can optionally turn 
        on Pyradiomics settings by providing a path to a Pyradiomics settings YAML file in `extraction_settings_yaml_filepath`.

        Parameters:
            data_dicts (Optional[List[Dict]]): List of dictionaries containing case data. Each data dict should have fields:
                - image (Union[str, Path]): Path to image [REQUIRED]
                - mask (Union[str, Path]): Path to segmentation mask [REQUIRED]
                - case_id (Optional[str]): String id assigned to case. Will be called whatever id_col is set as  [OPTIONAL]
                - metadata (Optional[Dict]): Dictionary containing additional metadata fields (e.g. label, modality, timepoint) [OPTIONAL]
            
            load_transform (Optional[monai.transforms.Transform]): Custom transform for image/mask loading. Leave as None
            to use default loading transform.

            preprocessing (Optional[monai.transforms.Transform]): Preprocessing transform, applied to each image prior to extraction.
            Use monai.transforms.Compose to concatenate multiple transforms.

            augmentations (List[Optional[monai.transforms.Transform]]): List of augmentation transforms; an additional image is created for each
            augmentation transform and each image (i.e. these increase dataset size). 
            Note: This adds an additional metadata field called `augmentation`

            voxel_based (bool): Whether to extract feature dataframes (False) or voxel-wise feature maps (True).

            bin_count (Optional[int]): Number of bins if performing gray-level discretization with fixed bin count (default: None)
            bin_width (Optional[float]): Width of bins if performing gray-level discretization with fixed bin size (default: 25.0).

            feature_classes (List[str]): List of radiomics feature classes to extract.
            filter_types (List[str]): Types of image filters applied before extraction.
            filter_settings_by_type (Dict): Configuration of filter settings per filter type.
            extraction_settings_yaml_filepath (Optional[str]): YAML file path with extraction parameters. IMPORTANT: IF YOU USE THIS, THE YAML FILE SETTINGS WILL OVERRIDE ALL OTHER SETTINGS.

            case_ids (Optional[List[str]]): List of case_ids you want to filter by. Leave as None if extracting from all cases.
            case_limit (Optional[int]): Maximum number of cases to process.

            engine (str): Feature extraction engine (only Pyradiomics for now).
            device (str): Processing device ("cpu" or "gpu").

            logging_levels (Optional[Dict]): Logging settings for pipeline and extraction. 
            Example: {'pipeline' : logging.INFO, 'extraction' : logging.WARNING}

            label (int): Segmentation label ID used for feature extraction.
            validate_inputs (bool): Whether to validate input data is valid.
            id_col (str): Identifier column in metadata ("case_id" by default).
            reader (str): Monai image reader type ("ITKReader" by default).

            save_results (bool): Whether to save extracted results.
            save_dirpath (Optional[str]): Directory path for saving outputs.
            save_suffix (str): Suffix for saved filenames. For example, setting as 'test' appends '_test' to each saved file.
            save_results_to_existing_file (bool): Whether to append results to an existing file.
            save_by_columns (Optional[List[str]]): List of metadata columns by which to separate feature sets for saving as Excel file.
            For example, setting as ['modality', 'timepoint'] will save a different feature set for each modality and timepoint
            pair. Leave as None to save all extraction results to a single Excel file.
        """

        # Initialize pipeline data containers
        self.results: List[Dict] = []
        self.excluded_cases: List[str] = []
        self.runtime_errors: List[str] = []

        # Set pipeline settings
        self.label: int = label
        self.id_col: str = id_col
        self.device: torch.device = self._set_device(device)
        
        # Initialize logging
        self.logging_levels: Optional[Dict] = logging_levels or {'pipeline' : logging.INFO, 'extractor' : logging.ERROR}
        self.logger = self._set_loggers()

        # Set settings for results saving
        self.save_results: bool = save_results
        self.save_by_columns: List[str] = save_by_columns
        self.save_suffix: str = save_suffix
        self.save_dirpath = save_dirpath or os.path.join(os.getcwd(), "Results")
        self.save_results_to_existing_file: bool = save_results_to_existing_file
        
        # Set monai transforms for loading, preprocessing, and augmentation
        self.load_transform = load_transform or self._default_load_transform(reader=reader)
        self.preprocessing = preprocessing
        self.augmentations = augmentations
            
        # Validate and set list of input data dicts
        self.data_dicts, self.excluded_cases = self._initialize_data_dicts(
            data_dicts=data_dicts,
            case_ids=case_ids,
            case_limit=case_limit,
            validate_inputs=validate_inputs
        )
        
        # Set monai Dataset to handle loading, augmentation, and preprocessing
        from rocqiomics.dataset import AugmentedDataset
        self.dataset = AugmentedDataset(
            data=self.data_dicts,
            load_transform=self.load_transform,
            preprocessing=self.preprocessing,
            augmentations=self.augmentations
        )

        # Set extraction settings
        self.engine: str = engine
        self.voxel_based: bool = voxel_based
        self.voxel_based_settings = voxel_based_settings
        self.force_2D = force_2D
        self.force_2D_dimension = force_2D_dimension
        self.bin_count: Optional[int] = bin_count
        self.bin_width: Optional[float] = bin_width
        self.filter_types: List[str] = filter_types or ["Original"]
        self.filter_settings_by_type: Dict = filter_settings_by_type or {}
        self._set_feature_classes(feature_classes)
        self.features = features

        # Set radiomics feature extractor
        from rocqiomics.feature_extractor import FeatureExtractor
        self.extractor = FeatureExtractor(
            engine=self.engine,
            label=self.label,
            voxel_based=self.voxel_based,
            voxel_based_settings=self.voxel_based_settings,
            force_2D=self.force_2D,
            force_2D_dimension=self.force_2D_dimension,
            bin_width=self.bin_width,
            feature_classes=self.feature_classes,
            features=self.features,
            filter_types=self.filter_types,
            filter_settings_by_type=self.filter_settings_by_type,
            extraction_settings_yaml_filepath=extraction_settings_yaml_filepath,
        )

    def __len__(self):
        return len(self.dataset)
    
    def _run_case(self, idx, case):
        start_time = time.time()

        # Get loaded, preprocessed, and (potentially) augmented data
        case_id, image, mask, metadata = [case.get(x) for x in [self.id_col, "image", "mask", "metadata"]]
    
        # Convert tensors to SimpleITK images expected by Pyradiomics
        image, mask = tensor_to_sitk(image), tensor_to_sitk(mask)
        
        # Extract feature vector or map depending on voxel_based extraction mode
        extraction_results = self._extract_features(image, mask)

        # Handle results metadata addition and/or saving depending on voxel_based extraction mode
        if self.voxel_based:
            self._handle_feature_map(case_id, extraction_results, metadata, image)
        else:
            self._handle_feature_vectors(case_id, extraction_results, metadata)

        # Log results
        self._log_case_data(idx, case, start_time)
        
    def run_pipeline(self):

        self.logger.info(f'Pipeline Initialized | Cases: {len(self)} | Excluded cases: {len(self.get_excluded_cases())}')

        for idx, case in enumerate(self.dataset):
            try:
                self._run_case(idx, case)
            except Exception as e:
                self._handle_case_error(case=case, error=e)

        # Handle results saving when not extracting feature maps (those are handled separately)
        if not self.voxel_based and self.save_results:
            self.save_results_df()

        return self.get_results()
    
    def get_data_dicts(self):
        return self.data_dicts

    def get_excluded_cases(self):
        return self.excluded_cases
                    
    def get_results(self):
        if self.voxel_based:
            return self.results
        
        df = pd.DataFrame(self.results)
        if self.id_col in df.columns:
            df = df.set_index(self.id_col)

        return df

    def save_results_df(self):
        save_filepath = ''
        results_df = self.get_results()

        # Create results directory if necessary
        os.makedirs(self.save_dirpath, exist_ok=True)

        # If results will be saved all to one file
        if self.save_by_columns is None:
            save_filepath = self._get_save_filepath(include_current_date=True)
            self.save_or_update_feature_set(results_df, save_filepath)
        
        # If results will be saved across multiple files by unique metadata column values (e.g. by timepoint, modality, etc.)
        else:
            unique_vals_dicts = split_dataframe_by_unique_values_in_columns(results_df, self.save_by_columns)
            
            for d in unique_vals_dicts:
                save_column_values = [d[key] for key in self.save_by_columns]

                save_filepath = self._get_save_filepath(
                    save_column_values=save_column_values,
                    include_current_date=False
                )

                self.save_or_update_feature_set(d['df'], save_filepath)

        return save_filepath
        
    def save_results_maps(self,
                          maps=None,
                          case_id=None,
                          case_data=None
                          ):
        for feature, fmap in maps.items():
            save_dir = os.path.join(self.save_dirpath, 'Feature Maps', case_id)
            os.makedirs(save_dir, exist_ok=True)

            save_filepath = self._get_save_filepath_feature_maps(
                save_dir=save_dir,
                feature=feature,
                metadata=case_data,
                include_current_date=False
            )
            sitk.WriteImage(fmap, save_filepath)

        self.logger.info(f'Feature maps saved to {save_dir}')

        with open(os.path.join(save_dir, 'extraction_metadata.json'), "w") as fp:
            json.dump(case_data, fp) 

    def save_or_update_feature_set(self, df, filepath):
        if self.save_results_to_existing_file and os.path.exists(filepath):
            existing_df = pd.read_excel(filepath)

            if self.id_col in existing_df.columns:
                existing_df = existing_df.set_index(self.id_col)

            updated_df = pd.concat([existing_df, df], axis=0)
            updated_df = updated_df.sort_index()

            updated_df.to_excel(filepath)

            self.logger.info(f'Results added to existing feature set {filepath}')
        else:
            df.to_excel(filepath)
            self.logger.info(f'Results saved to {filepath}')

    def _handle_feature_vectors(self, case_id, feature_vect, metadata=None):
        # Add case metadata to feature vector
        feature_vect[self.id_col] = case_id
        if metadata is not None:
            for key, mdata in metadata.items():
                feature_vect[key] = mdata
        
        if self.preprocessing:
            feature_vect['diagnostics_preprocessing'] = str(self._stringify_transforms(self.preprocessing))
        if self.augmentations:
            feature_vect['diagnostics_augmentations'] = str(self._stringify_transforms(self.augmentations))

        self.results.append(feature_vect)

    def _handle_feature_map(self, case_id, feature_vect, metadata=None, original_image=None):
        # Separate feature maps from diagnostic data in results
        maps = {k:v for k,v in feature_vect.items() if isinstance(v, sitk.SimpleITK.Image)}
        case_data = {k:v for k,v in feature_vect.items() if k not in maps.keys()}
        
        # Resample feature map to have the same geometry as the original image
        if original_image is not None:
            maps = {k:resample_to_target_image(v, original_image) for k,v in maps.items()}

        # Add case metadata to diagnostic data
        case_data[self.id_col] = case_id
        if metadata is not None:
            for key, mdata in metadata.items():
                case_data[key] = mdata

        # Only add case data to results, as adding all feature maps for all cases would quickly exceed memory constraints
        self.results.append(case_data)

        # If enabled, save feature maps to disk
        if self.save_results:
            self.save_results_maps(
                maps=maps,
                case_id=case_id,
                case_data=case_data
            )

    def _initialize_data_dicts(self,
                               data_dicts: Optional[List[Dict]],
                               case_ids: Optional[List[str]],
                               case_limit: Optional[int],
                               validate_inputs: bool
                               ) -> List[Dict]:
        if data_dicts is None:
            return [], []
        
        excluded_cases = []
        
        # If list of case_ids given, only preserve data_dicts with those case_ids
        if case_ids:
            data_dicts = [d for d in data_dicts if d.get(self.id_col) in case_ids]
        
        # If limit N of cases to consider given, only choose the top N cases
        if case_limit is not None:
            data_dicts = data_dicts[:case_limit]

        # If desired, validate data prior to extraction and only retain data_dicts that pass all tests
        if validate_inputs:
            self.logger.info('Validating images and masks...')
            data_dicts, excluded_cases = self._validate_input_data_and_exclude_cases_with_errors(data_dicts)

        return data_dicts, excluded_cases

    def _validate_input_data_and_exclude_cases_with_errors(self, data_dicts):
        excluded_cases = []
        tests = self._get_input_validation_tests()

        for case in data_dicts:

            for test_name, test_func in tests.items():
                test_result = test_func(case)

                # If test fails, error message saved in test_result
                if len(test_result) > 0:
                    excluded_cases.append({
                        'case' : case,
                        'failed_test' : test_name,
                        'failed_test_result' : test_result
                    })
                    data_dicts.remove(case)
                    break

        return data_dicts, excluded_cases
 
    def _get_input_validation_tests(self):
        return {
            'key_existence' : self._key_existence_tests,
            'file_existence' : self._file_existence_tests,
            'file_validity' : self._file_validity_tests
        }

    def _key_existence_tests(self, case):
        keys = list(case.keys())
        
        key_tests = {
            'case_id key exists' : self.id_col in keys,
            'image key exists' : 'image' in keys,
            'mask key exists' : 'mask' in keys,
        }

        return [key for key, value in key_tests.items() if not value]
    
    def _file_existence_tests(self, case):
        file_existence_tests = {
            'image file exists' : os.path.exists(case['image']),
            'mask file exists' : os.path.exists(case['mask']),
        }

        return [key for key, value in file_existence_tests.items() if not value]

    def _file_validity_tests(self, case):
        mask = sitk.GetArrayFromImage(sitk.ReadImage(case['mask']))
        
        file_validity_tests = {
            'mask contains labels' : len(np.unique(mask)) > 1,
            'mask has multiple voxels' : len(mask[mask == self.label]) > 1
        }

        return [key for key, value in file_validity_tests.items() if not value]

    def _extract_features(self, image, mask):
        return self.extractor.transform(image, mask)

    def _get_save_filepath(self, 
                           save_column_values=None, 
                           include_current_date=False
                           ):
        save_filepath = ''
        save_filename_pieces = [self.save_suffix]

        if include_current_date:
            save_filename_pieces.insert(0, datetime.today().strftime('%m%d%y'))

        if save_column_values is None:
            save_filename_pieces.insert(0, 'AllFeatures')
        else:
            for p in save_column_values:
                save_filename_pieces.insert(0, p)
        
        # Join filename pieces by underscores to create name after filtering empty strings
        save_filename = '_'.join(filter(None, save_filename_pieces)) + '.xlsx'
        
        save_filepath = os.path.join(self.save_dirpath, save_filename)
        return save_filepath
    
    def _get_save_filepath_feature_maps(self,
                                        save_dir,
                                        feature,
                                        metadata,
                                        include_current_date=False
                                        ):
        save_filename_pieces = []
        for save_column in self.save_by_columns:
            if save_column in list(metadata.keys()):
                save_filename_pieces.insert(0, metadata[save_column])

        if include_current_date:
            save_filename_pieces.append(datetime.today().strftime('%m%d%y'))

        # Join filename pieces by underscores to create name after filtering empty strings
        save_filename_pieces.append(feature)
        save_filename = '_'.join(filter(None, save_filename_pieces)) + '.nrrd'

        return os.path.join(save_dir, save_filename)

    def _default_load_transform(self, reader) -> monai.transforms.Transform:
        return monai.transforms.LoadImaged(keys=['image', 'mask'], 
                        image_only=False, 
                        ensure_channel_first=True, 
                        reader=reader
                        )

    def _set_device(self, device):
        return torch.device('cuda' if (torch.cuda.is_available() and (device == 'cuda')) else 'cpu')

    def _set_feature_classes(self, feature_classes=None):
        classes = feature_classes or ["shape", "firstorder", "glcm", "gldm", "glrlm", "glszm", "ngtdm"]
        # Handle shape case when image is 2D
        if "shape" in classes:
            if is2D(sitk.ReadImage(self.data_dicts[0]['image'])) or self.force_2D:
                classes = ["shape2D" if f == "shape" else f for f in classes]
        self.feature_classes = classes

    def _handle_case_error(self, case, error):
        self.logger.error(f'Runtime error - {case[self.id_col]} - {repr(error)}')
        self.runtime_errors.append({
            case[self.id_col] : repr(error)
        })

    def _set_loggers(self):
        pipeline_logging_level = self.logging_levels.get('pipeline', logging.INFO)
        extraction_logging_level = self.logging_levels.get('extractor', logging.WARNING)

        # Set pipeline logging settings
        logger_obj = logging.getLogger(PACKAGE_NAME)
        if logger_obj.hasHandlers():
            logger_obj.handlers.clear()
        logger_obj.setLevel(pipeline_logging_level)
        console_handler = logging.StreamHandler(stream=sys.stdout)
        console_handler.setLevel(pipeline_logging_level) 
        formatter = logging.Formatter("%(name)s %(levelname)s:\t %(message)s")
        console_handler.setFormatter(formatter)
        logger_obj.addHandler(console_handler)

        # Override logging settings for Pyradiomics loggers
        for name in ['radiomics', 
                     'radiomics.generalinfo',
                     'radiomics.featureextractor', 
                     'radiomics.imageoperations',
                     'radiomics.shape',
                     'radiomics.firstorder',
                     'radiomics.glcm',
                     'radiomics.gldm',
                     'radiomics.glrlm',
                     'radiomics.glrlm',
                     'radiomics.glszm',
                     'radiomics.ngtdm',
                     ]:
            pyradiomics_format = "Pyradiomics %(levelname)s:\t %(message)s"
            pyradiomics_logger = logging.getLogger(name)
            if pyradiomics_logger.hasHandlers():
                pyradiomics_logger.handlers.clear()
            pyradiomics_logger.setLevel(extraction_logging_level)
            console_handler = logging.StreamHandler(stream=sys.stdout)
            console_handler.setLevel(extraction_logging_level) 
            formatter = logging.Formatter(pyradiomics_format)
            console_handler.setFormatter(formatter)
            pyradiomics_logger.addHandler(console_handler)

        return logger_obj
    
    def _log_case_data(self, idx, case, start_time):
        log_data = []
        last_idx = len(self) - 1

        if self.id_col in case.keys():
            log_data.append(f'{self.id_col}: {case[self.id_col]}')
        
        if 'metadata' in case.keys():
            mdata = case['metadata']

            for md in mdata.keys():
                log_data.append(f'{md}: {str(mdata[md])}')

        log_txt = '\t'.join(log_data)
        
        run_time = time.time() - start_time
        self.logger.info(f'Case {idx}/{last_idx} done in {run_time:.2f}s\t{log_txt}')
        self.logger.debug(case)

    def _stringify_value(self, v):
        if isinstance(v, (int, float, str, bool, type(None))):
            return repr(v)

        if isinstance(v, (list, tuple)):
            inner = ", ".join(self._stringify_value(x) for x in v)
            return f"[{inner}]" if isinstance(v, list) else f"({inner})"

        if isinstance(v, dict):
            inner = ", ".join(f"{k}={self._stringify_value(val)}"
                            for k, val in v.items()
                            if not k.startswith("_"))
            return f"{{{inner}}}"

        if hasattr(v, "__dict__"):
            cls = v.__class__.__name__
            params = ", ".join(
                f"{k}={self._stringify_value(val)}"
                for k, val in v.__dict__.items()
                if not k.startswith("_")
            )
            return f"{cls}({params})"

        return repr(v)

    def _stringify_transforms(self, obj):
        if hasattr(obj, "transforms"):
            return [self._stringify_value(t) for t in obj.transforms]

        if isinstance(obj, (list, tuple)):
            return [self._stringify_value(t) for t in obj]

        return [self._stringify_value(obj)]

