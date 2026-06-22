from itertools import islice
import os
import pickle
import warnings

import numpy as np
import SimpleITK as sitk
import rocqiomics as rq

from .habitat_generator import HabitatGenerator

class RadiomicsHabitatGenerator:
    def __init__(self,
                 preprocessing=None,
                 augmentations=None,
                 features=None,
                 filter_types=None,
                 algorithm='kmeans',
                 n_clusters=3,
                 batch_size=50,
                 bin_width=25.0,
                 bin_count=None,
                 engine='pyradiomics',
                 voxel_based_settings=None,
                 save_fmaps_dirpath=None,
                 save_vector_dirpath=None,
                 average_augmentations=False,
                 include_spatial_features=False
                 ):
        self.preprocessing = preprocessing
        self.augmentations = augmentations if augmentations is not None else []
        self.features = features or ['Mean']
        self.filter_types = filter_types if filter_types is not None else ['Original']
        self.algorithm_name = algorithm
        self.bin_width = bin_width
        self.bin_count = bin_count
        self.batch_size = batch_size
        self.n_clusters = n_clusters
        self.engine = engine
        self.save_vector_dirpath = save_vector_dirpath
        self.save_fmaps_dirpath = save_fmaps_dirpath
        self.average_augmentations = average_augmentations
        self.include_spatial_features = include_spatial_features

        if voxel_based_settings is None:
            self.voxel_based_settings = {
                'kernelRadius' : 3,
                'maskedKernel' : False,
                'initValue' : 0.0,
            }   
        else:
            self.voxel_based_settings = voxel_based_settings

        # These are set from the data
        self.channels = None
        self.habitat_generator = None

        self._init_map_extractor()
        self.logger = self.map_extractor.logger
        self.logger.info('Radiomics Habitat Generation Pipeline Initialized.')

    @staticmethod
    def filter_feature_class_name(k):
        tokens = k.split('_')
        return tokens[0] + '_' + tokens[-1]
    
    def _init_map_extractor(self):
        self.map_extractor = rq.Rocqiomics(
            preprocessing=self.preprocessing,
            augmentations=self.augmentations,
            bin_width=self.bin_width,
            bin_count=self.bin_count,
            voxel_based=True,
            voxel_based_settings=self.voxel_based_settings,
            features=self.features,
            filter_types=self.filter_types,
            cache_feature_maps=True,
            save_results=self.save_fmaps_dirpath is not None,
            save_dirpath=self.save_fmaps_dirpath
        )

    def _init_habitat_generator(self):
        self.habitat_generator = HabitatGenerator(
            channels=self.channels,
            algorithm=self.algorithm_name,
            n_clusters=self.n_clusters,
            include_spatial_features=self.include_spatial_features
        )
    
    def _check_or_set_channels(self, map_dict):
        map_dict = {self.filter_feature_class_name(k):v for k,v in map_dict.items()}
        items = sorted(map_dict.items())

        if not items:
            warnings.warn("No features extracted for sample.")

        feature_names, fmaps = zip(*items)
        feature_names, fmaps = list(feature_names), list(fmaps)

        if self.channels is None:
            self.channels = feature_names
            self._init_habitat_generator()
        else:
            if feature_names != self.channels:
                raise ValueError(
                    f"Inconsistent channels detected.\n"
                    f"Expected: {self.channels}\n"
                    f"Got: {feature_names}"
                )
        return feature_names, fmaps

    @staticmethod
    def _batch_generator(generator, k):
        # Loads outputs from generators in batches of size k.
        # Used to load all augmented copies of an image in one step when using augmentations
        iterator = iter(generator)
        while True:
            # Pull next k elements
            batch = tuple(islice(iterator, k))
            if not batch:
                break
            yield batch

    def _save_vector_image(self, vector_img, metadata):
        if self.save_vector_dirpath is None:
            raise ValueError("save_vector_dirpath must be set")
        
        meta_to_use = {k:str(v) for k,v in metadata.items() if 'path' not in k and 'diagnostics' not in k and 'augmentation' not in k}
        filename_parts = [l for l in list(meta_to_use.values()) if len(l) > 0][0:6] # Limit to 6 metadata fields

        if 'augmentation' in metadata and not self.average_augmentations:
            filename_parts.append(str(metadata['augmentation']))

        filename_parts = filename_parts + ['averaged'] if self.average_augmentations else filename_parts
        metadata_str = "_".join(filename_parts)
        savename = f"{metadata.get(self.map_extractor.id_col, '')}_{metadata_str}.nrrd"
        savepath = os.path.join(self.save_vector_dirpath, savename)

        os.makedirs(self.save_vector_dirpath, exist_ok=True)
        sitk.WriteImage(vector_img, savepath)

        return savepath
    
    def _save_habitat_map(self, habitat_map, data_dict, save_habitats_dirpath):
        meta_to_use = {k:str(v) for k,v in data_dict['metadata'].items() if 'path' not in k and 'diagnostics' not in k and 'augmentation' not in k}
        filename_parts = [l for l in list(meta_to_use.values()) if len(l) > 0][0:6] # Limit to 6 metadata fields

        if 'augmentation' in data_dict['metadata'] and not self.average_augmentations:
            filename_parts.append(str(data_dict['metadata']['augmentation']))

        filename_parts = filename_parts + ['averaged'] if self.average_augmentations else filename_parts
        metadata_str = "_".join(filename_parts)
        savename = f"{data_dict.get(self.map_extractor.id_col, '')}_{metadata_str}.nrrd"
        savepath = os.path.join(save_habitats_dirpath, savename)

        if isinstance(habitat_map, np.ndarray):
            habitat_map = sitk.GetImageFromArray(habitat_map)
            try:
                reference_image = sitk.ReadImage(data_dict['image'])
                habitat_map.CopyInformation(reference_image)
            except Exception as e:
                warnings.warn(f'Habitat map {savename} origin, spacing, directions could not be copied. Only voxel values will be saved. Error: {e}')

        os.makedirs(save_habitats_dirpath, exist_ok=True)
        sitk.WriteImage(habitat_map, savepath)

        return savepath
    
    def _generate_maps_for_habitats(self, data_dicts):
        # Run map generator to calculate maps dynamically (memory-efficient)
        results = self.map_extractor.run_generator(data_dicts)

        data_dicts_for_habitats = []

        if not self.augmentations:
            # Standard workflow when augmentations are not performed
            for res_dict in results:       
                metadata = {k:v for k,v in res_dict['result']['metadata'].items() if 'diagnostics' not in k} 
                feature_names, fmaps = self._check_or_set_channels(res_dict['result']['maps'])

                # Stack feature maps as one vector image (i.e. an image with 3 spatial dims + a channel dim)
                vector_img = sitk.Compose(fmaps)

                # Intermediate saving of vector images to disk (for memory efficiency purposes).
                # Savepaths are passed to the habitat generating algorithm, which loads them in batches.
                savepath = self._save_vector_image(vector_img, metadata)

                data_dicts_for_habitats.append({
                    'image' : savepath,
                    'mask' : metadata.get('mask_path', ''),
                    'metadata' : metadata,
                    'case_id' : res_dict.get('case_id', ''),
                    'feature_names' : feature_names
                })
        else:
            # Modified workflow when augmentations are enabled.
            # Maps for all augmented copies (plus original) of each case are loaded together as a batch

            # IMPORTANT: This assumes feature extractor returns all k+1 copies of each image (original + k augmentations)
            # consecutively. If this assumptions breaks, images from different cases may be added.
            for batch_result in self._batch_generator(results, k=len(self.augmentations) + 1):

                case_ids = [r.get('case_id') for r in batch_result]
                if len(set(case_ids)) != 1:
                    raise ValueError(f"Mixed case_ids in augmentation batch: {case_ids}")
                
                augmented_copies = []
                augmented_metadata = []
                
                for res_dict in batch_result:
                    metadata = {k:v for k,v in res_dict['result']['metadata'].items() if 'diagnostics' not in k}
                    feature_names, fmaps = self._check_or_set_channels(res_dict['result']['maps'])

                    vector_img = sitk.Compose(fmaps)

                    if self.average_augmentations:
                        # Collect augmented feature maps for downstream averaging
                        augmented_copies.append(vector_img)
                        augmented_metadata.append(metadata)
                    else:
                        # Save each augmented set of feature maps as a separate image
                        savepath = self._save_vector_image(vector_img, metadata)
                        
                        data_dicts_for_habitats.append({
                            'image' : savepath,
                            'mask' : metadata.get('mask_path', ''),
                            'metadata' : metadata,
                            'case_id' : res_dict.get('case_id', ''),
                            'feature_names' : feature_names
                        })
                
                if self.average_augmentations:
                    if len(augmented_copies) == 0:
                        continue

                    # Perform feature-wise averaging of augmented maps
                    # (i.e. separately average augmented copies of Busyness maps, Autocorrelation maps, etc.)
                    n_features = augmented_copies[0].GetNumberOfComponentsPerPixel()
                    channels = []
                    for c in range(n_features):
                        comps = [sitk.VectorIndexSelectionCast(img, c) for img in augmented_copies]
                        mean_c = comps[0]
                        for comp in comps[1:]:
                            mean_c = mean_c + comp
                        mean_c = mean_c / float(len(comps))
                        channels.append(mean_c)
                    averaged_vector_image = sitk.Compose(channels)
                
                    savepath = self._save_vector_image(averaged_vector_image, augmented_metadata[0])

                    data_dicts_for_habitats.append({
                            'image' : savepath,
                            'mask' : augmented_metadata[0].get('mask_path', ''),
                            'metadata' : augmented_metadata[0],
                            'case_id' : batch_result[0].get('case_id', ''),
                            'feature_names' : feature_names
                        })
        
        return data_dicts_for_habitats

    def fit(self, data_dicts):
        data_dicts_for_habitats = self._generate_maps_for_habitats(data_dicts)
        self.logger.info('Feature maps generated. Now fitting habitat generator.')
        self.habitat_generator.fit(data_dicts_for_habitats)
        self.logger.info('Habitat generator fitting done.')


    def predict(self, data_dicts, return_as_sitk_image=False, save_habitats_dirpath=None):
        data_dicts_for_habitats = self._generate_maps_for_habitats(data_dicts)
        self.logger.info('Feature maps generated. Now predicting habitats.')
        predictions = self.habitat_generator.predict(data_dicts_for_habitats, return_as_sitk_image=return_as_sitk_image)
        self.logger.info('Habitat predictions done.')
        if save_habitats_dirpath is not None:
            for dd, pred in zip(data_dicts, predictions):
                self._save_habitat_map(pred, dd, save_habitats_dirpath=save_habitats_dirpath)

        return predictions, data_dicts_for_habitats

    def fit_predict(self, data_dicts, return_as_sitk_image=False, save_habitats_dirpath=None):
        self.fit(data_dicts)
        return self.predict(data_dicts,
                            return_as_sitk_image=return_as_sitk_image,
                            save_habitats_dirpath=save_habitats_dirpath,
                            )

    def save(self, filepath):
        state = {
            'preprocessing': self.preprocessing,
            'augmentations': self.augmentations,
            'features': self.features,
            'filter_types': self.filter_types,
            'channels': self.channels,
            'algorithm': self.algorithm_name,
            'n_clusters': self.n_clusters,
            'batch_size': self.batch_size,
            'bin_width': self.bin_width,
            'bin_count': self.bin_count,
            'engine': self.engine,
            'voxel_based_settings': self.voxel_based_settings,
            'save_vector_dirpath': self.save_vector_dirpath,
            'save_fmaps_dirpath' : self.save_fmaps_dirpath,
            'average_augmentations': self.average_augmentations,
            'include_spatial_features' : self.include_spatial_features,
            'habitat_generator_state': self.habitat_generator.prepare_state_for_saving(),
        }

        with open(filepath, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, filepath):
        with open(filepath, "rb") as f:
            state = pickle.load(f)

        obj = cls(
            preprocessing=state['preprocessing'],
            augmentations=state['augmentations'],
            features=state['features'],
            filter_types=state['filter_types'],
            algorithm=state['algorithm'],
            n_clusters=state['n_clusters'],
            batch_size=state['batch_size'],
            bin_width=state['bin_width'],
            bin_count=state['bin_count'],
            engine=state['engine'],
            voxel_based_settings=state['voxel_based_settings'],
            save_vector_dirpath=state['save_vector_dirpath'],
            save_fmaps_dirpath=state['save_fmaps_dirpath'],
            average_augmentations=state['average_augmentations'],
            include_spatial_features=state['include_spatial_features']
        )

        obj.channels = state['channels']

        obj.habitat_generator = HabitatGenerator.load_from_state(
            state['habitat_generator_state']
        )

        obj.habitat_generator.channels = obj.channels

        obj._init_map_extractor()

        return obj
