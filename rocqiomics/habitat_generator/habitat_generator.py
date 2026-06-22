import os
import pickle
import warnings

import numpy as np
import SimpleITK as sitk

from .clustering_algorithms import (
    KMeansClustering,
    GMMClustering,
    BirchClusteringAlgorithm
)

# class HabitatGenerator:
#     _ALGORITHM_REGISTRY = {
#         "km" : KMeansClustering,
#         "kmeans": KMeansClustering,
#         "gmm": GMMClustering,
#         "gaussian_mixture": GMMClustering,
#         "birch" : BirchClusteringAlgorithm
#     }

#     def __init__(self, channels, algorithm, n_clusters=4, batch_size=50, normalize=True, **algorithm_kwargs):
#         self.channels = self._prepare_channels(channels)
#         self.batch_size = batch_size
#         self.normalize = normalize
#         self.mean_ = None
#         self.std_ = None
#         self.algorithm_name = algorithm
#         self.algorithm_kwargs = algorithm_kwargs
#         self.n_clusters = n_clusters
#         self.algorithm = None

#     def _prepare_channels(self, channels):
#         if isinstance(channels, list):
#             return {name: idx for idx, name in enumerate(channels)}
#         elif isinstance(channels, dict):
#             for k, v in channels.items():
#                 if not isinstance(v, int):
#                     raise ValueError
#             return channels
#         else:
#             raise TypeError

#     def _select_channels(self, image_4d):
#         indices = list(self.channels.values())
#         if image_4d.shape[-1] <= max(indices):
#             raise ValueError
#         return image_4d[..., indices]

#     def _prepare_mask(self, mask, image_4d):
#         if mask is None:
#             return None
#         if mask.ndim == 4:
#             mask = mask[..., 0]
#         if mask.shape != image_4d.shape[:3]:
#             raise ValueError(f"Mask and image spatial dimensions don't match. Mask: {mask.shape}, Image: {image_4d.shape}")
#         return mask
    
#     def _prepare_algorithm(self):
#         key = self.algorithm_name.lower()
#         algo_class = self._ALGORITHM_REGISTRY[key]
#         return algo_class(
#             n_clusters=self.n_clusters,
#             mean_=self.mean_,
#             std_=self.std_,
#             batch_size=self.batch_size,
#             **self.algorithm_kwargs
#         )

#     def _prepare_X_list(self, images, masks):
#         X_list = []
#         for img, mask in zip(images, masks):
#             vox = img.reshape(-1, img.shape[-1])
#             vox = vox[mask.reshape(-1).astype(bool)]
#             if vox.size > 0:
#                 X_list.append(vox)
#         return X_list

#     def _iter_batches(self, items):
#         for k in range(0, len(items), self.batch_size):
#             yield items[k:k + self.batch_size]

#     @staticmethod
#     def extract_geometry_info(img):
#         return {
#             'origin' : img.GetOrigin(),
#             'spacing' : img.GetSpacing(),
#             'direction' : img.GetDirection()
#         }

#     @staticmethod
#     def set_geometry_info(img, geometry_info):
#         dim = img.GetDimension()

#         origin = geometry_info.get('origin', (0.0,) * dim)
#         spacing = geometry_info.get('spacing', (1.0,) * dim)

#         default_direction = tuple(
#             1.0 if i == j else 0.0
#             for i in range(dim)
#             for j in range(dim)
#         )

#         direction = geometry_info.get('direction', default_direction)

#         img.SetOrigin(origin)
#         img.SetSpacing(spacing)
#         img.SetDirection(direction)
#         return img

#     def _load_image_as_numpy(self, dd):
#         if 'image' not in dd:
#             raise ValueError(f'Data dict missing image key: {dd}')
        
#         geometry_info = {}
        
#         image = dd['image']
#         if isinstance(image, sitk.Image):
#             image = sitk.GetArrayFromImage(image)
#             geometry_info = self.extract_geometry_info(image)
#         if isinstance(image, str):
#             image = sitk.ReadImage(image)
#             geometry_info = self.extract_geometry_info(image)
#             image = sitk.GetArrayFromImage(image)

#         if 'mask' in dd:
#             mask = dd['mask']
#         else:
#             mask = np.ones(image.shape[:3])

#         if isinstance(mask, sitk.Image):
#             mask = sitk.GetArrayFromImage(mask)
#         if isinstance(mask, str):
#             if os.path.exists(mask):
#                 mask = sitk.GetArrayFromImage(sitk.ReadImage(mask))
#             else:
#                 warnings.warn(f"Mask at path {mask} not found. Defaulting to whole-image mask.")
#                 mask = np.ones(image.shape[:3])
  
#         return image, mask, geometry_info
 
#     def _compute_stats(self, data):
#         total_sum = None
#         total_sq = None
#         total_n = 0
#         for batch in self._iter_batches(data):
#             for dd in batch:
#                 img, mask, _ = self._load_image_as_numpy(dd)
#                 img = self._select_channels(img)
#                 mask = self._prepare_mask(mask, img)
#                 vox = img.reshape(-1, img.shape[-1])
#                 vox = vox[mask.reshape(-1).astype(bool)]
#                 if vox.size == 0:
#                     continue
#                 if total_sum is None:
#                     total_sum = vox.sum(axis=0)
#                     total_sq = (vox ** 2).sum(axis=0)
#                 else:
#                     total_sum += vox.sum(axis=0)
#                     total_sq += (vox ** 2).sum(axis=0)
#                 total_n += vox.shape[0]

#         if total_n == 0:
#             raise ValueError("No valid voxels found.")
                
#         self.mean_ = total_sum / total_n
#         var = total_sq / total_n - self.mean_ ** 2
#         var = np.maximum(var, 0)
#         self.std_ = np.sqrt(var)
#         self.std_[self.std_ < 1e-8] = 1.0

#     def fit(self, data):
#         # If vector images directly given as data, wrap them in the expected dict format
#         if isinstance(data[0], np.ndarray):
#             data = [{
#                 'image' : d,
#                 'mask' : np.ones(d.shape[:3])
#             } for d in data]

#         if self.normalize:
#             self._compute_stats(data)

#         self.algorithm = self._prepare_algorithm() 

#         for i, batch in enumerate(self._iter_batches(data)):
#             images = []
#             masks = []

#             for dd in batch:
#                 img, mask, _ = self._load_image_as_numpy(dd)
#                 img = self._select_channels(img)
#                 mask = self._prepare_mask(mask, img)

#                 images.append(img)
#                 masks.append(mask)

#             X_list = self._prepare_X_list(images, masks)
#             X = np.vstack(X_list)

#             if i == 0:
#                 self.algorithm.fit(X)
#             else:
#                 self.algorithm.partial_fit(X)
        
#         return self
    
#     def predict(self, data, return_as_sitk_image=False):
#         # If vector images directly given as data, wrap them in the expected dict format
#         if isinstance(data[0], np.ndarray):
#             data = [{
#                 'image' : d,
#                 'mask' : np.ones(d.shape[:3])
#             } for d in data]

#         outputs = []

#         for dd in data:
#             img, mask, geometry_info = self._load_image_as_numpy(dd)
#             img = self._select_channels(img)
#             mask = self._prepare_mask(mask, img)

#             pred = self.algorithm.predict(img, mask=mask)

#             if return_as_sitk_image:
#                 pred = sitk.GetImageFromArray(pred)
#                 pred = self.set_geometry_info(pred, geometry_info)

#             outputs.append(pred)
                
#         return outputs

#     def fit_predict(self, data):
#         self.fit(data)
#         return self.predict(data)
    
#     def prepare_state_for_saving(self):
#         if self.algorithm is None:
#             raise ValueError("Cannot save an unfitted model (algorithm is None).")
#         return {
#             'channels': self.channels,
#             'batch_size': self.batch_size,
#             'normalize': self.normalize,
#             'mean_': self.mean_,
#             'std_': self.std_,
#             'algorithm_name': self.algorithm_name,
#             'algorithm_kwargs': self.algorithm_kwargs,
#             'n_clusters': self.n_clusters,
#             'algorithm': self.algorithm,
#         }
    
#     def save(self, filepath):
#         state = self.prepare_state_for_saving()
#         with open(filepath, "wb") as f:
#             pickle.dump(state, f)
        
#     @classmethod
#     def load_from_state(cls, state):
#         obj = cls(
#             channels=state['channels'],
#             algorithm=state['algorithm_name'],
#             n_clusters=state['n_clusters'],
#             batch_size=state['batch_size'],
#             normalize=state['normalize'],
#             **state['algorithm_kwargs'],
#         )

#         obj.mean_ = state['mean_']
#         obj.std_ = state['std_']
#         obj.algorithm = state['algorithm']

#         return obj
    
#     def load(self, cls, filepath):
#         with open(filepath, "rb") as f:
#             state = pickle.load(f)
#         return self.load_from_state(cls, state)


class HabitatGenerator:
    _ALGORITHM_REGISTRY = {
        "km" : KMeansClustering,
        "kmeans": KMeansClustering,
        "gmm": GMMClustering,
        "gaussian_mixture": GMMClustering,
        "birch" : BirchClusteringAlgorithm
    }

    def __init__(self,
                 channels,
                 algorithm,
                 n_clusters=4,
                 batch_size=50,
                 normalize=True,
                 include_spatial_features=False,
                 spatial_weight=1.0,
                 **algorithm_kwargs):
        self.channels = self._prepare_channels(channels)
        self.batch_size = batch_size
        self.normalize = normalize
        self.mean_ = None
        self.std_ = None
        self.algorithm_name = algorithm
        self.algorithm_kwargs = algorithm_kwargs
        self.n_clusters = n_clusters
        self.algorithm = None
        self.include_spatial_features = include_spatial_features
        self.fitted = False
        self.spatial_weight = spatial_weight
        self._coord_cache = {}

    def _prepare_channels(self, channels):
        if isinstance(channels, list):
            return {name: idx for idx, name in enumerate(channels)}
        elif isinstance(channels, dict):
            for k, v in channels.items():
                if not isinstance(v, int):
                    raise ValueError
            return channels
        else:
            raise TypeError

    def _select_channels(self, image_4d):
        indices = list(self.channels.values())
        if image_4d.shape[-1] <= max(indices):
            raise ValueError
        return image_4d[..., indices]

    def _prepare_mask(self, mask, image_4d):
        if mask is None:
            return np.ones(image_4d.shape[:3], dtype=bool)
        if mask.ndim == 4:
            mask = mask[..., 0]
        if mask.shape != image_4d.shape[:3]:
            raise ValueError(f"Mask and image spatial dimensions don't match. Mask: {mask.shape}, Image: {image_4d.shape}")
        return mask
    
    def _prepare_feature_vector(self, image_4d, mask, geometry_info=None):
        if mask is None:
            mask = np.ones(image_4d.shape[:3])
            
        vox = image_4d.reshape(-1, image_4d.shape[-1])
        mask_flat = mask.reshape(-1).astype(bool)
        vox = vox[mask_flat]

        if self.include_spatial_features:
            vox = self._append_spatial_features(
                vox, mask_flat, image_4d.shape, geometry_info
            )

        return vox, mask_flat
    
    def _append_spatial_features(self, vox, mask_flat, img_shape, geometry_info=None):
        key = img_shape[:3]

        if geometry_info is not None:
            spacing = np.array(geometry_info.get('spacing', (1.0, 1.0, 1.0)))
            key = (img_shape[0], img_shape[1], img_shape[2], tuple(spacing))
        else:
            spacing = None

        if key not in self._coord_cache:
            coords = np.stack(
                np.meshgrid(
                    np.arange(img_shape[0]),
                    np.arange(img_shape[1]),
                    np.arange(img_shape[2]),
                    indexing="ij"
                ),
                axis=-1
            ).reshape(-1, 3)

            if spacing is not None:
                coords = coords * spacing

            if spacing is not None:
                norm = np.array(img_shape[:3]) * spacing
            else:
                norm = np.array(img_shape[:3])

            coords = coords / norm
            coords *= self.spatial_weight

            self._coord_cache[key] = coords
        else:
            coords = self._coord_cache[key]

        coords = coords[mask_flat]

        return np.concatenate([vox, coords], axis=1)

    def _prepare_algorithm(self):
        key = self.algorithm_name.lower()
        algo_class = self._ALGORITHM_REGISTRY[key]
        return algo_class(
            n_clusters=self.n_clusters,
            mean_=self.mean_,
            std_=self.std_,
            batch_size=self.batch_size,
            **self.algorithm_kwargs
        )

    def _prepare_X_list(self, images, masks, geometry_infos=None):
        X_list = []
        for img, mask, geom in zip(images, masks, geometry_infos):
            vox, _ = self._prepare_feature_vector(img, mask, geom)
            if vox.size > 0:
                X_list.append(vox)
        return X_list

    def _iter_batches(self, items):
        for k in range(0, len(items), self.batch_size):
            yield items[k:k + self.batch_size]

    @staticmethod
    def extract_geometry_info(img):
        return {
            'origin' : img.GetOrigin(),
            'spacing' : img.GetSpacing(),
            'direction' : img.GetDirection()
        }

    @staticmethod
    def set_geometry_info(img, geometry_info):
        dim = img.GetDimension()

        origin = geometry_info.get('origin', (0.0,) * dim)
        spacing = geometry_info.get('spacing', (1.0,) * dim)

        default_direction = tuple(
            1.0 if i == j else 0.0
            for i in range(dim)
            for j in range(dim)
        )

        direction = geometry_info.get('direction', default_direction)

        img.SetOrigin(origin)
        img.SetSpacing(spacing)
        img.SetDirection(direction)
        return img

    def _load_image_as_numpy(self, dd):
        if 'image' not in dd:
            raise ValueError(f'Data dict missing image key: {dd}')
        
        geometry_info = {}
        
        image = dd['image']
        if isinstance(image, sitk.Image):
            geometry_info = self.extract_geometry_info(image)
            image = sitk.GetArrayFromImage(image)
        if isinstance(image, str):
            image = sitk.ReadImage(image)
            geometry_info = self.extract_geometry_info(image)
            image = sitk.GetArrayFromImage(image)

        if 'mask' in dd:
            mask = dd['mask']
        else:
            mask = np.ones(image.shape[:3])

        if isinstance(mask, sitk.Image):
            mask = sitk.GetArrayFromImage(mask)
        if isinstance(mask, str):
            if os.path.exists(mask):
                mask = sitk.GetArrayFromImage(sitk.ReadImage(mask))
            else:
                warnings.warn(f"Mask at path {mask} not found. Defaulting to whole-image mask.")
                mask = np.ones(image.shape[:3])
  
        return image, mask, geometry_info
 
    def _compute_stats(self, data):
        total_sum = None
        total_sq = None
        total_n = 0
        for batch in self._iter_batches(data):
            for dd in batch:
                img, mask, geometry_info = self._load_image_as_numpy(dd)
                img = self._select_channels(img)
                mask = self._prepare_mask(mask, img)
                vox, _ = self._prepare_feature_vector(img, mask,geometry_info=geometry_info)
                if vox.size == 0:
                    continue
                if total_sum is None:
                    total_sum = vox.sum(axis=0)
                    total_sq = (vox ** 2).sum(axis=0)
                else:
                    total_sum += vox.sum(axis=0)
                    total_sq += (vox ** 2).sum(axis=0)
                total_n += vox.shape[0]

        if total_n == 0:
            raise ValueError("No valid voxels found.")
                
        self.mean_ = total_sum / total_n
        var = total_sq / total_n - self.mean_ ** 2
        var = np.maximum(var, 0)
        self.std_ = np.sqrt(var)
        self.std_[self.std_ < 1e-8] = 1.0

    def fit(self, data):
        if isinstance(data[0], np.ndarray):
            data = [{
                'image' : d,
                'mask' : np.ones(d.shape[:3])
            } for d in data]

        if self.normalize:
            self._compute_stats(data)

        self.algorithm = self._prepare_algorithm() 
        self.fitted = False

        for i, batch in enumerate(self._iter_batches(data)):
            images = []
            masks = []
            geometry_infos = []

            for dd in batch:
                img, mask, geometry_info = self._load_image_as_numpy(dd)
                img = self._select_channels(img)
                mask = self._prepare_mask(mask, img)
                images.append(img)
                masks.append(mask)
                geometry_infos.append(geometry_info)

            X_list = self._prepare_X_list(images, masks, geometry_infos)
            if len(X_list) == 0:
                continue
            X = np.vstack(X_list)

            if i == 0:
                self.algorithm.fit(X)
                self.fitted = True
            else:
                self.algorithm.partial_fit(X)

        
        if not self.fitted:
            raise ValueError("No valid data for fitting.")

        return self
    
    def predict(self, data, return_as_sitk_image=False):
        if isinstance(data[0], np.ndarray):
            data = [{
                'image' : d,
                'mask' : np.ones(d.shape[:3])
            } for d in data]

        outputs = []

        for dd in data:
            img, mask, geometry_info = self._load_image_as_numpy(dd)
            img = self._select_channels(img)
            mask = self._prepare_mask(mask, img)

            vox, mask_flat = self._prepare_feature_vector(img, mask, geometry_info=geometry_info)

            labels = np.full(mask.shape, np.nan)
            if vox.size > 0:
                pred_flat = self.algorithm.predict(vox)
                labels_flat = labels.reshape(-1)
                labels_flat[mask_flat] = pred_flat
                labels = labels_flat.reshape(mask.shape)

            if return_as_sitk_image:
                labels = sitk.GetImageFromArray(labels)
                labels = self.set_geometry_info(labels, geometry_info)

            outputs.append(labels)
                
        return outputs

    def fit_predict(self, data):
        self.fit(data)
        return self.predict(data)
    
    def prepare_state_for_saving(self):
        if self.algorithm is None:
            raise ValueError("Cannot save an unfitted model (algorithm is None).")
        return {
            'channels': self.channels,
            'batch_size': self.batch_size,
            'normalize': self.normalize,
            'mean_': self.mean_,
            'std_': self.std_,
            'algorithm_name': self.algorithm_name,
            'algorithm_kwargs': self.algorithm_kwargs,
            'n_clusters': self.n_clusters,
            'algorithm': self.algorithm,
            'include_spatial_features': self.include_spatial_features,
            'spatial_weight' : self.spatial_weight
        }
    
    def save(self, filepath):
        state = self.prepare_state_for_saving()
        with open(filepath, "wb") as f:
            pickle.dump(state, f)
        
    @classmethod
    def load_from_state(cls, state):
        obj = cls(
            channels=state['channels'],
            algorithm=state['algorithm_name'],
            n_clusters=state['n_clusters'],
            batch_size=state['batch_size'],
            normalize=state['normalize'],
            include_spatial_features=state.get('include_spatial_features', False),
            spatial_weight=state.get('spatial_weight', 1.0),
            **state['algorithm_kwargs'],
        )

        obj.mean_ = state['mean_']
        obj.std_ = state['std_']
        obj.algorithm = state['algorithm']

        return obj
    
    def load(self, cls, filepath):
        with open(filepath, "rb") as f:
            state = pickle.load(f)
        return self.load_from_state(cls, state)