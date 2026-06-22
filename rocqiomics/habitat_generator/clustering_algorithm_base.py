from abc import ABC, abstractmethod
import numpy as np

# class VoxelClusteringAlgorithm(ABC):
#     def __init__(self, n_clusters, mean_=None, std_=None):
#         self.n_clusters = n_clusters
#         self.model = None
#         self.mean_ = mean_
#         self.std_ = std_

#     def _prepare_mask(self, mask, image_4d):
#         if mask is None:
#             return None
#         if mask.ndim == 4:
#             mask = mask[..., 0]
#         if mask.ndim != 3:
#             raise ValueError
#         if mask.shape != image_4d.shape[:3]:
#             raise ValueError
#         return mask.reshape(-1).astype(bool)

#     def _prepare_single(self, image_4d, mask=None):
#         shape = image_4d.shape[:3]
#         voxels = image_4d.reshape(-1, image_4d.shape[-1])
#         if mask is not None:
#             mask_flat = self._prepare_mask(mask, image_4d)
#         else:
#             mask_flat = None
#         return voxels, shape, mask_flat

#     def _apply_normalization(self, X):
#         if self.mean_ is None or self.std_ is None:
#             return X
#         return (X - self.mean_) / self.std_

#     @abstractmethod
#     def _init_model(self):
#         pass

#     def fit(self, X):
#         X = self._apply_normalization(X)
#         self._init_model()
#         self.model.fit(X)
#         return self
    
#     def fit_predict(self, X):
#         self.fit(X)
#         X = self._apply_normalization(X)
#         return self.model.predict(X)

#     def partial_fit(self, X):
#         if self.model is None:
#             self._init_model()
#         X = self._apply_normalization(X)
#         self.model.partial_fit(X)

#     def predict(self, image_4d, mask=None):
#         voxels, shape, mask_flat = self._prepare_single(image_4d, mask)
#         voxels = self._apply_normalization(voxels)
#         output = np.full(voxels.shape[0], np.nan)
#         if mask_flat is None:
#             labels = self.model.predict(voxels)
#             return labels.reshape(shape)
#         labels = self.model.predict(voxels[mask_flat])
#         output[mask_flat] = labels
#         return output.reshape(shape)



class VoxelClusteringAlgorithm(ABC):
    def __init__(self, n_clusters, mean_=None, std_=None):
        self.n_clusters = n_clusters
        self.model = None
        self.mean_ = mean_
        self.std_ = std_

    def _apply_normalization(self, X):
        if self.mean_ is None or self.std_ is None:
            return X
        return (X - self.mean_) / self.std_

    @abstractmethod
    def _init_model(self):
        pass

    def fit(self, X):
        X = self._apply_normalization(X)
        self._init_model()
        self.model.fit(X)
        return self
    
    def fit_predict(self, X):
        self.fit(X)
        X = self._apply_normalization(X)
        return self.model.predict(X)

    def partial_fit(self, X):
        if self.model is None:
            self._init_model()
        X = self._apply_normalization(X)
        self.model.partial_fit(X)

    def predict(self, X):
        X = self._apply_normalization(X)
        return self.model.predict(X)
