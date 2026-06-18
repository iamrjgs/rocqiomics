import numpy as np
from .clustering_algorithm_base import VoxelClusteringAlgorithm

class KMeansClustering(VoxelClusteringAlgorithm):
    def __init__(self, n_clusters, mean_=None, std_=None, batch_size=100, **kwargs):
        super().__init__(n_clusters, mean_=mean_, std_=std_)
        self.batch_size = batch_size
        self.kwargs = kwargs

    def _init_model(self):
        from sklearn.cluster import MiniBatchKMeans
        self.model = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            batch_size=self.batch_size,
            **self.kwargs
        )

class GMMClustering(VoxelClusteringAlgorithm):
    def __init__(self, n_clusters, mean_=None, std_=None, batch_size=100, covariance_type='full', **kwargs): 
        super().__init__(n_clusters, mean_=mean_, std_=std_)
        self.covariance_type = covariance_type
        self.kwargs = kwargs

    def _init_model(self):
        from sklearn.mixture import GaussianMixture
        self.model = GaussianMixture(
            n_components=self.n_clusters,
            covariance_type=self.covariance_type,
            **self.kwargs
        )

    def predict_proba(self, image_4d, mask=None):
        voxels, shape, mask_flat = self._prepare_single(image_4d, mask=mask)
        output = np.full(voxels.shape[0], np.nan)
        if mask_flat is None:
            labels = self.model.predict_proba(voxels)
            return labels.reshape(shape)
        labels = self.model.predict_proba(voxels[mask_flat])
        output[mask_flat] = labels
        return output.reshape(shape)
    

class BirchClusteringAlgorithm(VoxelClusteringAlgorithm):
    def __init__(self, n_clusters, mean_=None, std_=None, batch_size=100, **kwargs):
        super().__init__(n_clusters, mean_=mean_, std_=std_)
        self.kwargs = kwargs

    def _init_model(self):
        from sklearn.cluster import Birch
        self.model = Birch(
            n_clusters=self.n_clusters,
            **self.kwargs
        )