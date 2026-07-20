import numpy as np
from .clustering_algorithm_base import VoxelClusteringAlgorithm

class KMeansClustering(VoxelClusteringAlgorithm):
    def __init__(self, n_clusters, mean_=None, std_=None, batch_size=100, weights=None, **kwargs):
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
    def __init__(self, n_clusters, mean_=None, std_=None, batch_size=100, weights=None, covariance_type='full', **kwargs): 
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

    def fit_partial(self):
        raise NotImplementedError(
            "GMM does not implement fit_partial. "
            "It must be trained on the full dataset at once, not in batches. "
            "Either increase batch size to cover the whole dataset (if your memory can take it) "
            "or use a smaller dataset."
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
    
class BirchClustering(VoxelClusteringAlgorithm):
    def __init__(self, n_clusters, mean_=None, std_=None, batch_size=100, weights=None, **kwargs):
        super().__init__(n_clusters, mean_=mean_, std_=std_)
        self.kwargs = kwargs

    def _init_model(self):
        from sklearn.cluster import Birch
        self.model = Birch(
            n_clusters=self.n_clusters,
            **self.kwargs
        )


class FuzzyCMeansClustering(VoxelClusteringAlgorithm):
    def __init__(self, n_clusters, mean_=None, std_=None, batch_size=100, weights=None, **kwargs):
        try:
            import skfuzzy
            
            self.centroids = None
            """
            Scikit-fuzzy implements Fuzzy C Means algorithm as fit and predict functions
            (skfuzzy.cmeans and skfuzzy.cmeans_predicts), not as a class.
            """
            self.model = None 
        except ImportError:
            raise ImportError(
                "FuzzyCMeans clustering requires the scikit-fuzzy package. "
                "Please install this package or choose another algorithm. "
                )
        except Exception as e:
            raise(e)
        
        super().__init__(n_clusters, mean_=mean_, std_=std_)
        self.kwargs = kwargs

    def _init_model(self):
        self.model = None 
        self.centroids = None
        
    def get_labels_from_probs(self, u):
        return np.argmax(u, axis=0)
    
    def fit(self, X):
        X = self._apply_normalization_and_weights(X)
        cntr, u, u0, d, jm, p, fpc = skfuzzy.cmeans(
            data=X.T, 
            c=self.n_clusters,
            **self.kwargs
        )
        self.centroids = cntr
        return cntr, u, u0, d, jm, p, fpc

    def fit_predict(self, X):
        cntr, u, u0, d, jm, p, fpc = self.fit(X)
        return self.get_labels_from_probs(u)

    def predict(self, X):
        X = self._apply_normalization_and_weights(X)
        u, u0, d, jm, p, fpc = skfuzzy.cmeans_predict(
            test_data=X.T,
            cntr_trained=self.centroids,
        )
        return self.get_labels_from_probs(u)

    def fit_partial(self):
        raise NotImplementedError(
            "FuzzyCMeans does not implement fit_partial. "
            "It must be trained on the full dataset at once, not in batches. "
            "Either increase batch size to cover the whole dataset (if your memory can take it) "
            "or use a smaller dataset."
        )
