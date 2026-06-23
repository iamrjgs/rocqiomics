from abc import ABC, abstractmethod
import numpy as np

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
