from abc import ABC, abstractmethod
import numpy as np

class VoxelClusteringAlgorithm(ABC):
    def __init__(self, n_clusters, mean_=None, std_=None, batch_size=None, weights=None):
        self.n_clusters = n_clusters
        self.model = None
        self.mean_ = mean_
        self.std_ = std_
        self.batch_size_ = batch_size
        self.weights = None if weights is None else np.asarray(weights, dtype=float)

    def fit(self, X):
        X = self._apply_normalization_and_weights(X)
        self._init_model()
        self.model.fit(X)
        return self
    
    def fit_predict(self, X):
        self.fit(X)
        X = self._apply_normalization_and_weights(X)
        return self.model.predict(X)

    def partial_fit(self, X):
        if self.model is None:
            self._init_model()
        X = self._apply_normalization_and_weights(X)
        self.model.partial_fit(X)

    def predict(self, X):
        X = self._apply_normalization_and_weights(X)
        return self.model.predict(X)

    def _apply_normalization_and_weights(self, X):
        if self.mean_ is not None and self.std_ is not None:
            n_int = len(self.mean_)
            X_int = (X[:, :n_int] - self.mean_) / self.std_
            
            if X.shape[1] > n_int:
                # If there are more channels than means, only normalize the first n_int channels
                # This will happen, e.g. when spatial features are included, since mean/std are
                # not computed for these.
                X = np.concatenate([X_int, X[:, n_int:]], axis=1)
            else:
                X = X_int

        if self.weights is not None:
            if X.shape[1] != len(self.weights):
                raise ValueError(
                    f"Feature dimension mismatch: X has {X.shape[1]} features, "
                    f"but weights has length {len(self.weights)}"
                )
            X = X * self.weights

        return X

    @abstractmethod
    def _init_model(self):
        pass