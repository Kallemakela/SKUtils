"""
N-dimensional StandardScaler
"""
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator

class StandardScalerND(TransformerMixin, BaseEstimator):
    def __init__(self, dim=None):
        self.dim = dim
        self.means = None
        self.stds = None
        self.epsilon = 1e-8  # Small constant to avoid division by zero

    def fit(self, X, y=None):
        # Compute mean and std along the specified dimension
        if self.dim is not None:
            self.means = np.mean(X, axis=self.dim, keepdims=True)
            self.stds = np.std(X, axis=self.dim, keepdims=True)
        else:
			# everything but last
            axis = tuple(range(X.ndim - 1))
            self.means = np.mean(X, axis=axis, keepdims=True)
            self.stds = np.std(X, axis=axis, keepdims=True)
        return self

    def transform(self, X, y=None):
        if self.means is None or self.stds is None:
            raise RuntimeError("The scaler has not been fitted yet.")

        # Normalize X using broadcasting, works for different sample sizes
        # Add epsilon to stds to avoid division by zero
        X_normalized = (X - self.means) / (self.stds + self.epsilon)
        return X_normalized

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)