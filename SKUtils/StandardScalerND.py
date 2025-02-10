"""
N-dimensional StandardScaler
"""

import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator


class StandardScalerND(TransformerMixin, BaseEstimator):
    """
    Standardizes multi-dimensional arrays (tensors) along specified dimensions.

    This transformer extends the functionality of the standard StandardScaler to
    handle N-dimensional arrays by allowing the user to specify the axis along
    which to calculate the mean and standard deviation for normalization.
    E.g. for CNN input (b, c, h, w) and dim=(0, 2, 3) the mean and std will have
    shape (c) and every channel will be normalized separately.
    If dim=0, the mean and std will have shape (1, c, h, w), meaning that each
    individual feature will be normalized separately.


    Parameters
    ----------
    dim : int, optional (default=None)
        The axis along which to calculate the mean and standard deviation.
        If None, normalization is performed along the first axis (axis 0), which
        is equivalent to the behavior of the standard `StandardScaler`.

    Attributes
    ----------
    means_ : ndarray of shape (n_dims, ...)
        The mean of the data along the specified axis.  The number of
        dimensions of `means_` will be the same as the input data `X`, except
        for the dimension specified by `dim`, which will be 1.

    stds_ : ndarray of shape (n_dims, ...)
        The standard deviation of the data along the specified axis.  The
        number of dimensions of `stds_` will be the same as the input data
        `X`, except for the dimension specified by `dim`, which will be 1.

    epsilon_ : float (default=1e-8)
        A small value added to the standard deviation to prevent division by zero.
    """

    def __init__(self, dim=None):
        self.dim = dim
        self.means = None
        self.stds = None
        self.epsilon = 1e-8

    def fit(self, X, y=None):
        axis = self.dim if self.dim is not None else 0
        self.means = np.mean(X, axis=axis, keepdims=True)
        self.stds = np.std(X, axis=axis, keepdims=True)
        return self

    def transform(self, X, y=None):
        if self.means is None or self.stds is None:
            raise RuntimeError("The scaler has not been fitted yet.")
        X_normalized = (X - self.means) / (self.stds + self.epsilon)
        return X_normalized

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)
