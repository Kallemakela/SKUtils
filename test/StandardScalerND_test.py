import pytest
import numpy as np
from SKUtils.StandardScalerND import StandardScalerND
from sklearn.preprocessing import StandardScaler

def test_2d_array_normalization():
    X = np.array([[1, 2, 3], [4, 5, 6]])
    scaler = StandardScalerND(dim=0)
    X_scaled = scaler.fit_transform(X)
    expected_mean = np.zeros(3)
    expected_std = np.array([1, 1, 1])
    np.testing.assert_allclose(X_scaled.mean(axis=0), expected_mean, atol=1e-7)
    np.testing.assert_allclose(X_scaled.std(axis=0), expected_std, atol=1e-7)

def test_3d_array_normalization():
    X = np.random.rand(4, 3, 2)
    scaler = StandardScalerND(dim=(0,1))
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    # Check if the mean along the specified dimension is close to 0
    np.testing.assert_allclose(X_scaled.mean(axis=(0,1)), 0, atol=1e-7)

def test_overall_normalization():
    X = np.random.rand(4, 3, 2)
    scaler = StandardScalerND()  # dim=None for overall normalization
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    # Check overall mean and std
    assert np.isclose(X_scaled.mean(), 0, atol=1e-7)
    assert np.isclose(X_scaled.std(), 1, atol=1e-7)

def test_different_sample_sizes():
    X_train = np.random.rand(5, 3, 2)
    X_test = np.random.rand(3, 3, 2)
    scaler = StandardScalerND(dim=0)
    scaler.fit(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Just verify that transform does not raise an error and returns expected shape
    assert X_test_scaled.shape == X_test.shape

def test_transform_before_fit():
    scaler = StandardScalerND(dim=0)
    with pytest.raises(RuntimeError):
        scaler.transform(np.random.rand(3, 3))

def test_equivalence_with_sklearn():
	X = np.random.rand(4, 3)
	scaler = StandardScalerND()
	sklearn_scaler = StandardScaler()
	X_scaled = scaler.fit_transform(X)
	X_sklearn_scaled = sklearn_scaler.fit_transform(X)
	np.testing.assert_allclose(X_scaled, X_sklearn_scaled, atol=1e-7)