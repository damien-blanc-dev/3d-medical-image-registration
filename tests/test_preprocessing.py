"""Tests for preprocessing (preprocess.py)."""

import numpy as np
import pytest
import SimpleITK as sitk

from src.preprocessing.preprocess import (
    resample_to_spacing,
    normalize_intensity,
    match_spacing,
    crop_or_pad_to_shape,
)


@pytest.fixture
def anisotropic_volume():
    """Anisotropic volume: 0.7×0.7×5mm (typical CT slice thickness)."""
    arr = np.random.rand(30, 256, 256).astype(np.float32)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((0.7, 0.7, 5.0))
    return img


def test_resample_to_isotropic(anisotropic_volume):
    target = (1.0, 1.0, 1.0)
    resampled = resample_to_spacing(anisotropic_volume, target)
    np.testing.assert_allclose(resampled.GetSpacing(), target, rtol=1e-5)

    # Physical FOV should be approximately preserved
    original_fov = [
        sz * sp for sz, sp in zip(anisotropic_volume.GetSize(), anisotropic_volume.GetSpacing())
    ]
    resampled_fov = [
        sz * sp for sz, sp in zip(resampled.GetSize(), resampled.GetSpacing())
    ]
    np.testing.assert_allclose(resampled_fov, original_fov, rtol=0.02)


def test_normalize_zscore():
    arr = np.random.randn(20, 30, 40).astype(np.float32) * 500 + 200
    img = sitk.GetImageFromArray(arr)
    normalized = normalize_intensity(img, method='zscore')
    arr_n = sitk.GetArrayFromImage(normalized)
    assert abs(arr_n.mean()) < 0.1
    assert 0.5 < arr_n.std() < 2.0  # roughly unit variance


def test_normalize_minmax():
    arr = np.random.rand(10, 10, 10).astype(np.float32) * 1000
    img = sitk.GetImageFromArray(arr)
    normalized = normalize_intensity(img, method='minmax')
    arr_n = sitk.GetArrayFromImage(normalized)
    assert arr_n.min() >= 0.0 - 1e-5
    assert arr_n.max() <= 1.0 + 1e-5


def test_normalize_unknown_method():
    arr = np.zeros((5, 5, 5), dtype=np.float32)
    img = sitk.GetImageFromArray(arr)
    with pytest.raises(ValueError, match="Unknown normalization"):
        normalize_intensity(img, method='unknown_method')


def test_match_spacing():
    arr_f = np.zeros((50, 60, 70), dtype=np.float32)
    fixed = sitk.GetImageFromArray(arr_f)
    fixed.SetSpacing((1.0, 1.0, 1.0))

    arr_m = np.zeros((30, 40, 50), dtype=np.float32)
    moving = sitk.GetImageFromArray(arr_m)
    moving.SetSpacing((2.0, 2.0, 2.0))

    matched = match_spacing(moving, fixed)
    assert matched.GetSize() == fixed.GetSize()
    np.testing.assert_allclose(matched.GetSpacing(), fixed.GetSpacing(), rtol=1e-5)


def test_crop_or_pad_to_shape():
    arr = np.zeros((30, 40, 50), dtype=np.float32)
    img = sitk.GetImageFromArray(arr)

    # Pad to larger shape
    target = (60, 80, 100)
    padded = crop_or_pad_to_shape(img, target)
    assert sitk.GetArrayFromImage(padded).shape == target

    # Crop to smaller shape
    target_small = (20, 25, 30)
    cropped = crop_or_pad_to_shape(img, target_small)
    assert sitk.GetArrayFromImage(cropped).shape == target_small
