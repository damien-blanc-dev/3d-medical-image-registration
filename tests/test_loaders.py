"""Tests for volume I/O (loaders.py)."""

import numpy as np
import pytest
import SimpleITK as sitk

from src.io.loaders import (
    load_volume,
    save_volume,
    sitk_to_numpy,
    numpy_to_sitk,
    nibabel_to_sitk,
)


@pytest.fixture
def synthetic_volume(tmp_path):
    """Create a small synthetic 3D volume for testing."""
    arr = np.random.rand(30, 40, 50).astype(np.float32)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.5, 1.5, 3.0))
    img.SetOrigin((10.0, 20.0, -5.0))
    path = tmp_path / "test_volume.nii.gz"
    sitk.WriteImage(img, str(path))
    return path, img


def test_load_volume_nifti(synthetic_volume):
    path, original = synthetic_volume
    loaded = load_volume(path)
    assert loaded.GetSize() == original.GetSize()
    np.testing.assert_allclose(loaded.GetSpacing(), original.GetSpacing(), rtol=1e-5)


def test_load_volume_missing_file():
    with pytest.raises(FileNotFoundError):
        load_volume("/nonexistent/path/volume.nii.gz")


def test_save_and_reload(tmp_path):
    arr = np.ones((20, 30, 40), dtype=np.float32) * 42.0
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((2.0, 2.0, 2.0))

    out = tmp_path / "output.nii.gz"
    save_volume(img, out)
    assert out.exists()

    reloaded = load_volume(out)
    arr_reloaded = sitk_to_numpy(reloaded)
    np.testing.assert_allclose(arr_reloaded, arr, rtol=1e-5)


def test_sitk_to_numpy_shape():
    arr = np.zeros((10, 20, 30), dtype=np.float32)
    img = sitk.GetImageFromArray(arr)
    result = sitk_to_numpy(img)
    # GetArrayFromImage returns (z, y, x) — same as input
    assert result.shape == (10, 20, 30)


def test_numpy_to_sitk_copies_metadata():
    arr = np.ones((10, 20, 30), dtype=np.float32)
    ref = sitk.GetImageFromArray(arr)
    ref.SetSpacing((0.5, 0.5, 1.0))
    ref.SetOrigin((1.0, 2.0, 3.0))

    new_arr = arr * 2
    result = numpy_to_sitk(new_arr, ref)

    assert result.GetSize() == ref.GetSize()
    np.testing.assert_allclose(result.GetSpacing(), ref.GetSpacing())
    np.testing.assert_allclose(result.GetOrigin(), ref.GetOrigin())
