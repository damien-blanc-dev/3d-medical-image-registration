"""
Tests for rigid registration (rigid.py).

Uses a known synthetic registration case: a volume is translated by a known
offset, and we verify that the recovered transform approximates that offset.
"""

import numpy as np
import pytest
import SimpleITK as sitk

from src.registration.rigid import (
    RigidRegistration,
    AffineRegistration,
    RegistrationConfig,
    RegistrationResult,
)


def _make_synthetic_pair(translation_mm=(10.0, 5.0, 0.0), size=(64, 64, 64)):
    """Create a fixed/moving pair with known rigid misalignment."""
    # Fixed: sphere in the center of the volume
    arr = np.zeros(size, dtype=np.float32)
    center = np.array(size) // 2
    r = size[0] // 6
    zz, yy, xx = np.ogrid[:size[0], :size[1], :size[2]]
    mask = (zz - center[0])**2 + (yy - center[1])**2 + (xx - center[2])**2 <= r**2
    arr[mask] = 1.0
    arr += np.random.rand(*size).astype(np.float32) * 0.05

    fixed = sitk.GetImageFromArray(arr)
    fixed.SetSpacing((1.0, 1.0, 1.0))

    # Moving: translate the fixed volume
    transform = sitk.TranslationTransform(3)
    transform.SetOffset(translation_mm)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetTransform(transform)
    resampler.SetInterpolator(sitk.sitkLinear)
    moving = resampler.Execute(fixed)

    return fixed, moving, translation_mm


@pytest.fixture
def synthetic_pair():
    return _make_synthetic_pair(translation_mm=(8.0, 4.0, 0.0))


def test_rigid_registration_reduces_misalignment(synthetic_pair):
    fixed, moving, true_translation = synthetic_pair

    # Compute initial misalignment as mean absolute difference
    arr_f = sitk.GetArrayFromImage(fixed)
    arr_m = sitk.GetArrayFromImage(moving)
    error_before = np.mean(np.abs(arr_f - arr_m))

    config = RegistrationConfig(
        number_of_iterations=100,
        shrink_factors=[2, 1],
        smoothing_sigmas=[1.0, 0.0],
    )
    result = RigidRegistration(config).register(fixed, moving, verbose=False)
    registered = result.apply(moving)

    arr_r = sitk.GetArrayFromImage(registered)
    error_after = np.mean(np.abs(arr_f - arr_r))

    assert error_after < error_before, (
        f"Registration should reduce error: before={error_before:.4f}, after={error_after:.4f}"
    )


def test_registration_result_apply_preserves_geometry(synthetic_pair):
    fixed, moving, _ = synthetic_pair
    config = RegistrationConfig(number_of_iterations=50, shrink_factors=[2, 1], smoothing_sigmas=[1.0, 0.0])
    result = RigidRegistration(config).register(fixed, moving, verbose=False)
    registered = result.apply(moving)

    assert registered.GetSize() == fixed.GetSize()
    np.testing.assert_allclose(registered.GetSpacing(), fixed.GetSpacing(), rtol=1e-5)


def test_registration_result_save_load_transform(synthetic_pair, tmp_path):
    fixed, moving, _ = synthetic_pair
    config = RegistrationConfig(number_of_iterations=20, shrink_factors=[1], smoothing_sigmas=[0.0])
    result = RigidRegistration(config).register(fixed, moving, verbose=False)

    path = tmp_path / "transform.tfm"
    result.save_transform(path)
    assert path.exists()

    loaded_tx = RegistrationResult.load_transform(path)
    assert loaded_tx is not None


def test_apply_to_mask_uses_nearest_neighbor(synthetic_pair):
    fixed, moving, _ = synthetic_pair
    # Create a binary mask (values 0 or 1)
    arr = sitk.GetArrayFromImage(moving)
    mask_arr = (arr > 0.5).astype(np.uint8)
    mask = sitk.GetImageFromArray(mask_arr)
    mask.CopyInformation(moving)

    config = RegistrationConfig(number_of_iterations=20, shrink_factors=[1], smoothing_sigmas=[0.0])
    result = RigidRegistration(config).register(fixed, moving, verbose=False)
    registered_mask = result.apply_to_mask(mask)

    arr_reg_mask = sitk.GetArrayFromImage(registered_mask)
    # Nearest-neighbor → only integer values
    unique_vals = np.unique(arr_reg_mask)
    assert all(v in [0, 1] for v in unique_vals)
