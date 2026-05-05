"""Tests for evaluation metrics (metrics.py)."""

import numpy as np
import pytest
import SimpleITK as sitk

from src.evaluation.metrics import (
    compute_dice,
    compute_tre,
    compute_jacobian_determinant,
)


# ---------------------------------------------------------------------------
# TRE
# ---------------------------------------------------------------------------

def test_tre_identity_transform():
    """With identity transform, TRE should be 0 for all landmarks."""
    n = 10
    landmarks = np.random.rand(n, 3) * 100  # random mm positions
    identity = sitk.Transform(3, sitk.sitkIdentity)

    result = compute_tre(landmarks, landmarks.copy(), identity)
    assert result["mean_mm"] < 1e-6
    assert result["max_mm"] < 1e-6


def test_tre_known_translation():
    """Translation by (5, 3, 0) mm should give TRE ≈ sqrt(25+9+0) ≈ 5.83mm
    when no transform is applied (identity)."""
    n = 5
    lm_fixed = np.zeros((n, 3))
    lm_moving = np.array([[5.0, 3.0, 0.0]] * n)
    identity = sitk.Transform(3, sitk.sitkIdentity)

    result = compute_tre(lm_fixed, lm_moving, identity)
    expected = np.sqrt(5**2 + 3**2)
    assert abs(result["mean_mm"] - expected) < 0.01


def test_tre_shape_mismatch():
    lm1 = np.zeros((5, 3))
    lm2 = np.zeros((6, 3))
    identity = sitk.Transform(3, sitk.sitkIdentity)
    with pytest.raises(ValueError, match="Landmark shape mismatch"):
        compute_tre(lm1, lm2, identity)


# ---------------------------------------------------------------------------
# Dice
# ---------------------------------------------------------------------------

def _make_mask(shape, center, radius):
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    mask = (zz - center[0])**2 + (yy - center[1])**2 + (xx - center[2])**2 <= radius**2
    arr = mask.astype(np.uint8)
    img = sitk.GetImageFromArray(arr)
    return img


def test_dice_identical_masks():
    mask = _make_mask((30, 30, 30), center=(15, 15, 15), radius=8)
    result = compute_dice(mask, mask, labels=[1])
    assert abs(result[1] - 1.0) < 1e-6


def test_dice_disjoint_masks():
    mask_a = _make_mask((30, 30, 30), center=(5, 5, 5), radius=3)
    mask_b = _make_mask((30, 30, 30), center=(25, 25, 25), radius=3)
    result = compute_dice(mask_a, mask_b, labels=[1])
    assert result[1] < 0.01


def test_dice_partial_overlap():
    mask_a = _make_mask((50, 50, 50), center=(25, 25, 25), radius=10)
    # Shift by 5 voxels
    mask_b = _make_mask((50, 50, 50), center=(30, 25, 25), radius=10)
    result = compute_dice(mask_a, mask_b, labels=[1])
    assert 0.0 < result[1] < 1.0


# ---------------------------------------------------------------------------
# Jacobian determinant
# ---------------------------------------------------------------------------

def test_jacobian_identity_field():
    """Identity displacement field → Jacobian det. = 1 everywhere."""
    size = (20, 20, 20)
    # Zero displacement = identity transform
    arr = np.zeros((*size, 3), dtype=np.float64)
    field = sitk.GetImageFromArray(arr, isVector=True)
    field.SetSpacing((1.0, 1.0, 1.0))

    result = compute_jacobian_determinant(field)
    # For zero displacement, Jacobian should be ~1
    assert abs(result["mean"] - 1.0) < 0.1
    assert result["fraction_negative"] == 0.0
