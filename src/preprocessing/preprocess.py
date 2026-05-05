"""
Preprocessing — Resampling, Normalization, Spatial Matching
=============================================================
Medical volumes acquired on different scanners or at different time points
often differ in:
  - Voxel spacing (e.g., 0.7×0.7×5mm vs 1×1×1mm)
  - Image size
  - Intensity range and offset

All operations here work in physical space (mm) via SimpleITK to guarantee
geometric correctness independent of voxel resolution.
"""

import logging
from typing import Literal

import numpy as np
import SimpleITK as sitk

logger = logging.getLogger(__name__)


def resample_to_spacing(
    image: sitk.Image,
    target_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    interpolator=sitk.sitkLinear,
    default_pixel_value: float = 0.0,
) -> sitk.Image:
    """Resample a volume to a target isotropic voxel spacing.

    Registration is more reliable on isotropically resampled volumes because
    the optimiser treats all spatial directions equally. For CT scans, the
    typical reconstruction slice thickness (3-5mm) is much coarser than the
    in-plane resolution (0.5-0.8mm) — resampling to 1mm isotropic makes the
    volume's directional sensitivity uniform.

    Args:
        image: Input SimpleITK image (any spacing).
        target_spacing: Desired output spacing in mm (x, y, z).
        interpolator: SimpleITK interpolator. Use sitkLinear for intensity
            images and sitkNearestNeighbor for label/mask volumes.
        default_pixel_value: Value for voxels outside the original FOV.

    Returns:
        Resampled SimpleITK image at target_spacing.
    """
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    # Compute output size to preserve the physical field of view
    new_size = [
        int(round(orig_sz * orig_sp / tgt_sp))
        for orig_sz, orig_sp, tgt_sp in zip(
            original_size, original_spacing, target_spacing
        )
    ]

    logger.info(
        f"Resampling: {original_size} @ {original_spacing}mm "
        f"→ {new_size} @ {target_spacing}mm"
    )

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(default_pixel_value)
    resampler.SetTransform(sitk.Transform())  # Identity — only resampling

    return resampler.Execute(image)


def normalize_intensity(
    image: sitk.Image,
    method: Literal["zscore", "minmax", "ct_window"] = "zscore",
    ct_window: tuple[float, float] = (-1000, 3000),
    clip: bool = True,
) -> sitk.Image:
    """Normalize voxel intensities for registration similarity metrics.

    Mattes Mutual Information is metric-agnostic, but normalizing helps when
    using Mean Squares or Normalized Cross-Correlation as the similarity
    measure. Z-score normalization is recommended for paired modalities.

    Args:
        image: Input SimpleITK image.
        method:
            "zscore"    — zero-mean, unit-variance
            "minmax"    — scale to [0, 1]
            "ct_window" — clip to HU window then scale to [0, 1]
        ct_window: (min_hu, max_hu) for CT windowing.
        clip: Whether to clip outliers at ±3σ before z-score normalization.

    Returns:
        Normalized SimpleITK image (float32).
    """
    # Cast to float32 for all operations
    image = sitk.Cast(image, sitk.sitkFloat32)
    arr = sitk.GetArrayFromImage(image).astype(np.float32)

    if method == "zscore":
        if clip:
            mean, std = arr.mean(), arr.std()
            arr = np.clip(arr, mean - 3 * std, mean + 3 * std)
        arr = (arr - arr.mean()) / (arr.std() + 1e-8)

    elif method == "minmax":
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

    elif method == "ct_window":
        lo, hi = ct_window
        arr = np.clip(arr, lo, hi)
        arr = (arr - lo) / (hi - lo)

    else:
        raise ValueError(f"Unknown normalization method: {method!r}")

    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(image)
    logger.info(
        f"Normalization ({method}): range [{arr.min():.3f}, {arr.max():.3f}]"
    )
    return out


def match_spacing(
    moving: sitk.Image,
    fixed: sitk.Image,
    interpolator=sitk.sitkLinear,
) -> sitk.Image:
    """Resample `moving` to match `fixed`'s spacing, size, origin, direction.

    This is useful before feeding volumes to a deep learning model that
    expects a fixed grid size. After this operation both volumes live on the
    exact same voxel grid — no resampling is needed after the forward pass.

    Args:
        moving: Volume to resample.
        fixed: Reference volume (provides target geometry).
        interpolator: Interpolation method for the moving volume.

    Returns:
        Moving volume resampled onto fixed's grid.
    """
    logger.info("Resampling moving volume onto fixed volume's grid.")
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(0.0)
    resampler.SetTransform(sitk.Transform())
    return resampler.Execute(moving)


def crop_or_pad_to_shape(
    image: sitk.Image,
    target_shape: tuple[int, int, int],
    pad_value: float = 0.0,
) -> sitk.Image:
    """Center-crop or zero-pad a volume to a target (z, y, x) shape.

    Needed when feeding volumes to a CNN that requires fixed-size inputs.
    The volume is cropped/padded symmetrically around its center.

    Args:
        image: Input SimpleITK image.
        target_shape: Desired (z, y, x) voxel dimensions.
        pad_value: Value used for padding regions.

    Returns:
        Adjusted SimpleITK image of exactly target_shape.
    """
    arr = sitk.GetArrayFromImage(image)  # (z, y, x)
    current = np.array(arr.shape)
    target = np.array(target_shape)

    # Compute padding needed on each side
    pad = np.maximum(target - current, 0)
    pad_before = pad // 2
    pad_after = pad - pad_before

    padded = np.pad(
        arr,
        [(pad_before[i], pad_after[i]) for i in range(3)],
        constant_values=pad_value,
    )

    # Crop to target shape (center crop)
    starts = np.maximum((np.array(padded.shape) - target) // 2, 0)
    slices = tuple(slice(s, s + t) for s, t in zip(starts, target))
    cropped = padded[slices]

    out = sitk.GetImageFromArray(cropped)
    # CopyInformation requires identical sizes — copy metadata manually instead
    out.SetSpacing(image.GetSpacing())
    out.SetOrigin(image.GetOrigin())
    out.SetDirection(image.GetDirection())
    return out
