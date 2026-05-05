"""
Registration Evaluation Metrics
================================
Three complementary metrics for quantitative registration assessment:

1. **TRE** (Target Registration Error)
   Gold-standard metric when anatomical landmarks are available.
   Measures the Euclidean distance (mm) between corresponding point pairs
   after applying the transform — directly interpretable in clinical terms.
   A TRE < 2mm is typically acceptable for surgical guidance applications.

2. **Dice coefficient**
   When segmentation masks are available, Dice measures the volumetric
   overlap of corresponding structures before vs. after registration.
   Dice = 1 means perfect overlap.

3. **Jacobian determinant**
   For deformable registration only. The Jacobian det. at each voxel
   measures local volume change:
     - det(J) = 1 → isometric (no local volume change)
     - det(J) > 0 → topology preserved
     - det(J) ≤ 0 → folding (non-physical; topology broken)
   The fraction of voxels with det(J) ≤ 0 quantifies the
   "non-diffeomorphic" fraction of the deformation.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TRE — Target Registration Error
# ---------------------------------------------------------------------------

def compute_tre(
    landmarks_fixed: np.ndarray,
    landmarks_moving: np.ndarray,
    transform: sitk.Transform,
) -> dict[str, float]:
    """Compute TRE between corresponding landmark pairs.

    Args:
        landmarks_fixed: (N, 3) array of fixed-image landmarks in physical
            space (mm), LPS convention: (x, y, z).
        landmarks_moving: (N, 3) array of corresponding moving-image
            landmarks in physical space (mm).
        transform: The registration transform (maps moving → fixed coords).

    Returns:
        Dict with 'mean', 'std', 'median', 'max', 'min' TRE in mm.

    Raises:
        ValueError: If landmark arrays have different shapes.
    """
    if landmarks_fixed.shape != landmarks_moving.shape:
        raise ValueError(
            f"Landmark shape mismatch: {landmarks_fixed.shape} vs {landmarks_moving.shape}"
        )

    n = len(landmarks_fixed)
    distances = np.zeros(n)

    for i, (pt_fixed, pt_moving) in enumerate(
        zip(landmarks_fixed, landmarks_moving)
    ):
        # Apply transform: maps a moving-space point to fixed-space
        pt_transformed = np.array(transform.TransformPoint(pt_moving.tolist()))
        distances[i] = np.linalg.norm(pt_transformed - pt_fixed)

    result = {
        "n_landmarks": n,
        "mean_mm": float(distances.mean()),
        "std_mm": float(distances.std()),
        "median_mm": float(np.median(distances)),
        "max_mm": float(distances.max()),
        "min_mm": float(distances.min()),
    }

    logger.info(
        f"TRE ({n} landmarks): "
        f"{result['mean_mm']:.2f} ± {result['std_mm']:.2f} mm "
        f"(median={result['median_mm']:.2f}, max={result['max_mm']:.2f})"
    )
    return result


def load_landmarks_csv(path: str | Path) -> np.ndarray:
    """Load landmarks from a CSV file with columns: x, y, z (mm, LPS).

    Args:
        path: Path to CSV file. Expected columns: x, y, z.

    Returns:
        (N, 3) float array.
    """
    df = pd.read_csv(path)
    required = {"x", "y", "z"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must have columns {required}. Got: {list(df.columns)}")
    return df[["x", "y", "z"]].to_numpy(dtype=np.float64)


# ---------------------------------------------------------------------------
# Dice Coefficient
# ---------------------------------------------------------------------------

def compute_dice(
    mask_fixed: sitk.Image,
    mask_registered: sitk.Image,
    labels: list[int] | None = None,
) -> dict[str | int, float]:
    """Compute Dice similarity coefficient for one or more label classes.

    Args:
        mask_fixed: Ground-truth segmentation on the fixed image.
        mask_registered: Moving segmentation after registration (warped).
        labels: List of integer label values to evaluate. If None, all
            non-zero labels found in mask_fixed are used.

    Returns:
        Dict mapping label_id → Dice score.
    """
    arr_fixed = sitk.GetArrayFromImage(mask_fixed).astype(np.int32)
    arr_reg = sitk.GetArrayFromImage(mask_registered).astype(np.int32)

    if labels is None:
        labels = [int(l) for l in np.unique(arr_fixed) if l != 0]

    results: dict[str | int, float] = {}

    for label in labels:
        gt = arr_fixed == label
        pred = arr_reg == label

        intersection = np.logical_and(gt, pred).sum()
        union = gt.sum() + pred.sum()

        dice = (2.0 * intersection) / (union + 1e-8) if union > 0 else 1.0
        results[label] = float(dice)
        logger.info(f"  Dice label={label}: {dice:.4f}")

    if len(labels) > 1:
        results["mean"] = float(np.mean(list(results.values())))
        logger.info(f"  Mean Dice: {results['mean']:.4f}")

    return results


def compute_dice_before_after(
    mask_fixed: sitk.Image,
    mask_moving_original: sitk.Image,
    mask_moving_registered: sitk.Image,
    labels: list[int] | None = None,
) -> pd.DataFrame:
    """Compare Dice before and after registration for all labels.

    Args:
        mask_fixed: Fixed image segmentation (reference).
        mask_moving_original: Moving segmentation before registration.
        mask_moving_registered: Moving segmentation after registration.
        labels: Label IDs to evaluate.

    Returns:
        DataFrame with columns: label, dice_before, dice_after, improvement.
    """
    dice_before = compute_dice(mask_fixed, mask_moving_original, labels)
    dice_after = compute_dice(mask_fixed, mask_moving_registered, labels)

    rows = []
    for label in dice_before:
        if label == "mean":
            continue
        b = dice_before[label]
        a = dice_after[label]
        rows.append({
            "label": label,
            "dice_before": b,
            "dice_after": a,
            "improvement": a - b,
        })

    df = pd.DataFrame(rows)
    logger.info(f"\nDice improvement summary:\n{df.to_string(index=False)}")
    return df


# ---------------------------------------------------------------------------
# Jacobian Determinant (deformable registration quality)
# ---------------------------------------------------------------------------

def compute_jacobian_determinant(
    displacement_field: sitk.Image,
) -> dict[str, float]:
    """Compute statistics of the Jacobian determinant of a displacement field.

    A Jacobian determinant ≤ 0 indicates a non-diffeomorphic (folded)
    deformation — physically impossible and a sign of over-regularization
    or optimization failure.

    Args:
        displacement_field: SimpleITK vector image (displacement field),
            as produced by Demons or DisplacementFieldTransform.

    Returns:
        Dict with statistics: mean, std, min, max, fraction_negative.
    """
    jac_filter = sitk.DisplacementFieldJacobianDeterminantFilter()
    jac_image = jac_filter.Execute(displacement_field)
    jac_arr = sitk.GetArrayFromImage(jac_image)

    n_total = jac_arr.size
    n_negative = int((jac_arr <= 0).sum())
    fraction_negative = n_negative / n_total

    result = {
        "mean": float(jac_arr.mean()),
        "std": float(jac_arr.std()),
        "min": float(jac_arr.min()),
        "max": float(jac_arr.max()),
        "fraction_negative": fraction_negative,
        "n_folded_voxels": n_negative,
    }

    if fraction_negative > 0.01:
        logger.warning(
            f"Jacobian det.: {fraction_negative:.2%} of voxels have det ≤ 0 "
            f"({n_negative}/{n_total}) — deformation is not diffeomorphic."
        )
    else:
        logger.info(
            f"Jacobian det.: mean={result['mean']:.3f}, "
            f"min={result['min']:.3f}, folded={fraction_negative:.4%}"
        )

    return result
