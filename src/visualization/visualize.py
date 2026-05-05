"""
Visualization — 2D Slices, Overlays, Checkerboard, Deformation Fields
=======================================================================
All plotting functions accept SimpleITK images and produce matplotlib figures.
Figures can be saved to disk or displayed interactively.

The checkerboard pattern is the standard qualitative tool for assessing
alignment: misregistered regions show discontinuities at checkerboard edges,
while well-aligned regions are visually seamless.
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import SimpleITK as sitk

logger = logging.getLogger(__name__)


def _to_array(image: sitk.Image) -> np.ndarray:
    """Convert SimpleITK image to numpy (z, y, x) float32 array."""
    return sitk.GetArrayFromImage(image).astype(np.float32)


def _select_slices(
    array: np.ndarray,
    slice_indices: Optional[list[int]] = None,
) -> list[int]:
    """Select slice indices; default to quartile slices if not specified."""
    depth = array.shape[0]
    if slice_indices is None:
        return [depth // 4, depth // 2, 3 * depth // 4]
    return slice_indices


# ---------------------------------------------------------------------------
# Three-panel comparison: fixed / moving / registered
# ---------------------------------------------------------------------------

def plot_slices_comparison(
    fixed: sitk.Image,
    moving: sitk.Image,
    registered: sitk.Image,
    slice_indices: Optional[list[int]] = None,
    titles: tuple[str, str, str] = ("Fixed", "Moving", "Registered"),
    cmap: str = "gray",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot axial slices of fixed, moving, and registered volumes side-by-side.

    Shows 3 rows (one per axial slice) × 3 columns (fixed / moving / registered).

    Args:
        fixed: Reference volume.
        moving: Original moving volume.
        registered: Moving volume after registration.
        slice_indices: Axial slice indices to display. Default: quartiles.
        titles: Column labels.
        cmap: Colormap.
        save_path: If provided, save figure to this path.

    Returns:
        Matplotlib Figure object.
    """
    arr_f = _to_array(fixed)
    arr_m = _to_array(moving)
    arr_r = _to_array(registered)

    slices = _select_slices(arr_f, slice_indices)
    n_slices = len(slices)

    fig, axes = plt.subplots(
        n_slices, 3, figsize=(12, 4 * n_slices),
        gridspec_kw={"hspace": 0.05, "wspace": 0.05},
    )
    if n_slices == 1:
        axes = axes[np.newaxis, :]

    for row, z in enumerate(slices):
        for col, (arr, title) in enumerate(
            zip([arr_f, arr_m, arr_r], titles)
        ):
            ax = axes[row, col]
            # Clip and normalize for display
            vmin, vmax = np.percentile(arr, [1, 99])
            ax.imshow(arr[z], cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
            ax.axis("off")
            if row == 0:
                ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
            if col == 0:
                ax.set_ylabel(f"z={z}", fontsize=9, rotation=0, labelpad=35, va="center")

    fig.suptitle("Registration Result — Axial Slices", fontsize=14, y=1.01)
    _maybe_save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Checkerboard overlay
# ---------------------------------------------------------------------------

def plot_checkerboard(
    fixed: sitk.Image,
    registered: sitk.Image,
    pattern_size: int = 5,
    slice_indices: Optional[list[int]] = None,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Generate a checkerboard overlay of fixed and registered volumes.

    Alternating blocks show regions from the fixed and registered volumes.
    Perfect alignment → seamless transitions at block boundaries.

    Args:
        fixed: Reference volume.
        registered: Registered moving volume (should match fixed's grid).
        pattern_size: Number of voxels per checkerboard square.
        slice_indices: Axial slices to display.
        save_path: Output file path.

    Returns:
        Matplotlib Figure.
    """
    # Use SimpleITK's built-in checkerboard filter
    checker_filter = sitk.CheckerBoardImageFilter()
    checker_filter.SetCheckerPattern([pattern_size] * fixed.GetDimension())

    # Normalize both volumes to [0, 255] before checkerboard
    norm_fixed = sitk.RescaleIntensity(
        sitk.Cast(fixed, sitk.sitkFloat32), 0, 255
    )
    norm_reg = sitk.RescaleIntensity(
        sitk.Cast(registered, sitk.sitkFloat32), 0, 255
    )

    checker = checker_filter.Execute(norm_fixed, norm_reg)
    arr = _to_array(checker)

    slices = _select_slices(arr, slice_indices)
    n_slices = len(slices)

    fig, axes = plt.subplots(
        1, n_slices, figsize=(5 * n_slices, 5),
        gridspec_kw={"wspace": 0.05},
    )
    if n_slices == 1:
        axes = [axes]

    for ax, z in zip(axes, slices):
        ax.imshow(arr[z], cmap="gray", origin="lower")
        ax.axis("off")
        ax.set_title(f"z={z}", fontsize=10)

    fig.suptitle(
        "Checkerboard (Fixed ↔ Registered)\n"
        "Seamless boundaries = good alignment",
        fontsize=12,
    )
    _maybe_save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Color-coded overlay (fixed in red, moving in cyan)
# ---------------------------------------------------------------------------

def save_overlay(
    fixed: sitk.Image,
    registered: sitk.Image,
    slice_indices: Optional[list[int]] = None,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """RGB overlay: fixed channel=red, registered channel=cyan (green+blue).

    Perfect alignment → gray image (red + cyan = gray).
    Misalignment → colored residuals.

    Args:
        fixed: Reference volume (red channel).
        registered: Registered volume (cyan = green + blue channels).
        slice_indices: Axial slices.
        save_path: Output path.

    Returns:
        Matplotlib Figure.
    """
    arr_f = _to_array(fixed)
    arr_r = _to_array(registered)

    # Normalize to [0, 1]
    def _norm(a):
        lo, hi = np.percentile(a, [1, 99])
        return np.clip((a - lo) / (hi - lo + 1e-8), 0, 1)

    arr_f_n = _norm(arr_f)
    arr_r_n = _norm(arr_r)

    slices = _select_slices(arr_f_n, slice_indices)

    fig, axes = plt.subplots(1, len(slices), figsize=(5 * len(slices), 5))
    if len(slices) == 1:
        axes = [axes]

    for ax, z in zip(axes, slices):
        # R=fixed, G=registered, B=registered  →  red vs cyan overlay
        rgb = np.stack([arr_f_n[z], arr_r_n[z], arr_r_n[z]], axis=-1)
        ax.imshow(rgb, origin="lower")
        ax.axis("off")
        ax.set_title(f"z={z}", fontsize=10)

    fig.suptitle(
        "Color Overlay (Red=Fixed, Cyan=Registered)\n"
        "Gray → aligned, Colors → residual misalignment",
        fontsize=11,
    )
    _maybe_save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Deformation field visualization
# ---------------------------------------------------------------------------

def plot_deformation_field(
    displacement_field: sitk.Image,
    slice_index: Optional[int] = None,
    stride: int = 8,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Quiver plot of a 2D slice of the 3D displacement field.

    Args:
        displacement_field: SimpleITK vector image (output of Demons /
            DisplacementFieldTransform), shape (z, y, x, 3).
        slice_index: Axial slice to display. Default: mid-slice.
        stride: Show every `stride`-th vector to avoid clutter.
        save_path: Output path.

    Returns:
        Matplotlib Figure.
    """
    arr = sitk.GetArrayFromImage(displacement_field)  # (z, y, x, 3)

    z = slice_index if slice_index is not None else arr.shape[0] // 2
    slice_2d = arr[z]  # (y, x, 3)

    # Magnitude for background
    magnitude = np.linalg.norm(slice_2d, axis=-1)

    y_idx, x_idx = np.mgrid[0:slice_2d.shape[0], 0:slice_2d.shape[1]]
    dy = slice_2d[:, :, 1]  # y-component
    dx = slice_2d[:, :, 0]  # x-component

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(magnitude, cmap="hot", origin="lower", alpha=0.7)
    ax.quiver(
        x_idx[::stride, ::stride],
        y_idx[::stride, ::stride],
        dx[::stride, ::stride],
        dy[::stride, ::stride],
        magnitude[::stride, ::stride],
        cmap="cool",
        scale=None,
        scale_units="xy",
        angles="xy",
        alpha=0.8,
    )
    ax.set_title(
        f"Displacement Field — Axial slice z={z}\n"
        f"(background=magnitude, arrows=direction)",
        fontsize=11,
    )
    ax.axis("off")
    _maybe_save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Metric convergence plot
# ---------------------------------------------------------------------------

def plot_metric_convergence(
    metric_values: list[float],
    method_name: str = "Registration",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot the optimizer metric value across iterations.

    Args:
        metric_values: List of metric values recorded per iteration.
        method_name: Title label.
        save_path: Output path.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(metric_values, color="steelblue", linewidth=1.5)
    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("Metric value", fontsize=11)
    ax.set_title(f"{method_name} — Convergence", fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _maybe_save(fig: plt.Figure, path: Optional[str | Path]) -> None:
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"Figure saved to: {path}")
