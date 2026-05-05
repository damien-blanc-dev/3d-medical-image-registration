"""
Classical Rigid & Affine Registration — SimpleITK
===================================================
Implements multi-resolution intensity-based registration using the
ITK v4 registration framework (RegistrationMethod API).

Key design decisions:
  - Physical-space transforms: all parameters are in mm / radians, not voxels.
  - Multi-resolution pyramid: coarse-to-fine prevents convergence to local
    optima when misalignment is large (common in longitudinal CT).
  - Mattes Mutual Information: modality-agnostic metric, works on CT-CT and
    CT-MRI pairs without requiring intensity correspondence.

Reference:
  Mattes et al., "PET-CT Image Registration in the Chest Using Free-Form
  Deformations", IEEE TMI 2003.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import SimpleITK as sitk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RegistrationConfig:
    """Hyperparameters for iterative intensity-based registration."""

    # Similarity metric
    metric: str = "mattes_mi"           # "mattes_mi" | "mean_squares" | "correlation"
    number_of_histogram_bins: int = 50
    sampling_strategy: str = "RANDOM"   # "RANDOM" | "REGULAR" | "NONE"
    sampling_percentage: float = 0.01

    # Optimizer
    learning_rate: float = 1.0
    min_step: float = 1e-3
    number_of_iterations: int = 200
    convergence_min_value: float = 1e-6
    convergence_window_size: int = 10

    # Multi-resolution pyramid (coarse → fine)
    shrink_factors: list[int] = field(default_factory=lambda: [4, 2, 1])
    smoothing_sigmas: list[float] = field(default_factory=lambda: [2.0, 1.0, 0.0])


# ---------------------------------------------------------------------------
# Rigid Registration (6 DoF: 3 rotations + 3 translations)
# ---------------------------------------------------------------------------

class RigidRegistration:
    """Multi-resolution rigid registration using SimpleITK.

    Euler3DTransform parameterizes rigid body motion as 3 rotation angles
    (roll, pitch, yaw) + 3 translation components. With 6 degrees of freedom,
    it is appropriate when the anatomy is assumed not to deform (e.g., bone,
    brain in the same session).

    Usage:
        reg = RigidRegistration(config)
        result = reg.register(fixed, moving)
        registered = result.apply(moving)
    """

    def __init__(self, config: RegistrationConfig | None = None):
        self.config = config or RegistrationConfig()

    def register(
        self,
        fixed: sitk.Image,
        moving: sitk.Image,
        initial_transform: sitk.Transform | None = None,
        verbose: bool = True,
    ) -> "RegistrationResult":
        """Run rigid registration.

        Args:
            fixed: Reference (target) volume.
            moving: Volume to be aligned to fixed.
            initial_transform: Optional warm-start transform. If None, the
                center-of-mass alignment is used as initialization.
            verbose: Print per-iteration metric values.

        Returns:
            RegistrationResult with the optimized transform and metadata.
        """
        logger.info("Starting rigid registration (Euler3D, 6 DoF)...")
        cfg = self.config

        # Cast both to float32 for metric computation
        fixed_f = sitk.Cast(fixed, sitk.sitkFloat32)
        moving_f = sitk.Cast(moving, sitk.sitkFloat32)

        # Initialize registration method
        method = sitk.ImageRegistrationMethod()

        # --- Similarity metric ---
        self._set_metric(method, cfg)

        # --- Optimizer ---
        method.SetOptimizerAsGradientDescentLineSearch(
            learningRate=cfg.learning_rate,
            numberOfIterations=cfg.number_of_iterations,
            convergenceMinimumValue=cfg.convergence_min_value,
            convergenceWindowSize=cfg.convergence_window_size,
        )
        method.SetOptimizerScalesFromPhysicalShift()

        # --- Initial transform ---
        if initial_transform is None:
            initial_transform = sitk.CenteredTransformInitializer(
                fixed_f,
                moving_f,
                sitk.Euler3DTransform(),
                sitk.CenteredTransformInitializerFilter.MOMENTS,
            )
        method.SetInitialTransform(initial_transform, inPlace=False)

        # --- Multi-resolution pyramid ---
        method.SetShrinkFactorsPerLevel(cfg.shrink_factors)
        method.SetSmoothingSigmasPerLevel(cfg.smoothing_sigmas)
        method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        method.SetInterpolator(sitk.sitkLinear)

        # --- Iteration callback for logging ---
        metric_values: list[float] = []
        if verbose:
            method.AddCommand(
                sitk.sitkIterationEvent,
                lambda: _log_iteration(method, metric_values),
            )

        # --- Execute ---
        final_transform = method.Execute(fixed_f, moving_f)

        logger.info(
            f"Rigid registration converged in {method.GetOptimizerIteration()} iterations"
            f"\n  Final metric: {method.GetMetricValue():.6f}"
            f"\n  Stop condition: {method.GetOptimizerStopConditionDescription()}"
        )

        return RegistrationResult(
            transform=final_transform,
            fixed=fixed,
            moving=moving,
            metric_values=metric_values,
            method_name="Rigid (Euler3D)",
        )


# ---------------------------------------------------------------------------
# Affine Registration (12 DoF: rotation + translation + scale + shear)
# ---------------------------------------------------------------------------

class AffineRegistration:
    """Multi-resolution affine registration (12 DoF).

    Affine extends rigid by adding anisotropic scaling and shear, useful
    when correcting for scanner-related geometric distortions or when
    the organs being registered can undergo slight shape changes between
    acquisitions (e.g., breathing state in abdominal CT).

    Not diffeomorphic — does not guarantee topology preservation.
    """

    def __init__(self, config: RegistrationConfig | None = None):
        self.config = config or RegistrationConfig()

    def register(
        self,
        fixed: sitk.Image,
        moving: sitk.Image,
        initial_transform: sitk.Transform | None = None,
        verbose: bool = True,
    ) -> "RegistrationResult":
        """Run affine registration. Same API as RigidRegistration."""
        logger.info("Starting affine registration (12 DoF)...")
        cfg = self.config

        fixed_f = sitk.Cast(fixed, sitk.sitkFloat32)
        moving_f = sitk.Cast(moving, sitk.sitkFloat32)

        method = sitk.ImageRegistrationMethod()
        self._set_metric(method, cfg)

        method.SetOptimizerAsGradientDescentLineSearch(
            learningRate=cfg.learning_rate,
            numberOfIterations=cfg.number_of_iterations,
            convergenceMinimumValue=cfg.convergence_min_value,
            convergenceWindowSize=cfg.convergence_window_size,
        )
        method.SetOptimizerScalesFromPhysicalShift()

        if initial_transform is None:
            initial_transform = sitk.CenteredTransformInitializer(
                fixed_f,
                moving_f,
                sitk.AffineTransform(3),
                sitk.CenteredTransformInitializerFilter.MOMENTS,
            )
        method.SetInitialTransform(initial_transform, inPlace=False)

        method.SetShrinkFactorsPerLevel(cfg.shrink_factors)
        method.SetSmoothingSigmasPerLevel(cfg.smoothing_sigmas)
        method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        method.SetInterpolator(sitk.sitkLinear)

        metric_values: list[float] = []
        if verbose:
            method.AddCommand(
                sitk.sitkIterationEvent,
                lambda: _log_iteration(method, metric_values),
            )

        final_transform = method.Execute(fixed_f, moving_f)

        logger.info(
            f"Affine registration converged in {method.GetOptimizerIteration()} iterations"
            f"\n  Final metric: {method.GetMetricValue():.6f}"
        )

        return RegistrationResult(
            transform=final_transform,
            fixed=fixed,
            moving=moving,
            metric_values=metric_values,
            method_name="Affine (12 DoF)",
        )

    def _set_metric(self, method: sitk.ImageRegistrationMethod, cfg: RegistrationConfig):
        """Attach the configured similarity metric to the registration method."""
        if cfg.metric == "mattes_mi":
            method.SetMetricAsMattesMutualInformation(
                numberOfHistogramBins=cfg.number_of_histogram_bins
            )
        elif cfg.metric == "mean_squares":
            method.SetMetricAsMeanSquares()
        elif cfg.metric == "correlation":
            method.SetMetricAsCorrelation()
        else:
            raise ValueError(f"Unknown metric: {cfg.metric!r}")

        strategy_map = {
            "RANDOM": sitk.ImageRegistrationMethod.RANDOM,
            "REGULAR": sitk.ImageRegistrationMethod.REGULAR,
            "NONE": sitk.ImageRegistrationMethod.NONE,
        }
        method.SetMetricSamplingStrategy(strategy_map[cfg.sampling_strategy])
        method.SetMetricSamplingPercentage(cfg.sampling_percentage)


# Patch _set_metric onto RigidRegistration (same implementation)
RigidRegistration._set_metric = AffineRegistration._set_metric


# ---------------------------------------------------------------------------
# Registration Result
# ---------------------------------------------------------------------------

@dataclass
class RegistrationResult:
    """Container for registration output: transform + diagnostic data."""

    transform: sitk.Transform
    fixed: sitk.Image
    moving: sitk.Image
    metric_values: list[float]
    method_name: str

    def apply(
        self,
        image: sitk.Image,
        interpolator=sitk.sitkLinear,
        default_pixel_value: float = 0.0,
    ) -> sitk.Image:
        """Apply the optimized transform to resample `image` onto fixed's grid.

        Args:
            image: Volume to warp (typically the moving volume or its mask).
            interpolator: Use sitkLinear for intensity, sitkNearestNeighbor
                for label volumes (preserves integer labels).
            default_pixel_value: Fill value for voxels outside the FOV.

        Returns:
            Resampled image aligned with the fixed volume's geometry.
        """
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(self.fixed)
        resampler.SetTransform(self.transform)
        resampler.SetInterpolator(interpolator)
        resampler.SetDefaultPixelValue(default_pixel_value)
        return resampler.Execute(image)

    def apply_to_mask(self, mask: sitk.Image) -> sitk.Image:
        """Apply transform to a binary mask using nearest-neighbor interpolation."""
        return self.apply(mask, interpolator=sitk.sitkNearestNeighbor)

    def save_transform(self, path: str | Path) -> None:
        """Persist the transform to disk (.tfm or .txt format)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteTransform(self.transform, str(path))
        logger.info(f"Transform saved to: {path}")

    @classmethod
    def load_transform(cls, path: str | Path) -> sitk.Transform:
        """Load a previously saved ITK transform file."""
        return sitk.ReadTransform(str(path))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _log_iteration(
    method: sitk.ImageRegistrationMethod,
    metric_values: list[float],
) -> None:
    """Callback executed at each optimizer iteration."""
    value = method.GetMetricValue()
    metric_values.append(value)

    if method.GetOptimizerIteration() % 20 == 0:
        logger.debug(
            f"  Iter {method.GetOptimizerIteration():4d} | "
            f"Metric: {value:.6f} | "
            f"Position: {[f'{p:.3f}' for p in method.GetOptimizerPosition()]}"
        )
