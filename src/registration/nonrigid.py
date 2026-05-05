"""
Non-Rigid Registration — B-Spline FFD & Demons (SimpleITK)
============================================================
Non-rigid (deformable) registration estimates a dense displacement field
(DDF) or a parametric deformation that maps every voxel of the moving image
to a corresponding location in the fixed image.

Two classical algorithms are implemented:

1. **B-Spline Free-Form Deformation (FFD)**
   Deformation is parameterized by a sparse grid of control points with
   B-Spline basis functions. The control point grid spacing controls the
   trade-off between flexibility (small spacing = more local deformation)
   and regularity (large spacing = smoother, more global deformation).
   Reference: Rueckert et al., IEEE TMI 1999.

2. **Demons**
   Optical flow-inspired iterative algorithm. Each iteration computes a
   "force" image from intensity gradients and applies Gaussian smoothing
   as an implicit regularizer. Fast but less principled than FFD.
   Reference: Thirion, Medical Image Analysis 1998.

Both are suitable warm-start candidates for the deep learning model in Step 2.
"""

import logging
from dataclasses import dataclass, field

import SimpleITK as sitk

from .rigid import RegistrationResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# B-Spline Free-Form Deformation
# ---------------------------------------------------------------------------

@dataclass
class BSplineConfig:
    """Configuration for B-Spline non-rigid registration."""
    # Control point grid spacing in mm (smaller = more local deformation)
    grid_spacing_mm: float = 50.0
    # Similarity metric
    metric: str = "mattes_mi"
    number_of_histogram_bins: int = 50
    sampling_strategy: str = "RANDOM"
    sampling_percentage: float = 0.01
    # Optimizer
    number_of_iterations: int = 100
    learning_rate: float = 5.0
    convergence_min_value: float = 1e-6
    convergence_window_size: int = 10
    # Multi-resolution
    shrink_factors: list[int] = field(default_factory=lambda: [4, 2, 1])
    smoothing_sigmas: list[float] = field(default_factory=lambda: [2.0, 1.0, 0.0])


class BSplineRegistration:
    """B-Spline Free-Form Deformation registration.

    Typically initialized with a rigid pre-registration result to ensure
    global alignment before fitting local deformations.

    Usage:
        rigid_result = RigidRegistration().register(fixed, moving)
        bspline = BSplineRegistration()
        result = bspline.register(fixed, moving, rigid_result.transform)
    """

    def __init__(self, config: BSplineConfig | None = None):
        self.config = config or BSplineConfig()

    def register(
        self,
        fixed: sitk.Image,
        moving: sitk.Image,
        initial_transform: sitk.Transform | None = None,
        verbose: bool = True,
    ) -> RegistrationResult:
        """Run B-Spline non-rigid registration.

        Args:
            fixed: Reference volume.
            moving: Volume to deform.
            initial_transform: Rigid/affine pre-alignment (strongly recommended).
            verbose: Log iteration details.

        Returns:
            RegistrationResult with a BSplineTransform.
        """
        logger.info(
            f"Starting B-Spline FFD registration "
            f"(grid spacing={self.config.grid_spacing_mm}mm)..."
        )
        cfg = self.config

        fixed_f = sitk.Cast(fixed, sitk.sitkFloat32)
        moving_f = sitk.Cast(moving, sitk.sitkFloat32)

        # Build the B-Spline control point grid
        transform_domain_mesh_size = [
            max(1, int(sz * sp / cfg.grid_spacing_mm))
            for sz, sp in zip(fixed.GetSize(), fixed.GetSpacing())
        ]
        logger.info(f"  Control point grid: {transform_domain_mesh_size}")

        bspline_tx = sitk.BSplineTransformInitializer(
            image1=fixed_f,
            transformDomainMeshSize=transform_domain_mesh_size,
            order=3,
        )

        method = sitk.ImageRegistrationMethod()

        # Compose initial_transform (rigid pre-alignment) + B-Spline
        if initial_transform is not None:
            composite = sitk.CompositeTransform(3)
            composite.AddTransform(initial_transform)
            composite.AddTransform(bspline_tx)
            method.SetInitialTransformAsBSpline(
                bspline_tx,
                inPlace=True,
                scaleFactors=cfg.shrink_factors,
            )
            method.SetMovingInitialTransform(initial_transform)
        else:
            method.SetInitialTransformAsBSpline(
                bspline_tx,
                inPlace=True,
                scaleFactors=cfg.shrink_factors,
            )

        # Metric
        if cfg.metric == "mattes_mi":
            method.SetMetricAsMattesMutualInformation(cfg.number_of_histogram_bins)
        elif cfg.metric == "mean_squares":
            method.SetMetricAsMeanSquares()

        strategy_map = {
            "RANDOM": sitk.ImageRegistrationMethod.RANDOM,
            "REGULAR": sitk.ImageRegistrationMethod.REGULAR,
        }
        method.SetMetricSamplingStrategy(strategy_map.get(cfg.sampling_strategy, sitk.ImageRegistrationMethod.RANDOM))
        method.SetMetricSamplingPercentage(cfg.sampling_percentage)

        # Optimizer — LBFGS2 is better suited for B-Spline's high DoF
        method.SetOptimizerAsLBFGS2(
            solutionAccuracy=1e-5,
            numberOfIterations=cfg.number_of_iterations,
            deltaConvergenceTolerance=cfg.convergence_min_value,
        )

        method.SetShrinkFactorsPerLevel(cfg.shrink_factors)
        method.SetSmoothingSigmasPerLevel(cfg.smoothing_sigmas)
        method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        method.SetInterpolator(sitk.sitkLinear)

        metric_values: list[float] = []
        if verbose:
            method.AddCommand(
                sitk.sitkIterationEvent,
                lambda: metric_values.append(method.GetMetricValue()),
            )

        final_transform = method.Execute(fixed_f, moving_f)

        logger.info(
            f"B-Spline registration done. "
            f"Final metric: {method.GetMetricValue():.6f}"
        )

        return RegistrationResult(
            transform=final_transform,
            fixed=fixed,
            moving=moving,
            metric_values=metric_values,
            method_name=f"B-Spline FFD (grid={cfg.grid_spacing_mm}mm)",
        )


# ---------------------------------------------------------------------------
# Demons Registration
# ---------------------------------------------------------------------------

@dataclass
class DemonsConfig:
    """Configuration for Demons deformable registration."""
    # Number of iterations at each resolution level
    iterations_per_level: list[int] = field(default_factory=lambda: [200, 100, 50])
    # Gaussian smoothing of the displacement field (mm)
    displacement_field_sigma: float = 2.0
    # Gaussian smoothing of the update field (mm); 0 = no update smoothing
    update_field_sigma: float = 0.5
    # Demons variant: "diffeomorphic" | "symmetric" | "fast_symmetric"
    variant: str = "diffeomorphic"


class DemonsRegistration:
    """Demons optical-flow deformable registration.

    Faster than B-Spline for monomodal (same-scanner) pairs, but less
    accurate in the presence of large deformations. Produces a dense
    displacement field (DVF) directly.

    Args:
        config: DemonsConfig hyperparameters.
    """

    def __init__(self, config: DemonsConfig | None = None):
        self.config = config or DemonsConfig()

    def register(
        self,
        fixed: sitk.Image,
        moving: sitk.Image,
        initial_transform: sitk.Transform | None = None,
    ) -> "DemonsResult":
        """Run Demons registration.

        Args:
            fixed: Reference volume.
            moving: Volume to deform.
            initial_transform: Optional rigid pre-alignment to apply first.

        Returns:
            DemonsResult containing the displacement field image.
        """
        logger.info(f"Starting Demons registration (variant={self.config.variant})...")
        cfg = self.config

        # Pre-apply initial transform if provided
        if initial_transform is not None:
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(fixed)
            resampler.SetTransform(initial_transform)
            resampler.SetInterpolator(sitk.sitkLinear)
            moving = resampler.Execute(moving)

        fixed_f = sitk.Cast(fixed, sitk.sitkFloat32)
        moving_f = sitk.Cast(moving, sitk.sitkFloat32)

        # Select Demons filter variant
        if cfg.variant == "diffeomorphic":
            demons = sitk.DiffeomorphicDemonsRegistrationFilter()
        elif cfg.variant == "symmetric":
            demons = sitk.SymmetricForcesDemonsRegistrationFilter()
        else:
            demons = sitk.FastSymmetricForcesDemonsRegistrationFilter()

        demons.SetNumberOfIterations(cfg.iterations_per_level[-1])
        demons.SetStandardDeviations(cfg.displacement_field_sigma)
        demons.SetMaximumUpdateStepLength(0.5)

        displacement_field = demons.Execute(fixed_f, moving_f)

        logger.info(
            f"Demons done. RMS: {demons.GetRMSChange():.4f} | "
            f"Metric: {demons.GetMetric():.4f}"
        )

        return DemonsResult(
            displacement_field=displacement_field,
            fixed=fixed,
            moving=moving,
            method_name=f"Demons ({cfg.variant})",
        )


@dataclass
class DemonsResult:
    """Container for Demons registration output (displacement field)."""
    displacement_field: sitk.Image
    fixed: sitk.Image
    moving: sitk.Image
    method_name: str

    def apply(self, image: sitk.Image, interpolator=sitk.sitkLinear) -> sitk.Image:
        """Warp `image` using the estimated displacement field."""
        warp_filter = sitk.WarpImageFilter()
        warp_filter.SetInterpolator(interpolator)
        warp_filter.SetOutputParameteresFromImage(self.fixed)
        return warp_filter.Execute(image, self.displacement_field)

    def apply_to_mask(self, mask: sitk.Image) -> sitk.Image:
        return self.apply(mask, sitk.sitkNearestNeighbor)

    def get_displacement_magnitude(self) -> sitk.Image:
        """Compute voxel-wise displacement magnitude (mm)."""
        return sitk.VectorMagnitude(self.displacement_field)
