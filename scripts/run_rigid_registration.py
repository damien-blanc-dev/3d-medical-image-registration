"""
End-to-end rigid registration pipeline
=======================================
Usage:
    python scripts/run_rigid_registration.py \\
        --fixed data/raw/fixed.nii.gz \\
        --moving data/raw/moving.nii.gz \\
        --output results/ \\
        --config configs/rigid_registration.yaml

This script orchestrates:
  1. Volume loading (SimpleITK)
  2. Preprocessing (resampling to isotropic spacing, intensity normalization)
  3. Multi-resolution rigid registration (Euler3D, 6 DoF)
  4. Optional affine refinement (12 DoF)
  5. Result saving (registered volume + ITK transform)
  6. Visualization (slices, checkerboard, convergence plot)
  7. Evaluation (Dice if masks provided, TRE if landmarks provided)
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.io.loaders import load_volume, save_volume
from src.preprocessing.preprocess import resample_to_spacing, normalize_intensity
from src.registration.rigid import RigidRegistration, AffineRegistration, RegistrationConfig
from src.evaluation.metrics import (
    compute_dice_before_after,
    compute_tre,
    load_landmarks_csv,
)
from src.visualization.visualize import (
    plot_slices_comparison,
    plot_checkerboard,
    save_overlay,
    plot_metric_convergence,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="3D rigid registration pipeline (SimpleITK)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--fixed", required=True, help="Fixed (reference) volume path")
    parser.add_argument("--moving", required=True, help="Moving volume path")
    parser.add_argument("--output", default="results/", help="Output directory")
    parser.add_argument("--config", default="configs/rigid_registration.yaml")

    # Optional inputs
    parser.add_argument("--fixed-mask", default=None, help="Segmentation mask for fixed volume")
    parser.add_argument("--moving-mask", default=None, help="Segmentation mask for moving volume")
    parser.add_argument("--landmarks-fixed", default=None, help="CSV landmarks for fixed (x,y,z mm)")
    parser.add_argument("--landmarks-moving", default=None, help="CSV landmarks for moving (x,y,z mm)")

    # Registration options
    parser.add_argument(
        "--method", choices=["rigid", "affine", "rigid+affine"],
        default="rigid", help="Registration method"
    )
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    cfg_dict = load_config(args.config)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load volumes
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 1 — Loading volumes")
    logger.info("=" * 60)

    fixed_raw = load_volume(args.fixed)
    moving_raw = load_volume(args.moving)

    # ------------------------------------------------------------------
    # 2. Preprocessing
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 2 — Preprocessing")
    logger.info("=" * 60)

    pre_cfg = cfg_dict.get("preprocessing", {})
    target_spacing = tuple(pre_cfg.get("target_spacing", [1.0, 1.0, 1.0]))
    norm_method = pre_cfg.get("normalization", "zscore")

    import SimpleITK as sitk

    fixed = resample_to_spacing(fixed_raw, target_spacing)
    moving = resample_to_spacing(moving_raw, target_spacing)

    fixed_norm = normalize_intensity(fixed, method=norm_method)
    moving_norm = normalize_intensity(moving, method=norm_method)

    # ------------------------------------------------------------------
    # 3. Registration
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info(f"STEP 3 — Registration (method={args.method})")
    logger.info("=" * 60)

    reg_cfg_dict = cfg_dict.get("registration", {})
    opt_params = reg_cfg_dict.get("optimizer_params", {})
    metric_params = reg_cfg_dict.get("metric_params", {})

    reg_config = RegistrationConfig(
        metric=reg_cfg_dict.get("metric", "mattes_mi"),
        number_of_histogram_bins=metric_params.get("number_of_histogram_bins", 50),
        sampling_strategy=metric_params.get("sampling_strategy", "RANDOM"),
        sampling_percentage=metric_params.get("sampling_percentage", 0.01),
        learning_rate=opt_params.get("learning_rate", 1.0),
        min_step=opt_params.get("min_step", 0.001),
        number_of_iterations=opt_params.get("number_of_iterations", 200),
        convergence_min_value=opt_params.get("convergence_minimum_value", 1e-6),
        convergence_window_size=opt_params.get("convergence_window_size", 10),
        shrink_factors=reg_cfg_dict.get("shrink_factors", [4, 2, 1]),
        smoothing_sigmas=reg_cfg_dict.get("smoothing_sigmas", [2.0, 1.0, 0.0]),
    )

    result = None
    all_metric_values: list[float] = []

    if args.method in ("rigid", "rigid+affine"):
        rigid_reg = RigidRegistration(reg_config)
        result = rigid_reg.register(fixed_norm, moving_norm, verbose=True)
        all_metric_values.extend(result.metric_values)
        logger.info("Rigid registration complete.")

    if args.method in ("affine", "rigid+affine"):
        init_tx = result.transform if result is not None else None
        affine_reg = AffineRegistration(reg_config)
        result = affine_reg.register(fixed_norm, moving_norm, initial_transform=init_tx)
        all_metric_values.extend(result.metric_values)
        logger.info("Affine registration complete.")

    # ------------------------------------------------------------------
    # 4. Apply transform and save results
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 4 — Applying transform & saving")
    logger.info("=" * 60)

    registered = result.apply(moving)

    out_cfg = cfg_dict.get("output", {})
    if out_cfg.get("save_registered", True):
        save_volume(registered, output_dir / "registered.nii.gz")

    if out_cfg.get("save_transform", True):
        result.save_transform(output_dir / "transform.tfm")

    # ------------------------------------------------------------------
    # 5. Visualization
    # ------------------------------------------------------------------
    if not args.no_viz:
        logger.info("=" * 60)
        logger.info("STEP 5 — Visualization")
        logger.info("=" * 60)

        plot_slices_comparison(
            fixed, moving, registered,
            save_path=figures_dir / "slices_comparison.png",
        )
        plt_check = plot_checkerboard(
            fixed, registered,
            save_path=figures_dir / "checkerboard.png",
        )
        save_overlay(
            fixed, registered,
            save_path=figures_dir / "color_overlay.png",
        )
        if all_metric_values:
            plot_metric_convergence(
                all_metric_values,
                method_name=args.method,
                save_path=figures_dir / "convergence.png",
            )

    # ------------------------------------------------------------------
    # 6. Evaluation
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 6 — Evaluation")
    logger.info("=" * 60)

    metrics: dict = {"method": args.method}

    # TRE
    if args.landmarks_fixed and args.landmarks_moving:
        lm_fixed = load_landmarks_csv(args.landmarks_fixed)
        lm_moving = load_landmarks_csv(args.landmarks_moving)
        tre = compute_tre(lm_fixed, lm_moving, result.transform)
        metrics["TRE"] = tre

    # Dice
    if args.fixed_mask and args.moving_mask:
        fixed_mask = resample_to_spacing(
            load_volume(args.fixed_mask), target_spacing, interpolator=sitk.sitkNearestNeighbor
        )
        moving_mask = resample_to_spacing(
            load_volume(args.moving_mask), target_spacing, interpolator=sitk.sitkNearestNeighbor
        )
        registered_mask = result.apply_to_mask(moving_mask)

        df_dice = compute_dice_before_after(fixed_mask, moving_mask, registered_mask)
        metrics["Dice"] = df_dice.to_dict(orient="records")
        df_dice.to_csv(output_dir / "dice_results.csv", index=False)

    # Save metrics
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to: {metrics_path}")

    logger.info("=" * 60)
    logger.info("Pipeline complete.")
    logger.info(f"Results in: {output_dir.resolve()}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
