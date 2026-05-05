"""
Generate synthetic 3D CT-like volume pair for pipeline testing
==============================================================
Creates a fixed/moving pair with:
  - Realistic CT-like anatomy (multiple ellipsoidal structures with HU values)
  - Known rigid transform (rotation + translation) applied to produce moving
  - Segmentation masks for Dice evaluation
  - Landmark file for TRE evaluation

Ground truth transform is saved so you can verify registration accuracy.

Usage:
    python scripts/generate_synthetic_data.py --output data/raw/
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def make_phantom(shape=(128, 128, 128), spacing=(1.5, 1.5, 1.5)):
    """
    Build a CT-like phantom with:
      - Background (-1000 HU)
      - Body outline (soft tissue, ~50 HU)
      - Liver-like organ (~60 HU)
      - Vessel-like structure (~200 HU)
      - Bone-like structure (~700 HU)
    Returns:
        arr: float32 numpy array (z, y, x)
        mask: uint8 label array (0=bg, 1=body, 2=liver, 3=bone)
        landmarks: (N, 3) array of physical landmarks in mm (x, y, z)
    """
    arr = np.full(shape, fill_value=-1000.0, dtype=np.float32)
    mask = np.zeros(shape, dtype=np.uint8)

    cz, cy, cx = [s // 2 for s in shape]

    def ellipsoid(center, radii):
        zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
        return (
            ((zz - center[0]) / radii[0]) ** 2
            + ((yy - center[1]) / radii[1]) ** 2
            + ((xx - center[2]) / radii[2]) ** 2
        ) <= 1.0

    # 1. Body oval (label=1, ~50 HU soft tissue)
    body = ellipsoid((cz, cy, cx), (cz * 0.85, cy * 0.80, cx * 0.82))
    arr[body] = 50.0
    mask[body] = 1

    # 2. Liver-like blob (label=2, ~60 HU)
    liver = ellipsoid((cz + 5, cy - 8, cx + 12), (18, 22, 28))
    arr[liver] = 62.0
    mask[liver] = 2

    # 3. Bone-like structures (label=3, ~700 HU)
    spine = ellipsoid((cz, cy + 25, cx), (40, 6, 5))
    arr[spine] = 700.0
    mask[spine] = 3
    rib_l = ellipsoid((cz + 15, cy + 10, cx - 28), (5, 4, 12))
    arr[rib_l] = 680.0
    mask[rib_l] = 3
    rib_r = ellipsoid((cz + 15, cy + 10, cx + 28), (5, 4, 12))
    arr[rib_r] = 680.0
    mask[rib_r] = 3

    # 4. Vessel (high density ~200 HU)
    vessel = ellipsoid((cz - 5, cy - 20, cx + 5), (30, 3, 3))
    arr[vessel] = 200.0
    mask[vessel] = 1  # keep as soft tissue label

    # Add realistic CT noise (σ ~ 20 HU)
    rng = np.random.default_rng(42)
    arr += rng.normal(0, 20, shape).astype(np.float32)

    # Landmarks: 8 anatomically motivated points (voxel → physical mm)
    voxel_landmarks = np.array([
        [cx, cy, cz],               # center
        [cx + 12, cy - 8, cz + 5],  # liver center
        [cx, cy + 25, cz],          # spine
        [cx - 28, cy + 10, cz + 15], # rib L
        [cx + 28, cy + 10, cz + 15], # rib R
        [cx + 5, cy - 20, cz - 5],  # vessel
        [cx - 20, cy - 15, cz + 10],
        [cx + 15, cy + 5, cz - 20],
    ], dtype=np.float64)

    # Convert voxel → physical (LPS mm): x * spacing_x, y * spacing_y, z * spacing_z
    phys_landmarks = voxel_landmarks * np.array(spacing)[[0, 1, 2]]

    return arr, mask, phys_landmarks


def array_to_sitk(arr, spacing, origin=(0.0, 0.0, 0.0)):
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing)
    img.SetOrigin(origin)
    return img


def apply_rigid_transform(
    image: sitk.Image,
    rotation_deg: tuple,
    translation_mm: tuple,
) -> tuple[sitk.Image, sitk.Euler3DTransform]:
    """Apply a known Euler3D transform to an image.

    Args:
        image: Input volume.
        rotation_deg: (rx, ry, rz) rotation in degrees.
        translation_mm: (tx, ty, tz) translation in mm.

    Returns:
        (transformed_image, transform)
    """
    # Build transform in physical space
    tx = sitk.Euler3DTransform()

    # Set rotation center to image center in physical space
    size = image.GetSize()
    spacing = image.GetSpacing()
    origin = image.GetOrigin()
    center = [
        origin[i] + spacing[i] * size[i] / 2.0
        for i in range(3)
    ]
    tx.SetCenter(center)
    tx.SetRotation(*[np.deg2rad(r) for r in rotation_deg])
    tx.SetTranslation(translation_mm)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(image)
    resampler.SetTransform(tx)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(-1000.0)

    transformed = resampler.Execute(image)
    return transformed, tx


def apply_rigid_to_mask(
    mask: sitk.Image,
    transform: sitk.Euler3DTransform,
    reference: sitk.Image,
) -> sitk.Image:
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetTransform(transform)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(mask)


def apply_rigid_to_landmarks(
    landmarks: np.ndarray,
    transform: sitk.Euler3DTransform,
) -> np.ndarray:
    """Transform physical landmarks using the inverse of the image transform."""
    # The transform maps moving→fixed; landmarks in fixed→moving uses inverse
    inv = transform.GetInverse()
    transformed = np.array([
        inv.TransformPoint(pt.tolist())
        for pt in landmarks
    ])
    return transformed


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic CT-like registration test data")
    parser.add_argument("--output", default="data/raw/", help="Output directory")
    parser.add_argument("--shape", nargs=3, type=int, default=[128, 128, 128])
    parser.add_argument("--spacing", nargs=3, type=float, default=[1.5, 1.5, 1.5])
    # Ground truth misalignment
    parser.add_argument("--rotation-deg", nargs=3, type=float, default=[5.0, 3.0, -2.0],
                        help="Rotation (rx, ry, rz) in degrees applied to produce moving")
    parser.add_argument("--translation-mm", nargs=3, type=float, default=[8.0, -5.0, 3.0],
                        help="Translation (tx, ty, tz) in mm applied to produce moving")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    shape = tuple(args.shape)
    spacing = tuple(args.spacing)

    print(f"Generating phantom: shape={shape}, spacing={spacing}mm")
    arr, mask_arr, landmarks_fixed = make_phantom(shape, spacing)

    # --- Fixed volume ---
    fixed_img = array_to_sitk(arr, spacing)
    fixed_mask = array_to_sitk(mask_arr.astype(np.float32), spacing)
    fixed_mask = sitk.Cast(fixed_mask, sitk.sitkUInt8)

    # --- Moving volume (fixed + known transform) ---
    rotation_deg = tuple(args.rotation_deg)
    translation_mm = tuple(args.translation_mm)

    print(f"Applying transform: rotation={rotation_deg}°, translation={translation_mm}mm")
    moving_img, gt_transform = apply_rigid_transform(fixed_img, rotation_deg, translation_mm)
    moving_mask = apply_rigid_to_mask(fixed_mask, gt_transform, fixed_img)
    landmarks_moving = apply_rigid_to_landmarks(landmarks_fixed, gt_transform)

    # --- Save volumes ---
    sitk.WriteImage(fixed_img,  str(output / "fixed.nii.gz"))
    sitk.WriteImage(moving_img, str(output / "moving.nii.gz"))
    sitk.WriteImage(fixed_mask, str(output / "fixed_mask.nii.gz"))
    sitk.WriteImage(moving_mask, str(output / "moving_mask.nii.gz"))

    # --- Save landmarks as CSV ---
    import csv
    for name, lm in [("landmarks_fixed.csv", landmarks_fixed),
                     ("landmarks_moving.csv", landmarks_moving)]:
        with open(output / name, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y", "z"])
            writer.writerows(lm.tolist())

    # --- Save ground truth transform ---
    sitk.WriteTransform(gt_transform, str(output / "ground_truth_transform.tfm"))

    # --- Save metadata JSON ---
    metadata = {
        "shape_zyx": list(shape),
        "spacing_xyz_mm": list(spacing),
        "ground_truth": {
            "rotation_deg": list(rotation_deg),
            "translation_mm": list(translation_mm),
        },
        "files": {
            "fixed": "fixed.nii.gz",
            "moving": "moving.nii.gz",
            "fixed_mask": "fixed_mask.nii.gz",
            "moving_mask": "moving_mask.nii.gz",
            "landmarks_fixed": "landmarks_fixed.csv",
            "landmarks_moving": "landmarks_moving.csv",
            "ground_truth_transform": "ground_truth_transform.tfm",
        },
        "labels": {0: "background", 1: "soft tissue", 2: "liver", 3: "bone"},
        "expected_tre_before_mm": float(np.linalg.norm(translation_mm)),
    }
    with open(output / "synthetic_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nGenerated files:")
    for fname in sorted(output.glob("*")):
        size_kb = fname.stat().st_size / 1024
        print(f"  {fname.name:40s} {size_kb:8.1f} KB")

    print(f"\nGround truth misalignment:")
    print(f"  Rotation:    {rotation_deg} degrees")
    print(f"  Translation: {translation_mm} mm")
    print(f"  Expected TRE (translation only): {np.linalg.norm(translation_mm):.2f} mm")
    print("\nDone. Run the registration pipeline with:")
    print(f"  python scripts/run_rigid_registration.py \\")
    print(f"    --fixed {output}/fixed.nii.gz \\")
    print(f"    --moving {output}/moving.nii.gz \\")
    print(f"    --fixed-mask {output}/fixed_mask.nii.gz \\")
    print(f"    --moving-mask {output}/moving_mask.nii.gz \\")
    print(f"    --landmarks-fixed {output}/landmarks_fixed.csv \\")
    print(f"    --landmarks-moving {output}/landmarks_moving.csv \\")
    print(f"    --output results/synthetic/")


if __name__ == "__main__":
    main()
