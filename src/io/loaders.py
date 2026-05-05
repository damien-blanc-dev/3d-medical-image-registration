"""
Volume I/O — NIfTI, DICOM, MetaImage
=====================================
All loading functions return SimpleITK.Image objects, which preserve the
full spatial metadata (origin, spacing, direction cosines) required for
physically correct registration.

Why SimpleITK as the primary I/O layer?
  SimpleITK wraps ITK and handles the physical-space geometry natively.
  This ensures that downstream resampling and transform application
  operate in millimetres, not voxel indices.
"""

import logging
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Primary loader — SimpleITK (recommended for registration pipelines)
# ---------------------------------------------------------------------------

def load_volume(path: str | Path) -> sitk.Image:
    """Load a 3D medical volume using SimpleITK.

    Supports NIfTI (.nii, .nii.gz), MetaImage (.mha, .mhd),
    NRRD (.nrrd), and DICOM directories.

    Args:
        path: Path to the image file or DICOM directory.

    Returns:
        SimpleITK.Image with spatial metadata (spacing, origin, direction).

    Raises:
        FileNotFoundError: If the path does not exist.
        RuntimeError: If SimpleITK cannot read the file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Volume not found: {path}")

    if path.is_dir():
        # Assume DICOM series
        logger.info(f"Loading DICOM series from directory: {path}")
        image = _load_dicom_series(path)
    else:
        logger.info(f"Loading volume: {path}")
        image = sitk.ReadImage(str(path))

    logger.info(
        f"  Size:    {image.GetSize()}  (x, y, z)"
        f"\n  Spacing: {image.GetSpacing()} mm"
        f"\n  Origin:  {image.GetOrigin()} mm"
        f"\n  Pixel type: {image.GetPixelIDTypeAsString()}"
    )
    return image


def _load_dicom_series(directory: Path) -> sitk.Image:
    """Load a DICOM series from a directory, handling multi-series folders."""
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(directory))

    if not series_ids:
        raise RuntimeError(f"No DICOM series found in {directory}")

    if len(series_ids) > 1:
        logger.warning(
            f"Multiple DICOM series found ({len(series_ids)}). "
            f"Loading the first series: {series_ids[0]}"
        )

    dicom_names = reader.GetGDCMSeriesFileNames(str(directory), series_ids[0])
    reader.SetFileNames(dicom_names)
    return reader.Execute()


# ---------------------------------------------------------------------------
# NiBabel loader — useful for neuroimaging (NIfTI + affine matrix)
# ---------------------------------------------------------------------------

def load_volume_nibabel(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a NIfTI volume using NiBabel, returning array + affine.

    NiBabel is the standard library for neuroimaging workflows (FSL, SPM
    compatibility). Returns the raw numpy array and the 4x4 affine matrix
    mapping voxel indices to RAS+ physical coordinates.

    Args:
        path: Path to a .nii or .nii.gz file.

    Returns:
        Tuple of (data array [H, W, D], affine [4, 4]).
    """
    path = Path(path)
    logger.info(f"Loading volume with NiBabel: {path}")

    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    affine = img.affine

    logger.info(
        f"  Shape:  {data.shape}"
        f"\n  Voxel size: {nib.affines.voxel_sizes(affine).round(3)} mm"
        f"\n  Data range: [{data.min():.1f}, {data.max():.1f}]"
    )
    return data, affine


def nibabel_to_sitk(data: np.ndarray, affine: np.ndarray) -> sitk.Image:
    """Convert a NiBabel volume (array + affine) to a SimpleITK image.

    NiBabel uses RAS+ convention (x→R, y→A, z→S), while SimpleITK/ITK
    use LPS (x→L, y→P, z→S). This function applies the flip needed.

    Args:
        data: Numpy array [H, W, D] in RAS+ order.
        affine: 4x4 affine matrix (voxel→RAS mm).

    Returns:
        SimpleITK.Image with correct spatial metadata.
    """
    # Flip RAS → LPS on first two axes
    data_lps = data[::-1, ::-1, :]

    image = sitk.GetImageFromArray(np.transpose(data_lps, (2, 1, 0)))

    # Extract spacing, origin, direction from affine
    spacing = nib.affines.voxel_sizes(affine).tolist()
    origin_ras = affine[:3, 3]
    origin_lps = [-origin_ras[0], -origin_ras[1], origin_ras[2]]

    # Direction cosines (3×3 rotation from affine, normalized)
    rot = affine[:3, :3] / np.array(spacing)
    rot_lps = rot.copy()
    rot_lps[:2, :] *= -1  # flip RAS → LPS

    image.SetSpacing(spacing)
    image.SetOrigin(origin_lps)
    image.SetDirection(rot_lps.flatten().tolist())

    return image


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_volume(image: sitk.Image, path: str | Path) -> None:
    """Save a SimpleITK image to disk.

    The output format is determined by the file extension
    (.nii.gz, .mha, .nrrd, etc.).

    Args:
        image: SimpleITK.Image to save.
        path: Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving volume to: {path}")
    sitk.WriteImage(image, str(path))


def sitk_to_numpy(image: sitk.Image) -> np.ndarray:
    """Extract the numpy array from a SimpleITK image.

    SimpleITK stores data as (x, y, z); numpy convention is (z, y, x).
    GetArrayFromImage handles this transposition automatically.

    Args:
        image: SimpleITK.Image.

    Returns:
        Numpy array with shape (D, H, W) — z-first convention.
    """
    return sitk.GetArrayFromImage(image).astype(np.float32)


def numpy_to_sitk(
    array: np.ndarray,
    reference: sitk.Image,
) -> sitk.Image:
    """Wrap a numpy array into a SimpleITK image, copying spatial metadata.

    Args:
        array: Numpy array (D, H, W) — z-first.
        reference: SimpleITK image whose spacing/origin/direction to copy.

    Returns:
        SimpleITK.Image with the array's data and reference's geometry.
    """
    image = sitk.GetImageFromArray(array)
    image.CopyInformation(reference)
    return image
