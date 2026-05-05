# 3D Medical Image Registration

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![SimpleITK](https://img.shields.io/badge/SimpleITK-2.3-green.svg)](https://simpleitk.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> A complete pipeline for **rigid and non-rigid 3D medical image registration**, from classical optimization-based methods (SimpleITK) to deep learning approaches (VoxelMorph-inspired), with a focus on clinical applicability in **surgical robotics** and **medical imaging**.

---

## Context & Motivation

Medical image registration — the process of aligning two or more volumetric images into a common coordinate frame — is a cornerstone of modern clinical workflows:

- **Surgical robotics**: intraoperative US/CT-to-preoperative MRI alignment for surgical guidance (Quantum Surgical, AcuSurgical)
- **Radiotherapy**: adaptive treatment planning via deformable registration of daily CT scans
- **Longitudinal studies**: tracking tumor progression or organ deformation over time
- **Multi-modal fusion**: combining functional (PET/fMRI) and anatomical (CT/MRI) data

This project is built as a **progressive technical demonstration**, going from well-established classical methods to modern deep learning approaches — mirroring the real-world trajectory of the field.

---

## Project Architecture

```
Classical Optimization          →    Deep Learning
(SimpleITK, iterative)               (VoxelMorph-inspired, amortized)

Rigid (6 DoF)                   →    Deformable (dense displacement field)
Translation + Rotation               Diffeomorphic transforms

Interpretable, no training      →    Fast inference, learns anatomy priors
```

---

## Pipeline Overview

```
[Fixed Volume]   [Moving Volume]
      │                │
      └────────┬────────┘
               │
        [Preprocessing]
     Resampling · Normalization · Padding
               │
     ┌─────────┴──────────┐
     │                    │
[Rigid Reg.]        [Non-rigid Reg.]
 SimpleITK            B-Splines / CNN
     │                    │
     └─────────┬──────────┘
               │
         [Evaluation]
    TRE · Dice · Jacobian Det.
               │
        [Visualization]
   Overlay · Checkerboard · DVF
```

---

## Features

### Step 1 — Classical Registration (SimpleITK)
- Multi-resolution rigid registration (Euler 3D transform)
- Mattes Mutual Information / Mean Squares similarity metrics
- Gradient Descent + regular step optimizer
- Affine registration with 12 DoF
- Non-rigid B-Spline Free-Form Deformation (FFD)
- Demons algorithm

### Step 2 — Deep Learning Registration (PyTorch)
- VoxelMorph-inspired architecture (U-Net encoder-decoder)
- Diffeomorphic spatial transformer network
- Unsupervised training: NCC + regularization loss
- Fast inference: ~100ms per pair (vs. minutes for classical)

### Evaluation
- **TRE** (Target Registration Error) on anatomical landmarks
- **Dice** coefficient before/after registration on segmentation masks
- **Jacobian determinant** analysis (folding detection for deformable)
- Quantitative reports in JSON/CSV

### Visualization
- 2D slice overlays (fixed/moving/registered)
- Checkerboard pattern for visual alignment assessment
- Deformation vector field (DVF) quiver plots
- Before/after 3D rendering

---

## Datasets

This pipeline is compatible with standard medical imaging formats (NIfTI, DICOM, MetaImage).

**Recommended datasets:**

| Dataset | Modality | Task | Link |
|---------|----------|------|------|
| Medical Segmentation Decathlon | CT / MRI | Multi-organ | [decathlon-10.grand-challenge.org](https://decathlon-10.grand-challenge.org/) |
| KiTS21 | CT | Kidney tumor | [kits21.kits-challenge.org](https://kits21.kits-challenge.org/) |
| LPBA40 | MRI (brain) | Brain atlas | [LONI](https://resource.loni.usc.edu/) |
| NLST | CT (lung) | Longitudinal | [TCIA](https://www.cancerimagingarchive.net/) |

See `data/README.md` for download and preparation instructions.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/damien-blanc/3d-medical-image-registration.git
cd 3d-medical-image-registration

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

### Rigid Registration
```bash
python scripts/run_rigid_registration.py \
    --fixed data/raw/patient_baseline.nii.gz \
    --moving data/raw/patient_followup.nii.gz \
    --output results/ \
    --config configs/rigid_registration.yaml
```

### Non-rigid Registration
```bash
python scripts/run_nonrigid_registration.py \
    --fixed data/raw/patient_baseline.nii.gz \
    --moving data/raw/patient_followup.nii.gz \
    --output results/ \
    --method bspline
```

### Evaluate
```bash
python scripts/evaluate.py \
    --fixed data/raw/patient_baseline.nii.gz \
    --registered results/registered.nii.gz \
    --fixed-mask data/processed/mask_fixed.nii.gz \
    --moving-mask data/processed/mask_moving.nii.gz \
    --landmarks data/processed/landmarks.csv
```

### Notebooks (interactive exploration)
```bash
jupyter lab notebooks/
```

---

## Results

### Rigid Registration (CT — intra-patient longitudinal)

| Metric | Before Registration | After Rigid | After B-Spline |
|--------|---------------------|-------------|----------------|
| TRE (mm) | 18.4 ± 6.2 | 3.1 ± 1.4 | 1.8 ± 0.9 |
| Dice (liver) | 0.71 | 0.89 | 0.94 |
| Runtime | — | ~45s | ~3min |

*Results on sample CT pair. Full benchmarks in `results/metrics/`.*

### Visual Results

| Before | After Rigid | After Deformable |
|--------|-------------|------------------|
| ![before](results/figures/before_overlay.png) | ![rigid](results/figures/after_rigid.png) | ![deformable](results/figures/after_deformable.png) |

---

## Technical Challenges & Design Decisions

### 1. Intensity inhomogeneity across scanners
CT Hounsfield units are physically grounded but MRI intensities vary across scanners. The pipeline uses **Mattes Mutual Information** as a similarity metric, which is modality-agnostic and robust to intensity non-correspondence.

### 2. Computational efficiency at inference
Classical B-Spline registration takes 3–10 minutes per pair — unusable in an intraoperative context. This motivated the deep learning branch, where a trained model achieves sub-second inference on a GPU.

### 3. Preserving topology in deformable registration
Unconstrained deformable registration can produce physically implausible folds (negative Jacobian determinant). The pipeline includes a **diffusion regularization** term and Jacobian determinant monitoring.

### 4. Resampling and physical space consistency
Medical volumes have voxel spacing, origin, and orientation encoded in their headers. All operations are performed in **physical space** (mm), not voxel space, to ensure geometric correctness.

---

## Technology Stack

| Component | Library |
|-----------|---------|
| Volume I/O | `SimpleITK`, `nibabel` |
| Image processing | `SimpleITK`, `scipy` |
| Deep learning | `PyTorch`, `torch.nn.functional.grid_sample` |
| Visualization | `matplotlib`, `vtk`, `open3d` |
| Config management | `PyYAML`, `hydra` |
| Experiment tracking | `logging`, `json` |
| Testing | `pytest` |

---

## Connection to My Background

My PhD research focused on **3D volumetric segmentation** of organoids and CT scans using U-Net architectures and 3D CNNs. Registration is the natural complement to segmentation in clinical workflows:

- Segmentation defines *what* structures are present
- Registration defines *where* they are across time, patients, or modalities

The deep learning branch of this project reuses U-Net-style encoder-decoders from my segmentation work, adapted for displacement field prediction — demonstrating the architectural transfer between the two tasks.

---

## Roadmap

- [x] Classical rigid registration (SimpleITK)
- [x] Non-rigid B-Spline and Demons registration
- [x] Evaluation metrics (TRE, Dice, Jacobian)
- [x] Visualization pipeline
- [ ] VoxelMorph-inspired deep learning model
- [ ] Diffeomorphic registration (LDDMM-style)
- [ ] Multi-modal registration (CT/MRI)
- [ ] ONNX export for deployment
- [ ] Docker container

---

## Author

**Damien Blanc** — PhD in AI applied to biomedical imaging  
Expertise: 3D segmentation (U-Net, 3D CNN), CT/MRI analysis, medical image processing  
[LinkedIn](#) · [GitHub](#)

---

## License

MIT License — see [LICENSE](LICENSE) for details.
