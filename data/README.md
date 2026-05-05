# Data — Download Instructions

Medical data is **not committed** to this repository. Place your volumes in the `raw/` folder.

---

## Option A — Medical Segmentation Decathlon (recommended)

Task03_Liver or Task06_Lung provide paired CT volumes with segmentation masks.

```bash
# Via Google Drive (official mirror)
# See: http://medicaldecathlon.com/
# Download Task03_Liver.tar (~28GB) or Task06_Lung.tar

tar -xf Task03_Liver.tar -C data/raw/
```

Expected layout after extraction:
```
data/raw/Task03_Liver/
├── imagesTr/          # Training CT volumes (.nii.gz)
├── labelsTr/          # Liver segmentation masks
└── imagesTs/          # Test CT volumes
```

---

## Option B — KiTS21 (Kidney Tumor Segmentation)

```bash
pip install kits21
python -c "from kits21.api import get_kits21_data; get_kits21_data()"
# Downloads to ~/.kits21/
```

---

## Option C — Small public CT pair (quick start)

For rapid prototyping, a pair of abdominal CT scans is available from the
Learn2Reg challenge (public, ~30MB per volume):

```
https://cloud.imi.uni-luebeck.de/s/xQPEy4zyXArmyRQ
```

Place the two volumes as:
```
data/raw/fixed.nii.gz
data/raw/moving.nii.gz
```

---

## Preprocessing

After downloading, run the preprocessing pipeline:

```bash
python scripts/preprocess.py --input data/raw/ --output data/processed/
```

This resamples all volumes to isotropic 1mm spacing and normalizes intensities.
