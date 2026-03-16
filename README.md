# 🔬 Zebrafish Cell Segmentation with SAM 1

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/SAM-Meta%20AI-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-Apache%202.0-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Hardware-CPU%20%7C%20GPU-lightgrey?style=for-the-badge"/>
</p>

A modular Python pipeline for the **automatic segmentation and counting of tumor cells** in Zebrafish Electron Microscopy (EM) images, powered by [Meta AI's Segment Anything Model (SAM 1)](https://github.com/facebookresearch/segment-anything).

---

## 📌 Project Description

Automating cell counting is a fundamental step in **Radiomics** workflows. This pipeline is designed to run on standard hardware (successfully tested with only **4 GB of RAM**) and features a **Global Switches** architecture, allowing users to easily toggle preprocessing and filtering stages on or off without modifying core logic.

---

## 🚀 Key Features

| Feature | Description |
|---|---|
| 🎨 **Adaptive Preprocessing** | CLAHE + Median Blur to sharpen cell membranes and enhance local contrast |
| 🤖 **Optimized SAM Parameters** | Grid density, IOU threshold, and stability tuned for microscopy data |
| 📐 **Area Filtering** | Removes detections that are too small (noise) or too large (background) |
| 💡 **Statistical Intensity Filtering** | Discards statistically dark/out-of-focus cells using mean ± N·σ cutoff |
| 📄 **Automatic Reporting** | Generates a `.txt` technical log with Raw and Filtered counts per image slice |
| 🔧 **Global Switch Architecture** | Enable/disable each pipeline stage with a single `True`/`False` flag |

---

## 🧠 Pipeline Architecture

```
Input Image (.tif / .png)
        │
        ▼
┌───────────────────┐
│  PREPROCESSING    │  CLAHE + Median Blur + Percentile Clipping
│  (toggleable)     │  Converts to 3-channel RGB for SAM
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│   SAM INFERENCE   │  Automatic mask generation
│   (vit_b / vit_h) │  Grid scan + IOU threshold + Stability filter
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  POST-FILTERING   │  Area constraints (MIN_AREA / MAX_AREA_FRAC)
│  (toggleable)     │  Intensity statistics (mean - N·sigma)
└────────┬──────────┘
         │
         ▼
┌───────────────────────────────────────────┐
│            OUTPUT FILES                   │
│  _labels.tif  |  _overlay.png  |  report  │
└───────────────────────────────────────────┘
```

---

## 💻 Hardware Requirements

- **RAM:** Minimum 4 GB (tested with `vit_b` model)
- **CPU:** Fully supported (slower but functional)
- **GPU:** NVIDIA CUDA recommended for large datasets (set `device="cuda"` in the script)

---

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/zebrafish-cell-segmentation.git
cd zebrafish-cell-segmentation
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> Make sure you have an up-to-date version of `pip` before running this command.

### 3. Download the SAM Model Checkpoint

Download the **ViT-B (Base)** model weights from the official SAM repository:

| Model | Size | Download |
|-------|------|----------|
| `sam_vit_b_01ec64.pth` | 375 MB | [Download](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) |

Save the file to a local folder and update `CHECKPOINT_PATH` inside the script.

---

## 📖 Usage

### 1. Configure the script

Open `SAM1_segmentation_pipeline.py` and update **Section 2**:

```python
# SECTION 2: INPUT/OUTPUT PATHS
INPUT_DIR       = r"C:\path\to\your\images"
CHECKPOINT_PATH = r"C:\path\to\sam_vit_b_01ec64.pth"
```

### 2. Tune the Global Switches *(optional)*

```python
ENABLE_PREPROCESSING      = True   # CLAHE + Median Blur
ENABLE_SAM_REGION_CLEANING = True  # Remove internal holes/islands
ENABLE_POST_FILTERING     = True   # Area-based filter
ENABLE_INTENSITY_FILTERING = True  # Statistical brightness filter
```

### 3. Run the pipeline

```bash
python SAM1_segmentation_pipeline.py
```

---

## ⚙️ Key Parameters

### SAM Generator (Section 3)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `POINTS_PER_SIDE` | `48` | Grid density for image scanning (48² = 2304 points) |
| `PRED_IOU_THRESH` | `0.78` | Confidence threshold (0–1). Higher = stricter quality |
| `STABILITY_THRESH` | `0.88` | Robustness filter against lighting variations |
| `MIN_MASK_REGION_AREA` | `10` | Internal SAM cleanup: fragments smaller than this are removed |

### Biological Filters (Section 4)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MIN_AREA` | `10 px` | Minimum cell area (removes noise artifacts) |
| `MAX_AREA_FRAC` | `0.00072` | Maximum cell area as fraction of image size |
| `INTENSITY_SIGMA_THRESH` | `1.5` | Cutoff: discard cells below `mean - N × std_dev` |

---

## 📊 Output Files

For each input image, the pipeline generates **three output files** in the same folder:

| File | Format | Description |
|------|--------|-------------|
| `<name>_labels.tif` | 16-bit TIFF | Label map — each cell has a unique integer ID |
| `<name>_overlay.png` | PNG | Colorized segmentation overlaid on the original image |
| `<name>_preprocessed.png` | PNG | Visualization of the image after CLAHE enhancement |

A single `segmentation_technical_report.txt` is also generated, logging **Raw SAM Count** and **Filtered Count** for every processed image.

---

## 🗂️ Project Structure

```
zebrafish-cell-segmentation/
│
├── SAM1_segmentation_pipeline.py   # Main pipeline script
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── (your image folder)
    ├── image001.tif
    ├── image001_labels.tif         # ← Generated output
    ├── image001_overlay.png        # ← Generated output
    ├── image001_preprocessed.png   # ← Generated output
    └── segmentation_technical_report.txt
```

---

## 📄 License

This project is distributed under the **Apache License 2.0**.

It incorporates the **Segment Anything Model (SAM)** by Meta AI — please refer to the [official SAM repository](https://github.com/facebookresearch/segment-anything) for original model licensing details.

---

## 🙏 Acknowledgements

- [**Meta AI – Segment Anything**](https://github.com/facebookresearch/segment-anything) for the SAM 1 model
- [**OpenCV**](https://opencv.org/) for image preprocessing utilities
- University of Bologna — Pattern Recognition course (Prof. Piccinini)
