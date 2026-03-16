import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# ==============================================================================
# SECTION 1: GLOBAL SWITCHES (PIPELINE CONTROL)
# ==============================================================================
# Use these Boolean (True/False) switches to define the workflow.

# 1. ENABLE_PREPROCESSING: 
# [True]: Applies filters (CLAHE, Median Blur) to improve image quality before SAM.
# [False]: SAM works on the original raw image. (Use False only for high-contrast images).
ENABLE_PREPROCESSING = True

# 2. ENABLE_SAM_REGION_CLEANING: 
# [True]: SAM internally fills small holes and removes "islands" inside masks.
# [False]: Masks remain "raw". (Always better set to True for biological tissues).
ENABLE_SAM_REGION_CLEANING = True 

# 3. ENABLE_POST_FILTERING: 
# [True]: Applies Section 4 area-filters to the final masks.
# [False]: Keeps all masks found by SAM, including tiny noise or background fragments.
ENABLE_POST_FILTERING = True

# 4. ENABLE_INTENSITY_FILTERING: (NEW)
# [True]: Calculates mean intensity of every detected cell. Discards cells that are 
#         statistically too dark (outliers) compared to the population average.
# [False]: Skips this check.
ENABLE_INTENSITY_FILTERING = True

# ==============================================================================
# SECTION 2: INPUT/OUTPUT PATHS
# ==============================================================================
INPUT_DIR = r"C:\Users\Carlo\Desktop\UNIBO\PATTERN RECOGNITION\PATTERN_Piccinini\Dataset1Zebrafish\OriginalImages\Red_channel\Image4_20X_red_slices TANTE PROVE FINALI\CONFRONTO_finale\prova ultima"
CHECKPOINT_PATH = r"C:\sam_models\sam_vit_b_01ec64.pth"
SAM_MODEL_TYPE = "vit_b" # "vit_b" is balanced, "vit_h" is high-accuracy but slow.
EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

# ==============================================================================
# SECTION 3: SAM GENERATOR PARAMETERS (AI TUNING)
# ==============================================================================
# These values control how the AI "scans" the image.

# 1. POINTS_PER_SIDE: Controls the density of the search grid.
# - 32: Standard (1024 points).
# - 64: High density. Better for tiny cells but significantly slower.
POINTS_PER_SIDE = 48

# 2. PRED_IOU_THRESH (0.0 to 1.0): Confidence filter.
# - Higher (0.9): Keeps only masks where the AI is 90% sure of the quality.
# - Lower (0.7): Allows more masks, but increases the risk of "junk" detections.
PRED_IOU_THRESH = 0.78

# 3. STABILITY_THRESH (0.0 to 1.0): Robustness filter.
# - Higher (0.95): Keeps only masks that don't change when lighting varies slightly.
# - Use this to eliminate "ghost" masks that aren't real objects.
STABILITY_THRESH = 0.88

# 4. MIN_MASK_REGION_AREA: Internal cleanup size.
# - If ENABLE_SAM_REGION_CLEANING is True, fragments smaller than this are erased.
MIN_MASK_REGION_AREA = 10 

# ==============================================================================
# SECTION 4: CUSTOM BIOLOGICAL FILTERS
# ==============================================================================
# These filters enforce size limits to match cell dimensions in your images.

# MIN_AREA: Discard any detection smaller than X pixels.
# (Prevents counting noise artifacts as cells).
#a good value is 10 pixels for allowing detection of very small fragments belonging
#  to peripherical part of a cell
MIN_AREA = 10  

# MAX_AREA_FRAC: Discard any detection larger than X% of the image.
# (Prevents counting the whole background as a single giant cell).
MAX_AREA_FRAC = 0.00072

# INTENSITY_SIGMA_THRESH: (NEW)
# Defines the cutoff for the statistical filter. 
# We calculate Mean (mu) and Std Dev (sigma) of all cell intensities.
# We discard any cell where: cell_intensity < (Mean - N * Sigma).
# 3.0 is a standard statistical choice (covers 99.7% of a normal distribution).
INTENSITY_SIGMA_THRESH = 1.5

# ==============================================================================
# CORE LOGIC: PREPROCESSING
# ==============================================================================
def preprocess_image(gray_u8: np.ndarray) -> np.ndarray:
    """
    Why this is essential for EM:
    1. Percentile Clipping: Removes camera artifacts and extreme noise.
    2. Normalization: Stretches contrast so boundaries are visible.
    3. Median Blur: Smooths textures while keeping membrane edges sharp.
    4. CLAHE: Enhances local contrast in dark areas of the Zebrafish tissue.
    """
    g = gray_u8.astype(np.float32)
    lo, hi = np.percentile(g, (1, 99))
    g = np.clip(g, lo, hi)
    g = (255 * (g - lo) / (hi - lo + 1e-8)).astype(np.uint8)
    g = cv2.medianBlur(g, 3)
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    g = clahe.apply(g)
    # SAM requires RGB (3 channels) to process the image correctly.
    return np.stack([g, g, g], axis=-1)

# ==============================================================================
# CORE LOGIC: FILTERING & LABELING
# ==============================================================================
def filter_masks_custom(masks, H, W):
    """Filters the list of masks using the area constraints from Section 4."""
    total_area = H * W
    return [m for m in masks if MIN_AREA <= m.get("area", 0) <= (MAX_AREA_FRAC * total_area)]

def filter_masks_by_intensity_stats(masks, image_gray):
    """
    (NEW FUNCTION)
    Filters masks based on statistical intensity analysis.
    1. Computes the mean intensity of pixels under each mask.
    2. Calculates global statistics (Mean and StdDev) of these intensities.
    3. Discards masks that are too dark (outliers below Mean - 3*StdDev).
    """
    if not masks:
        return []

    # Calculate mean intensity for every single mask
    mask_intensities = []
    for m in masks:
        # Extract the pixels belonging to this mask from the original image
        # m["segmentation"] is a boolean matrix (True where the cell is)
        cell_pixels = image_gray[m["segmentation"]]
        
        # Calculate mean brightness of this specific cell
        if cell_pixels.size > 0:
            mean_val = np.mean(cell_pixels)
        else:
            mean_val = 0
        mask_intensities.append(mean_val)

    # Convert to numpy array for statistics
    mask_intensities = np.array(mask_intensities)

    # Calculate global population stats
    global_mean = np.mean(mask_intensities)
    global_std = np.std(mask_intensities)
    
    # Define the cutoff threshold
    # Everything below this value is considered "noise" or "out of focus"
    cutoff_value = global_mean - (INTENSITY_SIGMA_THRESH * global_std)

    # Filter the list
    filtered_masks = []
    for i, m in enumerate(masks):
        if mask_intensities[i] >= cutoff_value:
            filtered_masks.append(m)
    
    return filtered_masks

def masks_to_label_image(masks, H, W):
    """Transforms a list of masks into a single 16-bit label map (one ID per cell)."""
    label_img = np.zeros((H, W), dtype=np.uint16)
    # Sort by area so smaller cells are drawn last (on top)
    masks_sorted = sorted(masks, key=lambda x: x.get("area", 0), reverse=True)
    for i, m in enumerate(masks_sorted, start=1):
        label_img[m["segmentation"].astype(bool)] = i
    return label_img

def make_overlay(base_gray: np.ndarray, label_img: np.ndarray, alpha=0.45):
    """Blends the original image with random colors for each detected instance."""
    base_bgr = cv2.cvtColor(base_gray, cv2.COLOR_GRAY2BGR)
    n = int(label_img.max())
    if n == 0: return base_bgr
    rng = np.random.default_rng(12345)
    colors = rng.integers(0, 255, size=(n + 1, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]
    color_mask = colors[label_img]
    tinted = np.where((label_img > 0)[..., None], color_mask, base_bgr).astype(np.uint8)
    return cv2.addWeighted(base_bgr, 1 - alpha, tinted, alpha, 0)

# ==============================================================================
# MAIN BATCH PROCESSING
# ==============================================================================
def main():
    input_path = Path(INPUT_DIR)
    report_path = input_path / "segmentation_technical_report.txt"

    # Step 1: Writing the Technical Header to the Report
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("SAM1 SEGMENTATION - TECHNICAL EXPERIMENT LOG\n")
        f.write("============================================\n\n")
        f.write("[SWITCHES CONFIGURATION]\n")
        f.write(f"- Preprocessing Enabled:     {ENABLE_PREPROCESSING}\n")
        f.write(f"- Internal Region Cleaning:  {ENABLE_SAM_REGION_CLEANING}\n")
        f.write(f"- Custom Post-Filtering:     {ENABLE_POST_FILTERING}\n")
        f.write(f"- Intensity Stats Filter:    {ENABLE_INTENSITY_FILTERING}\n\n") # NEW
        f.write("[HYPERPARAMETERS]\n")
        f.write(f"- Points per side:           {POINTS_PER_SIDE} (Total: {POINTS_PER_SIDE**2})\n")
        f.write(f"- Predicted IOU Thresh:      {PRED_IOU_THRESH}\n")
        f.write(f"- Stability Score Thresh:    {STABILITY_THRESH}\n")
        f.write(f"- Internal Cleanup Area:     {MIN_MASK_REGION_AREA} pixels\n\n")
        f.write("[BIOLOGICAL FILTERS]\n")
        f.write(f"- Minimum Cell Area:         {MIN_AREA} pixels\n")
        f.write(f"- Maximum Area Fraction:     {MAX_AREA_FRAC} (ratio)\n")
        f.write(f"- Intensity Sigma Thresh:    {INTENSITY_SIGMA_THRESH} std_devs\n\n") # NEW
        f.write("-" * 75 + "\n")
        f.write(f"{'Image Name':<45} | {'Raw SAM Count':<12} | {'Filtered Count':<12}\n")
        f.write("-" * 75 + "\n")

    # Step 2: Initialize SAM
    print(f"Loading {SAM_MODEL_TYPE} model...")
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    sam.to(device="cpu") # Change to "cuda" if you have an NVIDIA GPU

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=POINTS_PER_SIDE,
        pred_iou_thresh=PRED_IOU_THRESH,
        stability_score_thresh=STABILITY_THRESH,
        min_mask_region_area=MIN_MASK_REGION_AREA if ENABLE_SAM_REGION_CLEANING else 0,
    )

    img_files = [p for p in input_path.iterdir() if p.suffix.lower() in EXTS]
    print(f"Starting processing of {len(img_files)} images...")

    plt.ion() # Interaction mode for real-time plot updates

    for i, img_p in enumerate(img_files, start=1):
        print(f"[{i}/{len(img_files)}] Processing: {img_p.name}")
        gray = cv2.imread(str(img_p), cv2.IMREAD_GRAYSCALE)
        if gray is None: continue
        
        # 1. Image Enhancement
        proc_rgb = preprocess_image(gray) if ENABLE_PREPROCESSING else np.stack([gray]*3, -1)

        # 2. AI Inference
        t0 = time.perf_counter()
        masks = mask_generator.generate(proc_rgb)
        raw_count = len(masks)
        t1 = time.perf_counter()

        # 3. Filtering and Labeling
        if ENABLE_POST_FILTERING:
            # First, filter by Area (existing)
            masks = filter_masks_custom(masks, *gray.shape)
            
            # Second, filter by Intensity Statistics (NEW)
            if ENABLE_INTENSITY_FILTERING:
                # We pass 'gray' (original raw image) to check real intensities,
                # avoiding artifacts from the contrast enhancement/preprocessing.
                masks = filter_masks_by_intensity_stats(masks, gray)
                
        final_count = len(masks)

        label_img = masks_to_label_image(masks, *gray.shape)
        overlay = make_overlay(gray, label_img)

        # 4. Save Outputs
        cv2.imwrite(str(img_p.parent / f"{img_p.stem}_labels.tif"), label_img)
        cv2.imwrite(str(img_p.parent / f"{img_p.stem}_overlay.png"), overlay)
        cv2.imwrite(str(img_p.parent / f"{img_p.stem}_preprocessed.png"), proc_rgb[:, :, 0])

        # 5. Log Results
        with open(report_path, "a", encoding="utf-8") as f:
            f.write(f"{img_p.name:<45} | {raw_count:<12} | {final_count:<12}\n")

        # Live visualization
        plt.figure(1, figsize=(8, 8))
        plt.clf()
        plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        plt.title(f"{img_p.name} - Found: {final_count} cells")
        plt.axis("off")
        plt.pause(0.1)

    print(f"\nTask Complete. Technical Report: {report_path}")
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()