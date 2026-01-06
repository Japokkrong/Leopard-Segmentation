# Leopard Segmentation using Local Binary Patterns (LBP)

This project implements **texture-based image segmentation** using **Local Binary Patterns (LBP)** to identify and extract a leopard from its background.  
Instead of deep learning models, it uses **classical computer vision techniques** to analyze texture patterns and separate the leopard's spotted fur from surrounding areas.

This approach is lightweight, interpretable, and ideal for learning **texture analysis and segmentation pipelines**.

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![scikit--image](https://img.shields.io/badge/scikit--image-3E92CC?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)

---

## Project Overview

The system processes a leopard image and performs the following steps:

1. Extract **training patches** from leopard and non-leopard regions
2. Compute **LBP histograms** for texture representation
3. Use **sliding window** to scan the entire image
4. Calculate **KL divergence** to measure texture similarity
5. Apply **morphological operations** to refine segmentation
6. Extract and visualize the segmented leopard

This method works best with **textured patterns** and is intended for **educational and research purposes**.

---

## Installation

Install required dependencies:

```bash
conda install scikit-image -c conda-forge -y
# or
pip install scikit-image opencv-python numpy matplotlib
```

---

## File Structure

```
.
├── Leopard_segmentation.ipynb    # Jupyter Notebook with full implementation
├── leopard.jpg                   # Input leopard image
└── README.md
```

---

## How It Works (Pipeline)

### 1. Image Loading & Preprocessing
The leopard image is loaded in grayscale and resized for faster computation:

```python
leopardImage = cv2.imread('leopard.jpg', cv2.IMREAD_GRAYSCALE)
leopardImage = cv2.resize(leopardImage, None, fx=0.5, fy=0.5)
```

### 2. Training Patch Extraction
Sample patches are manually selected from:
- **Leopard regions**: Areas with spotted fur texture (3 patches)
- **Non-leopard regions**: Background areas (3 patches)

```python
leopardPatch1 = leopardImage[100:150, 400:450]
leopardPatch2 = leopardImage[100:150, 150:200]
leopardPatch3 = leopardImage[200:250, 300:350]

nonleopardPatch1 = leopardImage[0:50, 400:450]
nonleopardPatch2 = leopardImage[250:300, 0:50]
nonleopardPatch3 = leopardImage[250:300, 100:150]
```

These patches serve as training examples to learn texture patterns.

### 3. Local Binary Pattern (LBP) Feature Extraction
LBP encodes local texture information by comparing each pixel with its neighbors:

```python
P = 8  # Number of neighbors
R = 1  # Radius

def get_lbp_hist(patch):
    lbp = local_binary_pattern(patch, P, R, method='uniform')
    nBin = int(lbp.max()) + 1
    hist, _ = np.histogram(lbp, bins=nBin, range=(0, nBin), density=True)
    return hist
```

The histogram represents the texture signature of each patch.

### 4. Similarity Measurement (KL Divergence)
KL divergence quantifies the difference between texture histograms:

```python
def kldivergence(p, q):
    filt = np.logical_and(p != 0, q != 0)
    return np.sum(p[filt] * np.log2(p[filt] / q[filt]))
```

Lower divergence means more similar textures.

### 5. Sliding Window Classification
The image is scanned with a sliding window to classify each region:

```python
win_size = 50
step = 10

for y in range(0, leopardImage.shape[0] - win_size, step):
    for x in range(0, leopardImage.shape[1] - win_size, step):
        window = leopardImage[y:y+win_size, x:x+win_size]
        window_hist = get_lbp_hist(window)
        
        # Distance to leopard patches
        leopard_dists = [kldivergence(window_hist, h) for h in leopard_hists]
        min_leopard_dist = np.min(leopard_dists)
        
        # Distance to background patches
        nonleopard_dists = [kldivergence(window_hist, h) for h in nonleopard_hists]
        min_nonleopard_dist = np.min(nonleopard_dists)
        
        # Classify as leopard if similar to leopard patches
        if min_leopard_dist < 0.006 and min_leopard_dist < min_nonleopard_dist * 0.8:
            mask[y:y+win_size, x:x+win_size] = 255
```

### 6. Mask Refinement (Morphological Operations)
Multiple operations clean up the segmentation mask:

**Closing** - Fills small gaps:
```python
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
```

**Opening** - Removes small noise:
```python
kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
mask_opened = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel_open)
```

**Contour Filtering** - Keeps only the largest component (the leopard):
```python
contours, _ = cv2.findContours(mask_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if contours:
    largest_contour = max(contours, key=cv2.contourArea)
    mask_final = np.zeros_like(mask_opened)
    cv2.drawContours(mask_final, [largest_contour], -1, 255, -1)
```

### 7. Visualization
The notebook displays multiple stages of the segmentation process:
- Original image
- Initial mask
- After closing
- Final refined mask
- Segmentation overlay
- Extracted leopard

```python
plt.figure(figsize=(20, 10))
plt.subplot(2, 4, 1)
plt.imshow(leopardImage, cmap='gray')
plt.title('Original Image')
```

---

## Usage

Open and run the Jupyter Notebook:

```bash
jupyter notebook Leopard_segmentation.ipynb
```

Or run all cells in VS Code with Jupyter extension.

**Steps:**
1. Run the first cell to install dependencies
2. Execute cells sequentially to see each processing stage
3. Adjust patch coordinates to experiment with different training samples
4. Modify threshold (`0.006`) to tune sensitivity

---

## Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `P` | 8 | Number of LBP neighbors |
| `R` | 1 | LBP radius |
| `win_size` | 50 | Sliding window size (pixels) |
| `step` | 10 | Window stride for scanning |
| `threshold` | 0.006 | KL divergence threshold for classification |
| `kernel_close` | (25, 25) | Closing kernel size |
| `kernel_open` | (15, 15) | Opening kernel size |

---

## Limitations

- Requires **manual selection** of training patches
- Sensitive to **lighting and scale changes**
- May struggle with **non-textured backgrounds**
- Threshold tuning needed for different images
- Not rotation-invariant in current implementation
- Single image processing (not real-time)

---

## Best Use Cases

- Learning texture analysis techniques
- Understanding classical segmentation pipelines
- Image processing education
- Prototype development before deep learning
- Low-resource environments

---

## Possible Improvements

- **Automatic patch selection** using clustering
- **Multi-scale LBP** for scale invariance
- **Rotation-invariant LBP** variants
- **Adaptive thresholding** based on image statistics
- **Color information** integration (combine with RGB/HSV)
- **Graph-cut refinement** for smoother boundaries
- **Superpixel-based** processing for efficiency
- **Deep learning comparison** (U-Net, Mask R-CNN)

---

## Key Concepts Used

- **Local Binary Patterns (LBP)** - Texture descriptor
- **Histogram-based Representation** - Feature encoding
- **KL Divergence** - Similarity measurement
- **Sliding Window** - Spatial scanning technique
- **Morphological Operations** - Mask refinement
- **Contour Analysis** - Object extraction

---

## Algorithm Complexity

- **Feature Extraction**: O(N × P) per pixel, where N is image size, P is neighbors
- **Sliding Window**: O((H × W) / step²) windows to process
- **KL Divergence**: O(B × K) per window, B bins, K training patches
- **Overall**: Suitable for images up to ~1000×1000 pixels

---

## Results Interpretation

The notebook displays 6 key visualizations:

1. **Original Image** - Input grayscale leopard
2. **Initial Mask** - Raw classification before refinement
3. **After Closing** - Gaps filled in detected regions
4. **Final Mask** - Clean binary segmentation
5. **Segmentation Overlay** - Green overlay on original
6. **Extracted Leopard** - Leopard isolated on black background

Success metrics:
- Contour area (larger = better detection)
- Visual inspection of boundary accuracy
- Minimal false positives in background

---

## Mathematical Background

### Local Binary Pattern
For each pixel, compare with P neighbors at radius R:

$$
LBP_{P,R} = \sum_{p=0}^{P-1} s(g_p - g_c) \cdot 2^p
$$

where $s(x) = 1$ if $x \geq 0$, else $0$

### KL Divergence
Measures distributional difference:

$$
D_{KL}(P||Q) = \sum_{i} P(i) \log_2 \frac{P(i)}{Q(i)}
$$

Lower values indicate more similar textures.

---

## Summary

This project demonstrates how **texture-based segmentation** using Local Binary Patterns can effectively isolate objects with distinctive patterns from their backgrounds. While not as robust as modern deep learning approaches, it provides:

- Interpretable feature representation  
- Low computational requirements  
- No training data needed (few-shot learning)  
- Clear understanding of segmentation pipeline  

Perfect for learning classical computer vision before diving into neural networks!
