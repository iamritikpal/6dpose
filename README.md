# InstantPose-LINEMOD: Training-Free 6D Pose Estimation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A **production-ready** implementation of training-free 6D pose estimation inspired by **InstantPose: Zero-Shot Instance-Level 6D Pose Estimation From a Single View**. This pipeline estimates accurate 6D poses `(R, t, λ)` from RGB-D queries using foundation model features (DINOv2), cycle-consistent matching, and robust pose refinement.

## 🎯 Key Features

- **Training-Free**: No object-specific training required
- **Single Mesh Input**: Works with reconstructed meshes from Large Reconstruction Models (e.g., InstantMesh)
- **Foundation Features**: Leverages DINOv2 for robust visual features
- **Cycle-Consistent Matching**: Soft Nearest Neighbor (SNN) matching for reliable correspondences
- **Scale Estimation**: Handles arbitrary mesh scales via similarity transformation (Umeyama + RANSAC)
- **Standard Metrics**: ADD, ADD-S, and 2D reprojection error
- **Dataset Support**: LINEMOD and OCCLUSION_LINEMOD
- **Comprehensive Visualization**: Frame-by-frame pose overlays and error analysis plots

## 📋 Table of Contents

- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Mesh Preparation](#mesh-preparation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Output](#output)
- [Evaluation](#evaluation)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [License](#license)

## 🚀 Installation

### Requirements

- Python 3.10 or higher
- CUDA-capable GPU (recommended, 8GB+ VRAM)
- 16GB+ RAM

### Setup

1. **Clone or navigate to the repository:**

```bash
cd /path/to/6dpose
```

2. **Create and activate virtual environment:**

```bash
# Linux/Mac
python -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

**Note on offscreen rendering**: For headless servers (no display), ensure EGL is available:

```bash
# Ubuntu/Debian
sudo apt-get install libegl1-mesa-dev

# The code automatically sets PYOPENGL_PLATFORM='egl'
```

## 📁 Dataset Preparation

### OCCLUSION_LINEMOD (LM-O)

Expected directory structure:

```
OCCLUSION_LINEMOD/
├── RGB-D/
│   ├── color_00001.png
│   ├── color_00002.png
│   ├── depth_00001.png
│   ├── depth_00002.png
│   └── ...
├── poses/
│   ├── pose1.txt
│   ├── pose2.txt
│   └── ...
├── models/
│   ├── obj_01.ply   (ape)
│   ├── obj_05.ply   (can)
│   └── ...
└── intrinsics.json (optional)
```

**Download sources:**
- BOP Challenge: https://bop.felk.cvut.cz/datasets/
- Look for "LM-O" dataset

**intrinsics.json format** (optional, falls back to default LINEMOD intrinsics):

```json
{
  "camera_matrix": [
    [572.4114, 0.0, 325.2611],
    [0.0, 573.5704, 242.0489],
    [0.0, 0.0, 1.0]
  ]
}
```

### LINEMOD

Expected directory structure for each object:

```
LINEMOD/
└── <object_name>/    (e.g., ape, cat, duck)
    ├── JPEGImages/
    │   ├── 000000.jpg
    │   ├── 000001.jpg
    │   └── ...
    ├── depth/
    │   ├── 000000.png
    │   ├── 000001.png
    │   └── ...
    ├── poses/
    │   ├── 000000.txt
    │   ├── 000001.txt
    │   └── ...
    └── camera.json (optional)
```

**Download sources:**
- Original: http://ptak.felk.cvut.cz/6DB/public/datasets/hinterstoisser/
- BOP format: https://bop.felk.cvut.cz/datasets/

### Depth Format

Depth images are typically 16-bit PNG files in **millimeters**. The code automatically converts to meters:

```python
depth_meters = depth_uint16 / 1000.0
```

### Pose Format

Pose files are text files with 3×4 or 4×4 transformation matrices (object-to-camera):

```
R11 R12 R13 t1
R21 R22 R23 t2
R31 R32 R33 t3
[0   0   0   1]  (optional 4th row)
```

## 🎨 Mesh Preparation

### Using Reconstructed Meshes

This pipeline is designed to work with **reconstructed meshes** from Large Reconstruction Models (LRMs) such as:

- **InstantMesh** (recommended)
- TripoSR
- OpenLRM
- Shap-E

### Obtaining Meshes

1. **From reference images**: Use an LRM to reconstruct the object from one or more reference images
2. **From existing datasets**: If available, use ground truth meshes (but scale will be normalized)

### Mesh Placement

Place your reconstructed meshes in `assets/recon_meshes/`:

```bash
assets/recon_meshes/
├── ape.obj
├── cat.ply
├── driller.obj
└── ...
```

**Supported formats**: `.obj`, `.ply`

**Important notes:**
- The pipeline automatically normalizes meshes to unit scale
- The scale factor `λ` in the estimated pose corrects for arbitrary reconstruction scales
- Mesh quality directly affects pose accuracy

## 🎮 Usage

### Basic Demo

Run on OCCLUSION_LINEMOD dataset:

```bash
python -m src.main --config configs/config.yaml \
  DATA.DATASET OCCLUSION_LINEMOD \
  DATA.DATA_ROOT /path/to/OCCLUSION_LINEMOD \
  DATA.OBJECT_ID ape \
  PATHS.MESH assets/recon_meshes/ape.obj
```

### Evaluation Mode

For quantitative evaluation (requires ground truth poses):

```bash
python -m src.main --config configs/config.yaml \
  MODE eval \
  DATA.DATASET OCCLUSION_LINEMOD \
  DATA.DATA_ROOT /path/to/OCCLUSION_LINEMOD \
  DATA.OBJECT_ID cat \
  PATHS.MESH assets/recon_meshes/cat.obj
```

### LINEMOD Dataset

For classic LINEMOD sequences:

```bash
python -m src.main --config configs/config.yaml \
  DATA.DATASET LINEMOD \
  DATA.SEQ_PATH /path/to/LINEMOD/driller \
  DATA.OBJECT_ID driller \
  PATHS.MESH assets/recon_meshes/driller.obj
```

### Processing Subset of Frames

```bash
python -m src.main --config configs/config.yaml \
  DATA.DATA_ROOT /path/to/OCCLUSION_LINEMOD \
  DATA.OBJECT_ID can \
  DATA.START_IDX 1 \
  DATA.END_IDX 50 \
  PATHS.MESH assets/recon_meshes/can.obj
```

### High-Quality Settings

For better accuracy (slower):

```bash
python -m src.main --config configs/config.yaml \
  DATA.DATA_ROOT /path/to/OCCLUSION_LINEMOD \
  DATA.OBJECT_ID duck \
  PATHS.MESH assets/recon_meshes/duck.obj \
  RENDER.N_VIEWS 1000 \
  MATCH.TOPK 5000 \
  REFINE.RANSAC_ITERS 5000
```

## ⚙️ Configuration

The main configuration file is `configs/config.yaml`. Key parameters:

### Model Settings

```yaml
MODEL:
  BACKBONE: dinov2_vitb14  # Options: dinov2_vits14, dinov2_vitb14, dinov2_vitl14
  WEIGHTS: null            # Path to local weights (null = download from timm)
```

**Model sizes:**
- `dinov2_vits14`: Small, fastest (~22M params)
- `dinov2_vitb14`: Base, balanced (~86M params, recommended)
- `dinov2_vitl14`: Large, most accurate (~300M params, slowest)

### Rendering Settings

```yaml
RENDER:
  N_VIEWS: 400             # Number of template viewpoints
  ELEVATION_DEG: [0, 60]   # Elevation range (degrees)
  AZIMUTH_DEG: [0, 360]    # Azimuth range (degrees)
  DISTANCE: 0.7            # Camera distance from object
  IMG_SIZE: [480, 640]     # Image size [H, W]
  FOV_DEG: 57              # Field of view
```

**Tuning tips:**
- More views (e.g., 1000) → better coverage but slower
- Adjust elevation range based on expected object orientations
- Match IMG_SIZE to your query resolution for better feature alignment

### Matching Settings

```yaml
MATCH:
  TOPK: 2000  # Maximum number of matches to consider
```

### Refinement Settings

```yaml
REFINE:
  RANSAC_ITERS: 2000      # Number of RANSAC iterations
  INLIER_THRESH: 0.01     # RANSAC inlier threshold (meters)
```

### Evaluation Settings

```yaml
EVAL:
  ADD_THRESH: 0.1                      # ADD success threshold (fraction of diameter)
  SYMMETRIC_OBJECTS: ['eggbox', 'glue'] # Objects requiring ADD-S metric
```

## 📊 Output

After running, the `outputs/` directory contains:

```
outputs/
├── vis/                    # Frame visualizations
│   ├── 00001.png
│   ├── 00002.png
│   └── ...
├── curves/                 # Error plots
│   ├── summary.png         # 4-panel summary
│   ├── add_histogram.png
│   ├── add_success.png
│   └── ...
└── metrics.json            # Quantitative results
```

### Visualizations

Each frame visualization (`vis/*.png`) shows:
- **Green axes/bbox**: Predicted pose
- **Blue axes/bbox**: Ground truth pose (if available)
- **Legend**: Color coding at bottom

### Metrics JSON

Example `metrics.json`:

```json
{
  "mean_add": 0.0234,
  "median_add": 0.0189,
  "add_auc": 0.7821,
  "add_success_rate": 0.68,
  "mean_reproj": 12.3,
  "mean_rot": 8.45,
  "mean_trans": 0.0287,
  "num_frames": 120
}
```

## 📈 Evaluation

### Metrics

**ADD (Average Distance)**: Average distance between transformed model points

```
ADD = (1/N) Σ ||R_gt @ x_i + t_gt - (R_pred @ x_i + t_pred)||
```

**ADD-S (Symmetric)**: For symmetric objects, uses closest point distance

**Success Rate**: Percentage of frames with ADD < threshold (typically 10% of object diameter)

**2D Reprojection Error**: Average pixel distance of projected model points

### Symmetric Objects

For objects like `eggbox` and `glue`, the code automatically uses ADD-S. Configure in:

```yaml
EVAL:
  SYMMETRIC_OBJECTS: ['eggbox', 'glue']
```

### Running Tests

Verify installation with unit tests:

```bash
# Test Umeyama alignment
python tests/test_umeyama.py

# Test data loading
python tests/test_data_loading.py
```

## 🔧 Troubleshooting

### Issue: "CUDA out of memory"

**Solutions:**
1. Use smaller DINOv2 model: `MODEL.BACKBONE dinov2_vits14`
2. Reduce template views: `RENDER.N_VIEWS 200`
3. Process fewer frames: `DATA.END_IDX 10`

### Issue: Offscreen rendering fails

**Error**: `RuntimeError: No EGL device`

**Solution** (Linux):
```bash
sudo apt-get install libegl1-mesa-dev
export PYOPENGL_PLATFORM=egl
```

**Solution** (Windows): Ensure GPU drivers are up-to-date

### Issue: "Too few matches"

**Possible causes:**
- Mesh quality is poor
- Object appearance differs significantly from reconstruction
- Lighting/viewpoint differences

**Solutions:**
1. Increase template views: `RENDER.N_VIEWS 1000`
2. Try different elevation range: `RENDER.ELEVATION_DEG [0, 90]`
3. Use larger DINOv2: `MODEL.BACKBONE dinov2_vitl14`

### Issue: Poor pose accuracy

**Solutions:**
1. Increase RANSAC iterations: `REFINE.RANSAC_ITERS 5000`
2. Adjust inlier threshold: `REFINE.INLIER_THRESH 0.005` (stricter)
3. Check depth calibration
4. Verify mesh quality and scale

### Issue: Slow performance

**Optimizations:**
1. Use GPU: Check `torch.cuda.is_available()`
2. Use smaller backbone: `dinov2_vits14`
3. Reduce views: `RENDER.N_VIEWS 200`
4. Limit frames: `DATA.END_IDX 50`

## 📚 Citation

This implementation is inspired by the InstantPose paper:

```bibtex
@article{lee2024instantpose,
  title={InstantPose: Zero-Shot Instance-Level 6D Pose Estimation From a Single View},
  author={Lee, Jiaming and Xu, Xihui and others},
  journal={arXiv preprint},
  year={2024}
}
```

**DINOv2**:
```bibtex
@article{oquab2023dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and Darcet, Timothée and others},
  journal={TMLR},
  year={2023}
}
```

**LINEMOD Dataset**:
```bibtex
@inproceedings{hinterstoisser2012model,
  title={Model based training, detection and pose estimation of texture-less 3d objects in heavily cluttered scenes},
  author={Hinterstoisser, Stefan and others},
  booktitle={ACCV},
  year={2012}
}
```

## 🛠️ Technical Details

### Pipeline Overview

1. **Template Generation**: Render N views of the mesh from spherical camera positions
2. **Feature Extraction**: Extract dense DINOv2 features from query and all templates
3. **Template Selection**: Find best-matching template via cycle-consistent feature counts
4. **Correspondence Matching**: Compute SNN matches between query and best template
5. **3D Lifting**: Backproject 2D matches to 3D using depth maps and intrinsics
6. **Pose Estimation**: RANSAC + Umeyama to estimate (R, t, λ)
7. **Evaluation**: Compute ADD/ADD-S and visualize results

### Scale Estimation Rationale

Reconstructed meshes from LRMs have **arbitrary scale**. The Umeyama similarity transformation solves:

```
X_query = λ * R @ X_template + t
```

Where `λ` corrects the mesh scale to metric units.

### Cycle-Consistent Matching

For each query feature, we find:
1. Nearest neighbor in template: `q → r`
2. Nearest neighbor back: `r → q'`
3. Keep only if `q' == q` (cycle-consistent)

This filters unreliable matches and improves RANSAC convergence.

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Support for other foundation models (SAM, CLIP)
- Online mesh refinement
- Multi-object scenarios
- BOP toolkit integration

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

## 🙏 Acknowledgments

- InstantPose authors for the methodology
- Meta AI for DINOv2
- BOP Challenge for standardized datasets and evaluation
- LINEMOD dataset creators

---

**Maintained by**: InstantPose-LINEMOD Contributors  
**Issues**: Please report bugs and feature requests via GitHub Issues  
**Version**: 1.0.0

