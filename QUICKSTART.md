# Quick Start Guide

## 1. Install Dependencies (2 minutes)

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
```

## 2. Run Tests (1 minute)

Verify installation:

```bash
python tests/test_umeyama.py
python tests/test_data_loading.py
```

## 3. Prepare Your Data

### Option A: Using OCCLUSION_LINEMOD (already present)

You have OCCLUSION_LINEMOD at: `D:\University\Classes\CV\6dpose\OCCLUSION_LINEMOD`

### Option B: Using LINEMOD (already present)

You have LINEMOD at: `D:\University\Classes\CV\6dpose\LINEMOD`

## 4. Get a Reconstructed Mesh

Place a reconstructed mesh (e.g., from InstantMesh) in:

```
assets/recon_meshes/ape.obj
```

Alternatively, you can use the ground truth mesh from the dataset:
- OCCLUSION_LINEMOD: `OCCLUSION_LINEMOD/OCCLUSION_LINEMOD/models/obj_01.obj` (ape)
- LINEMOD: `LINEMOD/LINEMOD/ape/ape.ply`

Example copying ground truth mesh:

```bash
# Windows PowerShell
Copy-Item "OCCLUSION_LINEMOD\OCCLUSION_LINEMOD\models\obj_01.obj" "assets\recon_meshes\ape.obj"

# Linux/Mac
cp OCCLUSION_LINEMOD/OCCLUSION_LINEMOD/models/obj_01.obj assets/recon_meshes/ape.obj
```

## 5. Run Demo (5-10 minutes)

**Test on first 5 frames:**

```bash
python -m src.main --config configs/config.yaml \
  DATA.DATASET OCCLUSION_LINEMOD \
  DATA.DATA_ROOT OCCLUSION_LINEMOD/OCCLUSION_LINEMOD \
  DATA.OBJECT_ID ape \
  DATA.END_IDX 5 \
  PATHS.MESH assets/recon_meshes/ape.obj
```

**Windows PowerShell version:**

```powershell
python -m src.main --config configs/config.yaml DATA.DATASET OCCLUSION_LINEMOD DATA.DATA_ROOT OCCLUSION_LINEMOD/OCCLUSION_LINEMOD DATA.OBJECT_ID ape DATA.END_IDX 5 PATHS.MESH assets/recon_meshes/ape.obj
```

## 6. Check Results

```
outputs/
├── vis/              # Frame-by-frame visualizations
├── curves/           # Error plots (if eval mode)
└── metrics.json      # Quantitative results (if eval mode)
```

## Troubleshooting

### "Mesh not found"
Place your mesh at: `assets/recon_meshes/ape.obj`

### "CUDA out of memory"
Use smaller model:
```bash
python -m src.main --config configs/config.yaml \
  ... \
  MODEL.BACKBONE dinov2_vits14
```

### "No module named 'instantpose'"
Make sure you run from the project root and use `python -m src.main`

## Next Steps

1. **Full evaluation:**
   ```bash
   python -m src.main --config configs/config.yaml \
     MODE eval \
     DATA.DATASET OCCLUSION_LINEMOD \
     DATA.DATA_ROOT OCCLUSION_LINEMOD/OCCLUSION_LINEMOD \
     DATA.OBJECT_ID ape \
     PATHS.MESH assets/recon_meshes/ape.obj
   ```

2. **Try other objects:** cat, can, driller, duck, etc.

3. **Adjust settings:** See `configs/config.yaml` and README.md

## System Requirements

- **GPU:** Recommended (8GB+ VRAM)
- **RAM:** 16GB+
- **Time:** ~1-2 minutes per frame (depends on GPU)

