#!/bin/bash
# Example run commands for InstantPose

# Activate virtual environment (adjust path as needed)
# source .venv/bin/activate

echo "================================================================"
echo "InstantPose - Demo Scripts"
echo "================================================================"
echo ""

# Example 1: OCCLUSION_LINEMOD demo
echo "Example 1: OCCLUSION_LINEMOD Demo (first 10 frames)"
echo "----------------------------------------------------------------"
echo "python -m src.main --config configs/config.yaml \\"
echo "  DATA.DATASET OCCLUSION_LINEMOD \\"
echo "  DATA.DATA_ROOT /path/to/OCCLUSION_LINEMOD \\"
echo "  DATA.OBJECT_ID ape \\"
echo "  DATA.END_IDX 10 \\"
echo "  PATHS.MESH assets/recon_meshes/ape.obj"
echo ""

# Example 2: LINEMOD evaluation
echo "Example 2: LINEMOD Evaluation (full sequence)"
echo "----------------------------------------------------------------"
echo "python -m src.main --config configs/config.yaml \\"
echo "  MODE eval \\"
echo "  DATA.DATASET LINEMOD \\"
echo "  DATA.SEQ_PATH /path/to/LINEMOD/cat \\"
echo "  DATA.OBJECT_ID cat \\"
echo "  PATHS.MESH assets/recon_meshes/cat.obj"
echo ""

# Example 3: Custom rendering settings
echo "Example 3: High-quality rendering (more views)"
echo "----------------------------------------------------------------"
echo "python -m src.main --config configs/config.yaml \\"
echo "  DATA.DATASET OCCLUSION_LINEMOD \\"
echo "  DATA.DATA_ROOT /path/to/OCCLUSION_LINEMOD \\"
echo "  DATA.OBJECT_ID driller \\"
echo "  PATHS.MESH assets/recon_meshes/driller.obj \\"
echo "  RENDER.N_VIEWS 1000 \\"
echo "  RENDER.ELEVATION_DEG [0,90]"
echo ""

# Example 4: Different DINOv2 model
echo "Example 4: Using smaller DINOv2 model (faster)"
echo "----------------------------------------------------------------"
echo "python -m src.main --config configs/config.yaml \\"
echo "  DATA.DATASET OCCLUSION_LINEMOD \\"
echo "  DATA.DATA_ROOT /path/to/OCCLUSION_LINEMOD \\"
echo "  DATA.OBJECT_ID can \\"
echo "  PATHS.MESH assets/recon_meshes/can.obj \\"
echo "  MODEL.BACKBONE dinov2_vits14"
echo ""

echo "================================================================"
echo "To run any example, copy the command and replace paths with your actual data paths."
echo "================================================================"

