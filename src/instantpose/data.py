"""
Data loading utilities for LINEMOD, OCCLUSION_LINEMOD, and YCB-V datasets.
Handles RGB-D images, poses, and camera intrinsics.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from .utils import get_default_linemod_intrinsics, load_json


def read_rgb(path: str) -> np.ndarray:
    """
    Read RGB image.
    
    Args:
        path: Path to image file
        
    Returns:
        RGB image as uint8 array [H, W, 3]
    """
    img = Image.open(path).convert('RGB')
    return np.array(img, dtype=np.uint8)


def read_depth(path: str, scale: float = 1000.0) -> np.ndarray:
    """
    Read depth image and convert to meters.
    
    Args:
        path: Path to depth file (typically 16-bit PNG)
        scale: Depth scale factor (default 1000.0 for mm to meters)
        
    Returns:
        Depth map in meters as float32 array [H, W]
    """
    # Read depth as 16-bit
    depth = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    if depth is None:
        raise ValueError(f"Failed to read depth image: {path}")
    
    # Convert to meters
    depth = depth.astype(np.float32) / scale
    return depth


def read_pose(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read pose from text file (3x4 RT matrix).
    
    Args:
        path: Path to pose file
        
    Returns:
        Tuple of (R [3, 3], t [3])
    """
    pose = np.loadtxt(path, dtype=np.float32)
    
    if pose.shape == (3, 4):
        R = pose[:, :3]
        t = pose[:, 3]
    elif pose.shape == (4, 4):
        R = pose[:3, :3]
        t = pose[:3, 3]
    else:
        raise ValueError(f"Unexpected pose shape: {pose.shape}")
    
    return R, t


def load_occlusion_linemod(
    root: str,
    object_id: str,
    start_idx: int = 1,
    end_idx: int = -1
) -> Dict:
    """
    Load OCCLUSION_LINEMOD dataset.
    
    Directory structure expected:
    OCCLUSION_LINEMOD/
      RGB-D/
        color_*.png
        depth_*.png
      poses/
        *.txt
      models/
        obj_*.ply
      intrinsics.json (optional)
    
    Args:
        root: Root directory of OCCLUSION_LINEMOD
        object_id: Object name (e.g., 'ape', 'cat')
        start_idx: Starting frame index
        end_idx: Ending frame index (-1 for all)
        
    Returns:
        Dictionary with keys:
            - K: camera intrinsics [3, 3]
            - frames: list of dicts with {rgb, depth, pose, stem}
            - object_id: object name
    """
    root = Path(root)
    
    # Check if root exists
    if not root.exists():
        raise ValueError(f"Dataset root not found: {root}")
    
    # Try to load intrinsics
    intrinsics_path = root / "intrinsics.json"
    if intrinsics_path.exists():
        intrinsics_data = load_json(intrinsics_path)
        K = np.array(intrinsics_data['camera_matrix'], dtype=np.float32).reshape(3, 3)
    else:
        # Use default LINEMOD intrinsics
        K = get_default_linemod_intrinsics()
    
    # Find RGB-D directory
    rgbd_dir = root / "RGB-D"
    if not rgbd_dir.exists():
        # If direct RGB-D folder doesn't exist, check if we are already IN the RGB-D folder context or deeper
        # But typically for OCCLUSION_LINEMOD root, it should have RGB-D.
        # Let's check if the user pointed to the inner OCCLUSION_LINEMOD folder or the outer one.
        # Structure is typically OCCLUSION_LINEMOD/OCCLUSION_LINEMOD/RGB-D
        # Check if there's a nested OCCLUSION_LINEMOD directory
        nested_dir = root / "OCCLUSION_LINEMOD" / "RGB-D"
        if nested_dir.exists():
            rgbd_dir = nested_dir
            # Also update root to be the inner one for poses/models search
            root = root / "OCCLUSION_LINEMOD"
        else:
             raise ValueError(f"RGB-D directory not found in {root}")
    
    # Find all color images (check both direct path and rgb_noseg subdirectory)
    color_files = sorted(rgbd_dir.glob("color_*.png"))
    if len(color_files) == 0:
        # Try rgb_noseg subdirectory
        rgb_noseg_dir = rgbd_dir / "rgb_noseg"
        if rgb_noseg_dir.exists():
            color_files = sorted(rgb_noseg_dir.glob("color_*.png"))
    
    if len(color_files) == 0:
        raise ValueError(f"No color images found in {rgbd_dir} or {rgbd_dir / 'rgb_noseg'}")
    
    # Process frames
    frames = []
    for color_path in color_files:
        # Extract frame number from filename (e.g., color_00001.png -> 1)
        stem = color_path.stem
        frame_num = int(stem.split('_')[1])
        
        # Skip if outside requested range
        if frame_num < start_idx:
            continue
        if end_idx > 0 and frame_num > end_idx:
            break
        
        # Find corresponding depth
        depth_path = rgbd_dir / f"depth_{stem.split('_')[1]}.png"
        if not depth_path.exists():
            # Try depth_noseg subdirectory
            depth_noseg_dir = rgbd_dir / "depth_noseg"
            if depth_noseg_dir.exists():
                depth_path = depth_noseg_dir / f"depth_{stem.split('_')[1]}.png"
        
        if not depth_path.exists():
            print(f"Warning: Depth not found for {color_path.name}, skipping")
            continue
        
        # Find corresponding pose
        poses_dir = root / "poses"
        pose_path = poses_dir / f"pose{frame_num}.txt"
        
        frame_data = {
            'rgb': str(color_path),
            'depth': str(depth_path),
            'pose': str(pose_path) if pose_path.exists() else None,
            'stem': f"{frame_num:05d}",
            'frame_num': frame_num
        }
        frames.append(frame_data)
    
    if len(frames) == 0:
        raise ValueError(f"No valid frames found in {rgbd_dir}")
    
    return {
        'K': K,
        'frames': frames,
        'object_id': object_id,
        'dataset': 'OCCLUSION_LINEMOD'
    }


def load_linemod_sequence(
    seq_path: str,
    start_idx: int = 1,
    end_idx: int = -1
) -> Dict:
    """
    Load a single LINEMOD object sequence.
    
    Directory structure expected:
    <object>/
      JPEGImages/
        *.jpg
      depth/
        *.png
      poses/
        *.txt (optional)
      gt.yml or poses.txt
      camera.json (optional)
    
    Args:
        seq_path: Path to object sequence directory
        start_idx: Starting frame index
        end_idx: Ending frame index (-1 for all)
        
    Returns:
        Dictionary with keys:
            - K: camera intrinsics [3, 3]
            - frames: list of dicts with {rgb, depth, pose, stem}
            - object_id: object name
    """
    seq_path = Path(seq_path)
    
    if not seq_path.exists():
        raise ValueError(f"Sequence path not found: {seq_path}")
    
    object_id = seq_path.name
    
    # Try to load intrinsics
    camera_path = seq_path / "camera.json"
    if camera_path.exists():
        camera_data = load_json(camera_path)
        K = np.array(camera_data['K'], dtype=np.float32).reshape(3, 3)
    else:
        K = get_default_linemod_intrinsics()
    
    # Find RGB directory
    rgb_dir = seq_path / "JPEGImages"
    if not rgb_dir.exists():
        rgb_dir = seq_path / "rgb"
    if not rgb_dir.exists():
        raise ValueError(f"RGB directory not found in {seq_path}")
    
    # Find depth directory
    depth_dir = seq_path / "depth"
    if not depth_dir.exists():
        print(f"Warning: Depth directory not found in {seq_path}")
        depth_dir = None
    
    # Find RGB files
    rgb_files = sorted(list(rgb_dir.glob("*.jpg")) + list(rgb_dir.glob("*.png")))
    if len(rgb_files) == 0:
        raise ValueError(f"No RGB images found in {rgb_dir}")
    
    # Process frames
    frames = []
    for rgb_path in rgb_files:
        stem = rgb_path.stem
        
        # Try to extract frame number
        try:
            frame_num = int(''.join(filter(str.isdigit, stem)))
        except:
            frame_num = len(frames) + 1
        
        # Skip if outside requested range
        if frame_num < start_idx:
            continue
        if end_idx > 0 and frame_num > end_idx:
            break
        
        # Find corresponding depth
        depth_path = None
        if depth_dir is not None:
            for ext in ['.png', '.jpg']:
                candidate = depth_dir / f"{stem}{ext}"
                if candidate.exists():
                    depth_path = candidate
                    break
        
        # Find corresponding pose
        pose_path = seq_path / "poses" / f"{stem}.txt"
        if not pose_path.exists():
            pose_path = None
        
        frame_data = {
            'rgb': str(rgb_path),
            'depth': str(depth_path) if depth_path else None,
            'pose': str(pose_path) if pose_path else None,
            'stem': stem,
            'frame_num': frame_num
        }
        frames.append(frame_data)
    
    if len(frames) == 0:
        raise ValueError(f"No valid frames found in {seq_path}")
    
    return {
        'K': K,
        'frames': frames,
        'object_id': object_id,
        'dataset': 'LINEMOD'
    }


def load_ycbv_sequence(
    root: str,
    seq_id: int,
    start_idx: int = 1,
    end_idx: int = -1
) -> Dict:
    """
    Load a YCB-V sequence (BOP format).
    
    Directory structure expected:
    YCBV/
      test/
        {seq_id:06d}/
          rgb/
            *.png or *.jpg
          depth/
            *.png
          scene_gt.json
          scene_camera.json (for intrinsics)
    
    Args:
        root: Root directory of YCBV dataset
        seq_id: Sequence ID (e.g., 48)
        start_idx: Starting frame index
        end_idx: Ending frame index (-1 for all)
        
    Returns:
        Dictionary with keys:
            - K: camera intrinsics [3, 3]
            - frames: list of dicts with {rgb, depth, pose, stem}
            - object_id: seq_id as string
            - dataset: 'YCBV'
    """
    root = Path(root)
    seq_dir = root / "test" / f"{seq_id:06d}"
    
    if not seq_dir.exists():
        raise ValueError(f"Sequence directory not found: {seq_dir}")
    
    # Load scene info for camera and GT
    scene_camera_path = seq_dir / "scene_camera.json"
    scene_gt_path = seq_dir / "scene_gt.json"
    
    if not scene_camera_path.exists() or not scene_gt_path.exists():
        raise ValueError(f"Missing scene info files in {seq_dir}")
    
    scene_camera = load_json(scene_camera_path)
    scene_gt = load_json(scene_gt_path)
    
    # Get first frame intrinsics (assume constant)
    # Keys are string frame numbers
    first_key = sorted(scene_camera.keys())[0]
    K = np.array(scene_camera[first_key]['cam_K'], dtype=np.float32).reshape(3, 3)
    
    rgb_dir = seq_dir / "rgb"
    depth_dir = seq_dir / "depth"
    
    if not rgb_dir.exists():
        raise ValueError(f"RGB directory not found in {seq_dir}")
    
    rgb_files = sorted(list(rgb_dir.glob("*.png")) + list(rgb_dir.glob("*.jpg")))
    
    frames = []
    for rgb_path in rgb_files:
        stem = rgb_path.stem
        frame_num_str = stem
        
        # Remove leading zeros for json lookup
        frame_key = str(int(frame_num_str))
        
        frame_num = int(frame_num_str)
        
        if frame_num < start_idx:
            continue
        if end_idx > 0 and frame_num > end_idx:
            break
        
        depth_path = depth_dir / f"{stem}.png"
        if not depth_path.exists():
            print(f"Warning: Depth not found for {rgb_path.name}, skipping")
            continue
            
        # Get GT pose if available
        pose_data = None
        if frame_key in scene_gt:
            # YCBV scenes have multiple objects. We need to filter or handle this.
            # For now, we just take the first object's pose or all of them?
            # The current pipeline expects ONE object ID. 
            # However, YCBV sequences are scenes with multiple objects.
            # The pipeline is "Instance-Level".
            # We should probably allow specifying which object we are looking for in this scene.
            # But load_ycbv_sequence is just loading frames.
            # Let's assume we want to extract the pose of a SPECIFIC object_id if passed?
            # But here we just return the list of objects present?
            # Wait, the pipeline iterates frames and calls estimate_pose_for_frame.
            # If we are tracking one object, we need to know which one.
            # Let's assume the user provides OBJECT_ID in config which matches the object ID in YCBV (e.g. 1 for master_chef_can).
            
            # We'll store the GT poses for ALL objects in the frame data
            # And let the main loop filter.
            gt_objs = scene_gt[frame_key]
            pose_data = gt_objs # List of dicts {cam_R_m2c, cam_t_m2c, obj_id}
            
        frame_data = {
            'rgb': str(rgb_path),
            'depth': str(depth_path),
            'gt_poses': pose_data, # Special key for YCBV
            'pose': None, # Standard key used by main.py (single pose), we might need to adapt main.py
            'stem': stem,
            'frame_num': frame_num
        }
        frames.append(frame_data)
        
    return {
        'K': K,
        'frames': frames,
        'object_id': str(seq_id), # Use seq_id as object_id placeholder for the scene
        'dataset': 'YCBV'
    }


def load_dataset(config: Dict) -> Dict:
    """
    Load dataset based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dataset dictionary
    """
    dataset_type = config['DATA']['DATASET']
    
    if dataset_type == 'OCCLUSION_LINEMOD':
        return load_occlusion_linemod(
            root=config['DATA']['DATA_ROOT'],
            object_id=config['DATA']['OBJECT_ID'],
            start_idx=config['DATA']['START_IDX'],
            end_idx=config['DATA']['END_IDX']
        )
    elif dataset_type == 'LINEMOD':
        seq_path = config['DATA'].get('SEQ_PATH')
        if seq_path is None:
            # Try to construct from DATA_ROOT and OBJECT_ID
            seq_path = Path(config['DATA']['DATA_ROOT']) / config['DATA']['OBJECT_ID']
        return load_linemod_sequence(
            seq_path=str(seq_path),
            start_idx=config['DATA']['START_IDX'],
            end_idx=config['DATA']['END_IDX']
        )
    elif dataset_type == 'YCBV':
        return load_ycbv_sequence(
            root=config['DATA']['DATA_ROOT'],
            seq_id=int(config['DATA'].get('SEQ_ID', 48)),  # Default to first test sequence
            start_idx=config['DATA']['START_IDX'],
            end_idx=config['DATA']['END_IDX']
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
