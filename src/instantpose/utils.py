"""
Utility functions for InstantPose pipeline.
Includes projection, transformation, and I/O helpers.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch


def set_seed(seed: int = 0) -> None:
    """
    Set random seeds for reproducibility across numpy, torch, and random.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json(path: Union[str, Path]) -> Dict:
    """
    Load JSON file.
    
    Args:
        path: Path to JSON file
        
    Returns:
        Parsed JSON data
    """
    with open(path, 'r') as f:
        return json.load(f)


def save_json(data: Dict, path: Union[str, Path]) -> None:
    """
    Save data to JSON file.
    
    Args:
        data: Dictionary to save
        path: Output path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def project_points(
    pts_3d: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray
) -> np.ndarray:
    """
    Project 3D points to 2D image coordinates.
    
    Args:
        pts_3d: 3D points in world coordinates [N, 3]
        K: Camera intrinsics [3, 3]
        R: Rotation matrix [3, 3]
        t: Translation vector [3, 1] or [3]
        
    Returns:
        2D projected points [N, 2]
    """
    # Ensure t is column vector
    t = t.reshape(3, 1) if t.ndim == 1 else t
    
    # Transform to camera coordinates
    pts_cam = (R @ pts_3d.T + t).T  # [N, 3]
    
    # Project to image
    pts_2d_homog = (K @ pts_cam.T).T  # [N, 3]
    pts_2d = pts_2d_homog[:, :2] / (pts_2d_homog[:, 2:3] + 1e-8)
    
    return pts_2d


def compose_rt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Compose rotation and translation into 4x4 transformation matrix.
    
    Args:
        R: Rotation matrix [3, 3]
        t: Translation vector [3] or [3, 1]
        
    Returns:
        Transformation matrix [4, 4]
    """
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()
    return T


def decompose_rt(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decompose 4x4 transformation matrix into rotation and translation.
    
    Args:
        T: Transformation matrix [4, 4]
        
    Returns:
        Tuple of (R [3, 3], t [3])
    """
    R = T[:3, :3]
    t = T[:3, 3]
    return R, t


def depth_to_points(
    depth: np.ndarray,
    K: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Convert depth map to 3D point cloud in camera coordinates.
    
    Args:
        depth: Depth map [H, W] in meters
        K: Camera intrinsics [3, 3]
        mask: Optional binary mask [H, W] to select valid pixels
        
    Returns:
        3D points [N, 3] in camera coordinates
    """
    H, W = depth.shape
    
    # Create pixel grid
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    
    # Apply mask if provided
    if mask is not None:
        valid = (depth > 0) & (mask > 0)
    else:
        valid = depth > 0
    
    u = u[valid]
    v = v[valid]
    z = depth[valid]
    
    # Backproject to 3D
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    points = np.stack([x, y, z], axis=-1)
    return points


def normalize_points(points: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Normalize point cloud to unit sphere centered at origin.
    
    Args:
        points: Input points [N, 3]
        
    Returns:
        Tuple of (normalized_points, scale, center)
    """
    center = np.mean(points, axis=0)
    points_centered = points - center
    scale = np.max(np.linalg.norm(points_centered, axis=1))
    points_normalized = points_centered / (scale + 1e-8)
    return points_normalized, scale, center


def get_default_linemod_intrinsics() -> np.ndarray:
    """
    Get default LINEMOD camera intrinsics.
    
    Returns:
        Camera intrinsics matrix [3, 3]
    """
    K = np.array([
        [572.4114, 0.0, 325.2611],
        [0.0, 573.5704, 242.0489],
        [0.0, 0.0, 1.0]
    ])
    return K


def compute_model_diameter(points: np.ndarray) -> float:
    """
    Compute the diameter of a 3D model (max distance between any two points).
    Uses a sampling approach for efficiency on large point clouds.
    
    Args:
        points: 3D points [N, 3]
        
    Returns:
        Model diameter
    """
    # For large point clouds, sample subset
    if len(points) > 1000:
        idx = np.random.choice(len(points), 1000, replace=False)
        points = points[idx]
    
    # Compute pairwise distances (approximate with max range)
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    diameter = np.linalg.norm(max_coords - min_coords)
    
    return diameter


def rotation_matrix_to_axis_angle(R: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Convert rotation matrix to axis-angle representation.
    
    Args:
        R: Rotation matrix [3, 3]
        
    Returns:
        Tuple of (axis [3], angle in radians)
    """
    angle = np.arccos((np.trace(R) - 1) / 2)
    if np.sin(angle) < 1e-6:
        # Small angle, return arbitrary axis
        return np.array([1., 0., 0.]), angle
    
    axis = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1]
    ]) / (2 * np.sin(angle))
    
    return axis, angle


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

