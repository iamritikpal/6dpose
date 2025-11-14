"""
Pose refinement using Umeyama similarity alignment and RANSAC.
Handles scale estimation for reconstructed meshes with arbitrary units.
"""

from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation


def umeyama_alignment(
    X_src: np.ndarray,
    X_dst: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute similarity transformation (rotation, translation, scale) using Umeyama's method.
    Solves: X_dst = s * R @ X_src + t
    
    Args:
        X_src: Source 3D points [N, 3]
        X_dst: Destination 3D points [N, 3]
        
    Returns:
        Tuple of (R, t, scale)
        - R: Rotation matrix [3, 3]
        - t: Translation vector [3]
        - scale: Uniform scale factor
    """
    assert X_src.shape == X_dst.shape, "Point sets must have same shape"
    assert X_src.shape[1] == 3, "Points must be 3D"
    
    n = X_src.shape[0]
    
    # Compute centroids
    src_mean = np.mean(X_src, axis=0)
    dst_mean = np.mean(X_dst, axis=0)
    
    # Center the points
    X_src_centered = X_src - src_mean
    X_dst_centered = X_dst - dst_mean
    
    # Compute covariance matrix
    H = X_src_centered.T @ X_dst_centered / n
    
    # SVD
    U, S, Vt = np.linalg.svd(H)
    
    # Compute rotation
    R = Vt.T @ U.T
    
    # Handle reflection case
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Compute scale
    src_var = np.mean(np.sum(X_src_centered ** 2, axis=1))
    scale = np.sum(S) / src_var if src_var > 1e-8 else 1.0
    
    # Compute translation
    t = dst_mean - scale * R @ src_mean
    
    return R, t, scale


def ransac_umeyama(
    X_src: np.ndarray,
    X_dst: np.ndarray,
    n_iters: int = 2000,
    inlier_thresh: float = 0.01,
    min_samples: int = 3
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """
    RANSAC-based Umeyama alignment for robust pose estimation.
    
    Args:
        X_src: Source 3D points [N, 3]
        X_dst: Destination 3D points [N, 3]
        n_iters: Number of RANSAC iterations
        inlier_thresh: Inlier threshold in meters
        min_samples: Minimum number of samples for model estimation
        
    Returns:
        Tuple of (R, t, scale, inlier_mask)
        - R: Rotation matrix [3, 3]
        - t: Translation vector [3]
        - scale: Uniform scale factor
        - inlier_mask: Boolean mask of inliers [N]
    """
    assert X_src.shape == X_dst.shape
    assert len(X_src) >= min_samples, f"Need at least {min_samples} points"
    
    n_points = len(X_src)
    best_inliers = None
    best_n_inliers = 0
    best_model = None
    
    for _ in range(n_iters):
        # Sample random subset
        sample_idx = np.random.choice(n_points, size=min_samples, replace=False)
        X_src_sample = X_src[sample_idx]
        X_dst_sample = X_dst[sample_idx]
        
        # Estimate model
        try:
            R, t, scale = umeyama_alignment(X_src_sample, X_dst_sample)
        except:
            continue
        
        # Transform all source points
        X_src_transformed = scale * (R @ X_src.T).T + t
        
        # Compute errors
        errors = np.linalg.norm(X_src_transformed - X_dst, axis=1)
        inliers = errors < inlier_thresh
        n_inliers = np.sum(inliers)
        
        # Update best model
        if n_inliers > best_n_inliers:
            best_n_inliers = n_inliers
            best_inliers = inliers
            best_model = (R, t, scale)
    
    if best_model is None or best_n_inliers < min_samples:
        # Fallback to all points if RANSAC fails
        print(f"Warning: RANSAC failed (only {best_n_inliers} inliers), using all points")
        R, t, scale = umeyama_alignment(X_src, X_dst)
        inlier_mask = np.ones(n_points, dtype=bool)
        return R, t, scale, inlier_mask
    
    # Refine on inliers
    X_src_inliers = X_src[best_inliers]
    X_dst_inliers = X_dst[best_inliers]
    R, t, scale = umeyama_alignment(X_src_inliers, X_dst_inliers)
    
    # Recompute inliers with refined model
    X_src_transformed = scale * (R @ X_src.T).T + t
    errors = np.linalg.norm(X_src_transformed - X_dst, axis=1)
    final_inliers = errors < inlier_thresh
    
    print(f"RANSAC: {np.sum(final_inliers)}/{n_points} inliers")
    
    return R, t, scale, final_inliers


def estimate_pose_from_correspondences(
    pts_ref: np.ndarray,
    pts_query: np.ndarray,
    ransac_config: dict
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Estimate 6D pose with scale from 3D-3D correspondences.
    
    Args:
        pts_ref: Reference 3D points [N, 3] (from template)
        pts_query: Query 3D points [N, 3] (from query image)
        ransac_config: RANSAC configuration dict
        
    Returns:
        Tuple of (R, t, scale, inlier_mask)
    """
    if len(pts_ref) < 3:
        raise ValueError(f"Need at least 3 correspondences, got {len(pts_ref)}")
    
    # Run RANSAC Umeyama
    R, t, scale, inliers = ransac_umeyama(
        X_src=pts_ref,
        X_dst=pts_query,
        n_iters=ransac_config['RANSAC_ITERS'],
        inlier_thresh=ransac_config['INLIER_THRESH']
    )
    
    return R, t, scale, inliers


def transform_points(
    points: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    scale: float = 1.0
) -> np.ndarray:
    """
    Apply similarity transformation to points: X_out = scale * R @ X + t
    
    Args:
        points: Input points [N, 3]
        R: Rotation matrix [3, 3]
        t: Translation vector [3]
        scale: Scale factor
        
    Returns:
        Transformed points [N, 3]
    """
    return scale * (R @ points.T).T + t


def invert_similarity_transform(
    R: np.ndarray,
    t: np.ndarray,
    scale: float
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Invert similarity transformation.
    If X_dst = s * R @ X_src + t, compute inverse such that X_src = s_inv * R_inv @ X_dst + t_inv
    
    Args:
        R: Rotation matrix [3, 3]
        t: Translation vector [3]
        scale: Scale factor
        
    Returns:
        Tuple of (R_inv, t_inv, scale_inv)
    """
    R_inv = R.T
    scale_inv = 1.0 / scale if scale > 1e-8 else 1.0
    t_inv = -scale_inv * R_inv @ t
    
    return R_inv, t_inv, scale_inv


def compute_pose_error(
    R_gt: np.ndarray,
    t_gt: np.ndarray,
    R_pred: np.ndarray,
    t_pred: np.ndarray
) -> Tuple[float, float]:
    """
    Compute rotation and translation errors.
    
    Args:
        R_gt: Ground truth rotation [3, 3]
        t_gt: Ground truth translation [3]
        R_pred: Predicted rotation [3, 3]
        t_pred: Predicted translation [3]
        
    Returns:
        Tuple of (rotation_error_deg, translation_error_meters)
    """
    # Rotation error (geodesic distance on SO(3))
    R_err = R_pred @ R_gt.T
    rot_err_rad = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))
    rot_err_deg = np.rad2deg(rot_err_rad)
    
    # Translation error (Euclidean distance)
    trans_err = np.linalg.norm(t_pred - t_gt)
    
    return rot_err_deg, trans_err


def apply_scale_to_mesh_vertices(
    vertices: np.ndarray,
    scale: float
) -> np.ndarray:
    """
    Apply scale to mesh vertices (used to correct reconstructed mesh scale).
    
    Args:
        vertices: Mesh vertices [N, 3]
        scale: Scale factor
        
    Returns:
        Scaled vertices [N, 3]
    """
    return vertices * scale

