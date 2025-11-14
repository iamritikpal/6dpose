"""
Evaluation metrics for 6D pose estimation.
Implements ADD, ADD-S (for symmetric objects), and 2D reprojection error.
"""

from typing import Optional, Tuple

import numpy as np
import trimesh

from .utils import project_points


def add_metric(
    mesh_pts: np.ndarray,
    R_gt: np.ndarray,
    t_gt: np.ndarray,
    R_pred: np.ndarray,
    t_pred: np.ndarray,
    symmetric: bool = False
) -> float:
    """
    Compute Average Distance (ADD) or ADD-S metric.
    
    ADD: Average distance between transformed model points.
    ADD-S: Average closest point distance (for symmetric objects).
    
    Args:
        mesh_pts: Model 3D points [N, 3]
        R_gt: Ground truth rotation [3, 3]
        t_gt: Ground truth translation [3]
        R_pred: Predicted rotation [3, 3]
        t_pred: Predicted translation [3]
        symmetric: Whether object is symmetric (use ADD-S)
        
    Returns:
        ADD or ADD-S error (average distance in meters)
    """
    # Transform points with ground truth pose
    pts_gt = (R_gt @ mesh_pts.T).T + t_gt
    
    # Transform points with predicted pose
    pts_pred = (R_pred @ mesh_pts.T).T + t_pred
    
    if symmetric:
        # ADD-S: Find closest point distances
        from scipy.spatial import cKDTree
        tree = cKDTree(pts_pred)
        distances, _ = tree.query(pts_gt)
        error = np.mean(distances)
    else:
        # ADD: Direct point-to-point distance
        distances = np.linalg.norm(pts_gt - pts_pred, axis=1)
        error = np.mean(distances)
    
    return error


def reproj_error(
    pts_3d: np.ndarray,
    K: np.ndarray,
    R_gt: np.ndarray,
    t_gt: np.ndarray,
    R_pred: np.ndarray,
    t_pred: np.ndarray
) -> float:
    """
    Compute 2D reprojection error.
    
    Args:
        pts_3d: Model 3D points [N, 3]
        K: Camera intrinsics [3, 3]
        R_gt: Ground truth rotation [3, 3]
        t_gt: Ground truth translation [3]
        R_pred: Predicted rotation [3, 3]
        t_pred: Predicted translation [3]
        
    Returns:
        Average 2D reprojection error in pixels
    """
    # Project with ground truth pose
    pts_2d_gt = project_points(pts_3d, K, R_gt, t_gt)
    
    # Project with predicted pose
    pts_2d_pred = project_points(pts_3d, K, R_pred, t_pred)
    
    # Compute error
    error = np.linalg.norm(pts_2d_gt - pts_2d_pred, axis=1)
    return np.mean(error)


def load_model_points(
    mesh_path: str,
    n_points: int = 1000
) -> np.ndarray:
    """
    Load and sample points from mesh for evaluation.
    
    Args:
        mesh_path: Path to mesh file
        n_points: Number of points to sample
        
    Returns:
        Sampled 3D points [N, 3]
    """
    mesh = trimesh.load(mesh_path)
    
    # Center and normalize (same as rendering)
    mesh.vertices -= mesh.centroid
    scale = np.max(np.linalg.norm(mesh.vertices, axis=1))
    mesh.vertices /= scale
    
    # Sample points uniformly
    if len(mesh.vertices) > n_points:
        indices = np.random.choice(len(mesh.vertices), n_points, replace=False)
        points = mesh.vertices[indices]
    else:
        points = mesh.vertices
    
    return points


def compute_add_auc(
    add_errors: np.ndarray,
    max_threshold: float = 0.1,
    n_bins: int = 100
) -> float:
    """
    Compute Area Under Curve (AUC) for ADD metric.
    
    Args:
        add_errors: Array of ADD errors
        max_threshold: Maximum threshold for AUC computation
        n_bins: Number of bins
        
    Returns:
        AUC score (0 to 1)
    """
    thresholds = np.linspace(0, max_threshold, n_bins)
    accuracies = []
    
    for thresh in thresholds:
        acc = np.mean(add_errors < thresh)
        accuracies.append(acc)
    
    # Compute AUC using trapezoidal rule
    auc = np.trapz(accuracies, thresholds) / max_threshold
    return auc


def compute_success_rate(
    errors: np.ndarray,
    threshold: float
) -> float:
    """
    Compute success rate (percentage of samples below threshold).
    
    Args:
        errors: Array of errors
        threshold: Success threshold
        
    Returns:
        Success rate (0 to 1)
    """
    return np.mean(errors < threshold)


class PoseEvaluator:
    """
    Evaluator for tracking and computing pose estimation metrics.
    """
    
    def __init__(
        self,
        mesh_path: str,
        symmetric: bool = False,
        n_model_pts: int = 1000
    ):
        """
        Initialize evaluator.
        
        Args:
            mesh_path: Path to mesh file
            symmetric: Whether object is symmetric
            n_model_pts: Number of model points to sample
        """
        self.mesh_path = mesh_path
        self.symmetric = symmetric
        self.model_pts = load_model_points(mesh_path, n_model_pts)
        
        self.add_errors = []
        self.reproj_errors = []
        self.rot_errors = []
        self.trans_errors = []
    
    def evaluate_frame(
        self,
        R_gt: np.ndarray,
        t_gt: np.ndarray,
        R_pred: np.ndarray,
        t_pred: np.ndarray,
        K: Optional[np.ndarray] = None
    ) -> dict:
        """
        Evaluate single frame.
        
        Args:
            R_gt: Ground truth rotation [3, 3]
            t_gt: Ground truth translation [3]
            R_pred: Predicted rotation [3, 3]
            t_pred: Predicted translation [3]
            K: Camera intrinsics [3, 3] (optional, for reprojection error)
            
        Returns:
            Dictionary of metrics
        """
        # Compute ADD or ADD-S
        add_err = add_metric(
            self.model_pts,
            R_gt, t_gt,
            R_pred, t_pred,
            symmetric=self.symmetric
        )
        self.add_errors.append(add_err)
        
        # Compute reprojection error if K is provided
        reproj_err = None
        if K is not None:
            reproj_err = reproj_error(
                self.model_pts,
                K,
                R_gt, t_gt,
                R_pred, t_pred
            )
            self.reproj_errors.append(reproj_err)
        
        # Compute rotation error
        R_err = R_pred @ R_gt.T
        rot_err_rad = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))
        rot_err_deg = np.rad2deg(rot_err_rad)
        self.rot_errors.append(rot_err_deg)
        
        # Compute translation error
        trans_err = np.linalg.norm(t_pred - t_gt)
        self.trans_errors.append(trans_err)
        
        metrics = {
            'add': add_err,
            'reproj': reproj_err,
            'rot_deg': rot_err_deg,
            'trans_m': trans_err
        }
        
        return metrics
    
    def get_summary(self, add_threshold: float = 0.1) -> dict:
        """
        Get summary statistics.
        
        Args:
            add_threshold: ADD threshold for success rate
            
        Returns:
            Dictionary of summary metrics
        """
        summary = {
            'mean_add': np.mean(self.add_errors) if self.add_errors else None,
            'median_add': np.median(self.add_errors) if self.add_errors else None,
            'add_auc': compute_add_auc(np.array(self.add_errors)) if self.add_errors else None,
            'add_success_rate': compute_success_rate(np.array(self.add_errors), add_threshold) if self.add_errors else None,
            'mean_reproj': np.mean(self.reproj_errors) if self.reproj_errors else None,
            'mean_rot': np.mean(self.rot_errors) if self.rot_errors else None,
            'mean_trans': np.mean(self.trans_errors) if self.trans_errors else None,
            'num_frames': len(self.add_errors)
        }
        
        return summary
    
    def get_errors(self) -> dict:
        """
        Get raw error arrays.
        
        Returns:
            Dictionary of error arrays
        """
        return {
            'add': np.array(self.add_errors),
            'reproj': np.array(self.reproj_errors) if self.reproj_errors else None,
            'rot': np.array(self.rot_errors),
            'trans': np.array(self.trans_errors)
        }

