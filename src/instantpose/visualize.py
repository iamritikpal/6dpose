"""
Visualization utilities for 6D pose estimation.
Includes pose overlay, error plots, and result visualization.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .utils import project_points


def draw_axes(
    img: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    scale: float = 0.2,  # Increased from 0.1
    thickness: int = 4   # Increased from 3
) -> np.ndarray:
    """
    Draw 3D coordinate axes on image.
    
    Args:
        img: Input image [H, W, 3]
        K: Camera intrinsics [3, 3]
        R: Rotation matrix [3, 3]
        t: Translation vector [3]
        scale: Axis length
        thickness: Line thickness
        
    Returns:
        Image with axes drawn
    """
    img = img.copy()
    H, W = img.shape[:2]
    
    # Define axis endpoints in object coordinates
    axes_pts = np.array([
        [0, 0, 0],
        [scale, 0, 0],  # X-axis (red)
        [0, scale, 0],  # Y-axis (green)
        [0, 0, scale]   # Z-axis (blue)
    ], dtype=np.float32)
    
    # Project to image
    pts_2d = project_points(axes_pts, K, R, t)
    
    origin = tuple(pts_2d[0].astype(int))
    
    # Check if origin is within bounds (with some margin)
    if not (0 <= origin[0] < W and 0 <= origin[1] < H):
        print(f"  Warning: Pose origin {origin} is outside image bounds ({W}x{H})")
        
    x_end = tuple(pts_2d[1].astype(int))
    y_end = tuple(pts_2d[2].astype(int))
    z_end = tuple(pts_2d[3].astype(int))
    
    # Draw axes
    cv2.line(img, origin, x_end, (0, 0, 255), thickness)  # Red X
    cv2.line(img, origin, y_end, (0, 255, 0), thickness)  # Green Y
    cv2.line(img, origin, z_end, (255, 0, 0), thickness)  # Blue Z
    
    return img


def draw_bounding_box(
    img: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    bbox_3d: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 3  # Increased from 2
) -> np.ndarray:
    """
    Draw 3D bounding box on image.
    
    Args:
        img: Input image [H, W, 3]
        K: Camera intrinsics [3, 3]
        R: Rotation matrix [3, 3]
        t: Translation vector [3]
        bbox_3d: 3D bounding box corners [8, 3]
        color: Line color (B, G, R)
        thickness: Line thickness
        
    Returns:
        Image with bounding box drawn
    """
    img = img.copy()
    
    # Project corners to image
    pts_2d = project_points(bbox_3d, K, R, t).astype(int)
    
    # Define edges of box (indices into corners)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
        (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
    ]
    
    # Draw edges
    for i, j in edges:
        pt1 = tuple(pts_2d[i])
        pt2 = tuple(pts_2d[j])
        cv2.line(img, pt1, pt2, color, thickness)
    
    return img


def create_bbox_from_points(points: np.ndarray) -> np.ndarray:
    """
    Create 3D bounding box from point cloud.
    
    Args:
        points: 3D points [N, 3]
        
    Returns:
        Bounding box corners [8, 3]
    """
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    
    # Create 8 corners
    bbox = np.array([
        [min_coords[0], min_coords[1], min_coords[2]],
        [max_coords[0], min_coords[1], min_coords[2]],
        [max_coords[0], max_coords[1], min_coords[2]],
        [min_coords[0], max_coords[1], min_coords[2]],
        [min_coords[0], min_coords[1], max_coords[2]],
        [max_coords[0], min_coords[1], max_coords[2]],
        [max_coords[0], max_coords[1], max_coords[2]],
        [min_coords[0], max_coords[1], max_coords[2]]
    ])
    
    return bbox


def visualize_matches(
    img_query: np.ndarray,
    img_ref: np.ndarray,
    pts_query: np.ndarray,
    pts_ref: np.ndarray,
    max_display: int = 50
) -> np.ndarray:
    """
    Visualize feature matches between two images.
    
    Args:
        img_query: Query image [H, W, 3]
        img_ref: Reference image [H, W, 3]
        pts_query: Query keypoints [N, 2]
        pts_ref: Reference keypoints [N, 2]
        max_display: Maximum number of matches to display
        
    Returns:
        Concatenated visualization image
    """
    h1, w1 = img_query.shape[:2]
    h2, w2 = img_ref.shape[:2]
    
    # Create side-by-side image
    h = max(h1, h2)
    w = w1 + w2
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    vis[:h1, :w1] = img_query
    vis[:h2, w1:w1+w2] = img_ref
    
    # Sample matches if too many
    n_matches = len(pts_query)
    if n_matches > max_display:
        indices = np.random.choice(n_matches, max_display, replace=False)
        pts_query = pts_query[indices]
        pts_ref = pts_ref[indices]
    
    # Draw matches
    for pt_q, pt_r in zip(pts_query, pts_ref):
        pt_q = tuple(pt_q.astype(int))
        pt_r = tuple((pt_r + [w1, 0]).astype(int))
        
        # Random color for each match
        color = tuple(np.random.randint(0, 255, 3).tolist())
        
        cv2.circle(vis, pt_q, 3, color, -1)
        cv2.circle(vis, pt_r, 3, color, -1)
        cv2.line(vis, pt_q, pt_r, color, 1)
    
    return vis


def plot_error_histogram(
    errors: np.ndarray,
    title: str,
    xlabel: str,
    output_path: Optional[str] = None,
    bins: int = 50
) -> None:
    """
    Plot histogram of errors.
    
    Args:
        errors: Array of errors
        title: Plot title
        xlabel: X-axis label
        output_path: Output file path (optional)
        bins: Number of bins
    """
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=bins, edgecolor='black', alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    mean_err = np.mean(errors)
    median_err = np.median(errors)
    plt.axvline(mean_err, color='r', linestyle='--', label=f'Mean: {mean_err:.4f}')
    plt.axvline(median_err, color='g', linestyle='--', label=f'Median: {median_err:.4f}')
    plt.legend()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved histogram to {output_path}")
    plt.close()


def plot_success_curve(
    errors: np.ndarray,
    max_threshold: float,
    title: str,
    xlabel: str,
    output_path: Optional[str] = None,
    n_points: int = 100
) -> None:
    """
    Plot success rate curve (accuracy vs threshold).
    
    Args:
        errors: Array of errors
        max_threshold: Maximum threshold
        title: Plot title
        xlabel: X-axis label
        output_path: Output file path (optional)
        n_points: Number of points in curve
    """
    thresholds = np.linspace(0, max_threshold, n_points)
    success_rates = [np.mean(errors < t) for t in thresholds]
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, success_rates, linewidth=2)
    plt.xlabel(xlabel)
    plt.ylabel('Success Rate')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])
    
    # Compute AUC
    auc = np.trapz(success_rates, thresholds) / max_threshold
    plt.text(0.6 * max_threshold, 0.2, f'AUC: {auc:.4f}', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved success curve to {output_path}")
    plt.close()


def save_frame_visualization(
    img: np.ndarray,
    K: np.ndarray,
    R_pred: Optional[np.ndarray],
    t_pred: Optional[np.ndarray],
    R_gt: Optional[np.ndarray] = None,
    t_gt: Optional[np.ndarray] = None,
    output_path: Optional[str] = None,
    model_pts: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Save frame visualization with pose overlay.
    
    Args:
        img: Input image [H, W, 3]
        K: Camera intrinsics [3, 3]
        R_pred: Predicted rotation [3, 3]
        t_pred: Predicted translation [3]
        R_gt: Ground truth rotation [3, 3] (optional)
        t_gt: Ground truth translation [3] (optional)
        output_path: Output file path (optional)
        model_pts: Model points for bounding box (optional)
        
    Returns:
        Visualization image
    """
    vis = img.copy()
    
    # Draw predicted pose (green)
    if R_pred is not None and t_pred is not None:
        vis = draw_axes(vis, K, R_pred, t_pred, scale=0.1, thickness=3)
        
        if model_pts is not None:
            bbox = create_bbox_from_points(model_pts)
            vis = draw_bounding_box(vis, K, R_pred, t_pred, bbox, 
                                   color=(0, 255, 0), thickness=2)
    
    # Draw ground truth pose (blue)
    if R_gt is not None and t_gt is not None:
        vis = draw_axes(vis, K, R_gt, t_gt, scale=0.1, thickness=2)
        
        if model_pts is not None:
            bbox = create_bbox_from_points(model_pts)
            vis = draw_bounding_box(vis, K, R_gt, t_gt, bbox,
                                   color=(255, 0, 0), thickness=2)
    
    # Add legend
    legend_height = 80
    legend = np.ones((legend_height, vis.shape[1], 3), dtype=np.uint8) * 255
    if R_pred is not None:
        cv2.putText(legend, 'Predicted (Green)', (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    if R_gt is not None:
        cv2.putText(legend, 'Ground Truth (Blue)', (10, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    vis_with_legend = np.vstack([vis, legend])
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, cv2.cvtColor(vis_with_legend, cv2.COLOR_RGB2BGR))
    
    return vis_with_legend


def create_summary_figure(
    errors_dict: dict,
    output_path: str
) -> None:
    """
    Create summary figure with multiple error plots.
    
    Args:
        errors_dict: Dictionary of error arrays
        output_path: Output file path
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # ADD error histogram
    if errors_dict.get('add') is not None:
        ax = axes[0, 0]
        add_errors = errors_dict['add']
        ax.hist(add_errors, bins=50, edgecolor='black', alpha=0.7)
        ax.set_xlabel('ADD Error (m)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'ADD Error Distribution (mean: {np.mean(add_errors):.4f}m)')
        ax.grid(True, alpha=0.3)
    
    # ADD success curve
    if errors_dict.get('add') is not None:
        ax = axes[0, 1]
        add_errors = errors_dict['add']
        max_thresh = min(0.1, np.percentile(add_errors, 95))
        thresholds = np.linspace(0, max_thresh, 100)
        success_rates = [np.mean(add_errors < t) for t in thresholds]
        ax.plot(thresholds, success_rates, linewidth=2)
        ax.set_xlabel('ADD Threshold (m)')
        ax.set_ylabel('Success Rate')
        ax.set_title('ADD Success Curve')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
    
    # Rotation error
    if errors_dict.get('rot') is not None:
        ax = axes[1, 0]
        rot_errors = errors_dict['rot']
        ax.hist(rot_errors, bins=50, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Rotation Error (degrees)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Rotation Error (mean: {np.mean(rot_errors):.2f}°)')
        ax.grid(True, alpha=0.3)
    
    # Translation error
    if errors_dict.get('trans') is not None:
        ax = axes[1, 1]
        trans_errors = errors_dict['trans']
        ax.hist(trans_errors, bins=50, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Translation Error (m)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Translation Error (mean: {np.mean(trans_errors):.4f}m)')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved summary figure to {output_path}")
    plt.close()

