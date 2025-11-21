"""
Unit tests for data loading functionality.
"""

import numpy as np
import sys
from pathlib import Path
import tempfile
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from instantpose.utils import (
    get_default_linemod_intrinsics,
    depth_to_points,
    project_points,
    compose_rt,
    decompose_rt
)


def test_default_intrinsics():
    """Test default LINEMOD intrinsics."""
    print("Test 1: Default intrinsics")
    
    K = get_default_linemod_intrinsics()
    
    assert K.shape == (3, 3), "K should be 3x3"
    assert K[2, 2] == 1.0, "K[2,2] should be 1.0"
    assert K[0, 0] > 0, "fx should be positive"
    assert K[1, 1] > 0, "fy should be positive"
    
    print("  [OK] Passed")


def test_depth_to_points():
    """Test depth to point cloud conversion."""
    print("Test 2: Depth to points conversion")
    
    # Create simple depth map
    depth = np.ones((10, 10)) * 1.0  # All points at 1m
    K = np.array([
        [100.0, 0.0, 5.0],
        [0.0, 100.0, 5.0],
        [0.0, 0.0, 1.0]
    ])
    
    points = depth_to_points(depth, K)
    
    assert points.shape[1] == 3, "Points should be 3D"
    assert len(points) == 100, "Should have 100 points"
    
    # Check center point (u=5, v=5) should be at (0, 0, 1)
    # Find point closest to origin in xy
    xy_norms = np.linalg.norm(points[:, :2], axis=1)
    center_idx = np.argmin(xy_norms)
    center_point = points[center_idx]
    
    assert np.allclose(center_point[:2], [0, 0], atol=0.01), "Center should be at (0, 0)"
    assert np.allclose(center_point[2], 1.0, atol=0.01), "Depth should be 1.0"
    
    print("  [OK] Passed")


def test_projection():
    """Test 3D to 2D projection."""
    print("Test 3: 3D to 2D projection")
    
    K = np.array([
        [100.0, 0.0, 50.0],
        [0.0, 100.0, 50.0],
        [0.0, 0.0, 1.0]
    ])
    R = np.eye(3)
    t = np.zeros(3)
    
    # Project point at (0, 0, 1) should map to principal point
    pts_3d = np.array([[0.0, 0.0, 1.0]])
    pts_2d = project_points(pts_3d, K, R, t)
    
    assert np.allclose(pts_2d[0], [50.0, 50.0], atol=1e-6), "Should project to principal point"
    
    # Project point at (1, 0, 1) should map to (150, 50)
    pts_3d = np.array([[1.0, 0.0, 1.0]])
    pts_2d = project_points(pts_3d, K, R, t)
    
    assert np.allclose(pts_2d[0], [150.0, 50.0], atol=1e-6), "Should project correctly"
    
    print("  [OK] Passed")


def test_compose_decompose():
    """Test RT composition and decomposition."""
    print("Test 4: RT composition/decomposition")
    
    R = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ], dtype=float)
    t = np.array([1, 2, 3], dtype=float)
    
    # Compose
    T = compose_rt(R, t)
    
    assert T.shape == (4, 4), "T should be 4x4"
    assert T[3, 3] == 1.0, "T[3,3] should be 1.0"
    
    # Decompose
    R_out, t_out = decompose_rt(T)
    
    assert np.allclose(R_out, R), "R should match"
    assert np.allclose(t_out, t), "t should match"
    
    print("  [OK] Passed")


def test_projection_reprojection():
    """Test projection and back-projection consistency."""
    print("Test 5: Projection/back-projection round-trip")
    
    K = np.array([
        [200.0, 0.0, 160.0],
        [0.0, 200.0, 120.0],
        [0.0, 0.0, 1.0]
    ])
    R = np.eye(3)
    t = np.zeros(3)
    
    # Create 3D points
    pts_3d = np.random.randn(50, 3)
    pts_3d[:, 2] = np.abs(pts_3d[:, 2]) + 1.0  # Ensure positive depth
    
    # Project to 2D
    pts_2d = project_points(pts_3d, K, R, t)
    
    # Create depth map (simulate)
    H, W = 240, 320
    depth = np.zeros((H, W))
    for pt_3d, pt_2d in zip(pts_3d, pts_2d):
        u, v = int(pt_2d[0]), int(pt_2d[1])
        if 0 <= u < W and 0 <= v < H:
            depth[v, u] = pt_3d[2]
    
    # Back-project
    pts_3d_back = depth_to_points(depth, K)
    
    # Should have recovered some points
    assert len(pts_3d_back) > 0, "Should recover some points"
    
    print("  [OK] Passed")


def test_mask_filtering():
    """Test depth to points with mask."""
    print("Test 6: Depth to points with mask")
    
    depth = np.ones((10, 10))
    K = np.eye(3)
    K[0, 0] = K[1, 1] = 100.0
    K[0, 2] = K[1, 2] = 5.0
    
    # Create mask (only center 4 points)
    mask = np.zeros((10, 10))
    mask[4:6, 4:6] = 1
    
    points = depth_to_points(depth, K, mask)
    
    # Should only have 4 points
    assert len(points) == 4, f"Should have 4 points, got {len(points)}"
    
    print("  [OK] Passed")


def run_all_tests():
    """Run all tests."""
    print("=" * 80)
    print("Running Data Loading Tests")
    print("=" * 80)
    
    test_default_intrinsics()
    test_depth_to_points()
    test_projection()
    test_compose_decompose()
    test_projection_reprojection()
    test_mask_filtering()
    
    print("\n" + "=" * 80)
    print("All tests passed! [OK]")
    print("=" * 80)


if __name__ == '__main__':
    run_all_tests()

