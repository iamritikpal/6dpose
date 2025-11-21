"""
Unit tests for Umeyama alignment and RANSAC.
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from instantpose.refine import (
    umeyama_alignment,
    ransac_umeyama,
    transform_points,
    invert_similarity_transform
)


def test_umeyama_identity():
    """Test Umeyama with identity transformation."""
    print("Test 1: Identity transformation")
    
    # Create random points
    np.random.seed(42)
    X_src = np.random.randn(100, 3)
    X_dst = X_src.copy()
    
    # Estimate transformation
    R, t, scale = umeyama_alignment(X_src, X_dst)
    
    # Check results
    assert np.allclose(R, np.eye(3), atol=1e-6), "R should be identity"
    assert np.allclose(t, np.zeros(3), atol=1e-6), "t should be zero"
    assert np.allclose(scale, 1.0, atol=1e-6), "scale should be 1.0"
    
    print("  [OK] Passed")


def test_umeyama_translation():
    """Test Umeyama with pure translation."""
    print("Test 2: Pure translation")
    
    np.random.seed(42)
    X_src = np.random.randn(100, 3)
    t_true = np.array([1.0, 2.0, 3.0])
    X_dst = X_src + t_true
    
    R, t, scale = umeyama_alignment(X_src, X_dst)
    
    assert np.allclose(R, np.eye(3), atol=1e-6), "R should be identity"
    assert np.allclose(t, t_true, atol=1e-6), "t should match"
    assert np.allclose(scale, 1.0, atol=1e-6), "scale should be 1.0"
    
    print("  [OK] Passed")


def test_umeyama_scale():
    """Test Umeyama with scaling."""
    print("Test 3: Scaling transformation")
    
    np.random.seed(42)
    X_src = np.random.randn(100, 3)
    scale_true = 2.5
    X_dst = X_src * scale_true
    
    R, t, scale = umeyama_alignment(X_src, X_dst)
    
    assert np.allclose(R, np.eye(3), atol=1e-6), "R should be identity"
    assert np.allclose(t, np.zeros(3), atol=1e-6), "t should be zero"
    assert np.allclose(scale, scale_true, atol=1e-3), f"scale should be {scale_true}"
    
    print("  [OK] Passed")


def test_umeyama_full():
    """Test Umeyama with full similarity transformation."""
    print("Test 4: Full similarity transformation")
    
    np.random.seed(42)
    X_src = np.random.randn(100, 3)
    
    # Create random rotation
    from scipy.spatial.transform import Rotation
    R_true = Rotation.random().as_matrix()
    t_true = np.array([1.0, 2.0, 3.0])
    scale_true = 1.5
    
    X_dst = scale_true * (R_true @ X_src.T).T + t_true
    
    R, t, scale = umeyama_alignment(X_src, X_dst)
    
    # Transform points and check
    X_transformed = scale * (R @ X_src.T).T + t
    
    assert np.allclose(X_transformed, X_dst, atol=1e-6), "Transformed points should match"
    assert np.allclose(scale, scale_true, atol=1e-3), f"scale should be {scale_true}"
    
    print("  [OK] Passed")


def test_ransac_no_outliers():
    """Test RANSAC with no outliers."""
    print("Test 5: RANSAC with no outliers")
    
    np.random.seed(42)
    X_src = np.random.randn(50, 3)
    
    from scipy.spatial.transform import Rotation
    R_true = Rotation.random().as_matrix()
    t_true = np.array([0.5, 1.0, 1.5])
    scale_true = 1.2
    
    X_dst = scale_true * (R_true @ X_src.T).T + t_true
    
    R, t, scale, inliers = ransac_umeyama(
        X_src, X_dst,
        n_iters=100,
        inlier_thresh=0.01
    )
    
    # All points should be inliers
    assert np.sum(inliers) >= 45, f"Most points should be inliers, got {np.sum(inliers)}/50"
    
    # Transform and check
    X_transformed = scale * (R @ X_src.T).T + t
    error = np.mean(np.linalg.norm(X_transformed - X_dst, axis=1))
    assert error < 0.01, f"Average error should be small, got {error}"
    
    print("  [OK] Passed")


def test_ransac_with_outliers():
    """Test RANSAC with outliers."""
    print("Test 6: RANSAC with outliers")
    
    np.random.seed(42)
    n_inliers = 40
    n_outliers = 10
    
    X_src = np.random.randn(n_inliers + n_outliers, 3)
    
    from scipy.spatial.transform import Rotation
    R_true = Rotation.random().as_matrix()
    t_true = np.array([0.5, 1.0, 1.5])
    scale_true = 1.2
    
    X_dst = scale_true * (R_true @ X_src.T).T + t_true
    
    # Add outliers
    X_dst[-n_outliers:] += np.random.randn(n_outliers, 3) * 0.5
    
    R, t, scale, inliers = ransac_umeyama(
        X_src, X_dst,
        n_iters=500,
        inlier_thresh=0.01
    )
    
    # Should detect inliers
    assert np.sum(inliers) >= n_inliers * 0.8, f"Should detect most inliers, got {np.sum(inliers)}"
    
    print("  [OK] Passed")


def test_transform_inversion():
    """Test similarity transform inversion."""
    print("Test 7: Transform inversion")
    
    np.random.seed(42)
    from scipy.spatial.transform import Rotation
    
    R = Rotation.random().as_matrix()
    t = np.array([1.0, 2.0, 3.0])
    scale = 2.0
    
    R_inv, t_inv, scale_inv = invert_similarity_transform(R, t, scale)
    
    # Test round-trip
    X = np.random.randn(10, 3)
    X_transformed = scale * (R @ X.T).T + t
    X_back = scale_inv * (R_inv @ X_transformed.T).T + t_inv
    
    assert np.allclose(X, X_back, atol=1e-6), "Round-trip should recover original points"
    
    print("  [OK] Passed")


def run_all_tests():
    """Run all tests."""
    print("=" * 80)
    print("Running Umeyama & RANSAC Tests")
    print("=" * 80)
    
    test_umeyama_identity()
    test_umeyama_translation()
    test_umeyama_scale()
    test_umeyama_full()
    test_ransac_no_outliers()
    test_ransac_with_outliers()
    test_transform_inversion()
    
    print("\n" + "=" * 80)
    print("All tests passed! [OK]")
    print("=" * 80)


if __name__ == '__main__':
    run_all_tests()

