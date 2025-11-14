"""
Main entry point for InstantPose 6D pose estimation pipeline.
Handles CLI, configuration, and orchestrates the full pipeline.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml
from tqdm import tqdm

from instantpose.data import load_dataset, read_rgb, read_depth, read_pose
from instantpose.render import TemplateRenderer
from instantpose.features import FeatureExtractor, upsample_coords
from instantpose.refine import estimate_pose_from_correspondences
from instantpose.eval import PoseEvaluator
from instantpose.visualize import (
    save_frame_visualization,
    create_summary_figure,
    plot_error_histogram,
    plot_success_curve
)
from instantpose.utils import (
    set_seed,
    save_json,
    ensure_directory,
    depth_to_points
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='InstantPose: Training-free 6D Pose Estimation'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    
    # Allow overriding config values from command line
    parser.add_argument(
        'overrides',
        nargs='*',
        help='Config overrides in format: KEY.SUBKEY value'
    )
    
    return parser.parse_args()


def load_config(config_path: str, overrides: List[str]) -> Dict:
    """
    Load configuration from YAML file and apply command-line overrides.
    
    Args:
        config_path: Path to config file
        overrides: List of override strings
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Apply overrides
    i = 0
    while i < len(overrides):
        if i + 1 >= len(overrides):
            print(f"Warning: Ignoring unpaired override: {overrides[i]}")
            break
        
        key_path = overrides[i]
        value = overrides[i + 1]
        
        # Parse nested keys (e.g., "DATA.DATASET")
        keys = key_path.split('.')
        
        # Navigate to the target
        target = config
        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]
        
        # Set the value (try to infer type)
        final_key = keys[-1]
        try:
            # Try to parse as number
            if '.' in value:
                target[final_key] = float(value)
            else:
                target[final_key] = int(value)
        except ValueError:
            # Keep as string
            target[final_key] = value
        
        i += 2
    
    return config


def find_best_template(
    feat_query: np.ndarray,
    feats_templates: List[np.ndarray],
    feature_extractor: FeatureExtractor
) -> int:
    """
    Find best matching template using SNN matching.
    
    Args:
        feat_query: Query features
        feats_templates: List of template features
        feature_extractor: Feature extractor instance
        
    Returns:
        Index of best template
    """
    best_score = -1
    best_idx = 0
    
    for idx, feat_template in enumerate(feats_templates):
        # Match features
        query_coords, ref_coords, _ = feature_extractor.match(
            feat_query, feat_template
        )
        
        # Score by number of matches
        score = len(query_coords)
        
        if score > best_score:
            best_score = score
            best_idx = idx
    
    return best_idx


def estimate_pose_for_frame(
    rgb_query: np.ndarray,
    depth_query: np.ndarray,
    K_query: np.ndarray,
    feature_extractor: FeatureExtractor,
    templates: Dict,
    feats_templates: List,
    config: Dict
) -> Dict:
    """
    Estimate pose for a single frame.
    
    Args:
        rgb_query: Query RGB image
        depth_query: Query depth map
        K_query: Query camera intrinsics
        feature_extractor: Feature extractor instance
        templates: Template dictionary
        feats_templates: Precomputed template features
        config: Configuration dictionary
        
    Returns:
        Dictionary with pose estimation results
    """
    # Extract query features
    feat_query = feature_extractor.extract(rgb_query)
    
    # Find best template
    print("  Finding best template match...")
    best_template_idx = find_best_template(feat_query, feats_templates, feature_extractor)
    print(f"  Best template: {best_template_idx}")
    
    # Get best template data
    feat_template = feats_templates[best_template_idx]
    rgb_template = templates['rgb'][best_template_idx]
    depth_template = templates['depth'][best_template_idx]
    K_template = templates['K']
    extrinsic_template = templates['extrinsics'][best_template_idx]
    
    # Match features
    print("  Matching features...")
    query_coords_feat, template_coords_feat, (H_q, W_q, H_t, W_t) = feature_extractor.match(
        feat_query, feat_template
    )
    
    print(f"  Found {len(query_coords_feat)} cycle-consistent matches")
    
    if len(query_coords_feat) < 3:
        print("  Warning: Too few matches, skipping frame")
        return None
    
    # Upsample coordinates to image space
    query_coords_img = upsample_coords(
        query_coords_feat,
        (H_q, W_q),
        rgb_query.shape[:2]
    )
    template_coords_img = upsample_coords(
        template_coords_feat,
        (H_t, W_t),
        rgb_template.shape[:2]
    )
    
    # Lift to 3D using depth
    print("  Lifting to 3D...")
    pts_query_3d = []
    pts_template_3d = []
    
    for (u_q, v_q), (u_t, v_t) in zip(query_coords_img, template_coords_img):
        u_q, v_q = int(u_q), int(v_q)
        u_t, v_t = int(u_t), int(v_t)
        
        # Check bounds
        if not (0 <= v_q < depth_query.shape[0] and 0 <= u_q < depth_query.shape[1]):
            continue
        if not (0 <= v_t < depth_template.shape[0] and 0 <= u_t < depth_template.shape[1]):
            continue
        
        # Get depths
        z_q = depth_query[v_q, u_q]
        z_t = depth_template[v_t, u_t]
        
        # Check valid depths
        if z_q <= 0 or z_t <= 0:
            continue
        
        # Backproject query point
        x_q = (u_q - K_query[0, 2]) * z_q / K_query[0, 0]
        y_q = (v_q - K_query[1, 2]) * z_q / K_query[1, 1]
        pt_q = np.array([x_q, y_q, z_q])
        
        # Backproject template point
        x_t = (u_t - K_template[0, 2]) * z_t / K_template[0, 0]
        y_t = (v_t - K_template[1, 2]) * z_t / K_template[1, 1]
        pt_t_cam = np.array([x_t, y_t, z_t])
        
        # Transform template point to world coordinates (then to object frame)
        # Template is rendered in normalized object coordinates
        R_cam = extrinsic_template[:3, :3]
        t_cam = extrinsic_template[:3, 3]
        pt_t_world = R_cam.T @ (pt_t_cam - t_cam)
        
        pts_query_3d.append(pt_q)
        pts_template_3d.append(pt_t_world)
    
    pts_query_3d = np.array(pts_query_3d)
    pts_template_3d = np.array(pts_template_3d)
    
    print(f"  Valid 3D correspondences: {len(pts_query_3d)}")
    
    if len(pts_query_3d) < 3:
        print("  Warning: Too few 3D correspondences, skipping frame")
        return None
    
    # Estimate pose with RANSAC
    print("  Estimating pose with RANSAC...")
    R, t, scale, inliers = estimate_pose_from_correspondences(
        pts_template_3d,
        pts_query_3d,
        config['REFINE']
    )
    
    result = {
        'R': R,
        't': t,
        'scale': scale,
        'inliers': inliers,
        'n_matches': len(pts_query_3d),
        'n_inliers': np.sum(inliers),
        'best_template_idx': best_template_idx
    }
    
    return result


def run_pipeline(config: Dict) -> None:
    """
    Run the complete InstantPose pipeline.
    
    Args:
        config: Configuration dictionary
    """
    print("=" * 80)
    print("InstantPose: Training-free 6D Pose Estimation")
    print("=" * 80)
    
    # Set seed
    set_seed(config['SEED'])
    
    # Create output directory
    output_dir = Path(config['PATHS']['OUTPUT_DIR'])
    ensure_directory(output_dir)
    ensure_directory(output_dir / 'vis')
    ensure_directory(output_dir / 'curves')
    
    # Load dataset
    print("\n[1/6] Loading dataset...")
    dataset = load_dataset(config)
    print(f"  Dataset: {dataset['dataset']}")
    print(f"  Object: {dataset['object_id']}")
    print(f"  Frames: {len(dataset['frames'])}")
    
    # Check mesh exists
    mesh_path = Path(config['PATHS']['MESH'])
    if not mesh_path.exists():
        print(f"\nError: Mesh not found at {mesh_path}")
        print("Please place your reconstructed mesh in the assets/recon_meshes/ directory")
        sys.exit(1)
    
    # Initialize components
    print("\n[2/6] Initializing components...")
    feature_extractor = FeatureExtractor(config)
    template_renderer = TemplateRenderer(config)
    
    # Generate templates
    print("\n[3/6] Generating templates...")
    templates = template_renderer.generate_templates(str(mesh_path))
    
    # Extract template features
    print("\n[4/6] Extracting template features...")
    feats_templates = []
    for rgb_template in tqdm(templates['rgb'], desc="Extracting features"):
        feat = feature_extractor.extract(rgb_template)
        feats_templates.append(feat)
    
    # Initialize evaluator if in eval mode
    evaluator = None
    if config['MODE'] == 'eval':
        symmetric = dataset['object_id'] in config.get('EVAL', {}).get('SYMMETRIC_OBJECTS', [])
        evaluator = PoseEvaluator(
            mesh_path=str(mesh_path),
            symmetric=symmetric
        )
        print(f"\n[5/6] Running evaluation (symmetric={symmetric})...")
    else:
        print("\n[5/6] Running demo mode...")
    
    # Process frames
    results = []
    for frame_idx, frame_data in enumerate(tqdm(dataset['frames'], desc="Processing frames")):
        print(f"\nFrame {frame_idx + 1}/{len(dataset['frames'])}: {frame_data['stem']}")
        
        # Load frame data
        rgb_query = read_rgb(frame_data['rgb'])
        
        if frame_data['depth'] is None:
            print("  Warning: No depth data, skipping frame")
            continue
        
        depth_query = read_depth(frame_data['depth'])
        K_query = dataset['K']
        
        # Load GT pose if available
        R_gt, t_gt = None, None
        if frame_data['pose'] is not None:
            try:
                R_gt, t_gt = read_pose(frame_data['pose'])
            except Exception as e:
                print(f"  Warning: Failed to load pose: {e}")
        
        # Estimate pose
        try:
            result = estimate_pose_for_frame(
                rgb_query,
                depth_query,
                K_query,
                feature_extractor,
                templates,
                feats_templates,
                config
            )
        except Exception as e:
            print(f"  Error during pose estimation: {e}")
            import traceback
            traceback.print_exc()
            result = None
        
        if result is None:
            continue
        
        R_pred, t_pred = result['R'], result['t']
        
        # Evaluate if GT available
        if evaluator is not None and R_gt is not None:
            metrics = evaluator.evaluate_frame(R_gt, t_gt, R_pred, t_pred, K_query)
            print(f"  ADD: {metrics['add']:.4f}m, Rot: {metrics['rot_deg']:.2f}°, Trans: {metrics['trans_m']:.4f}m")
        
        # Visualize
        vis_path = output_dir / 'vis' / f"{frame_data['stem']}.png"
        save_frame_visualization(
            rgb_query,
            K_query,
            R_pred, t_pred,
            R_gt, t_gt,
            output_path=str(vis_path),
            model_pts=evaluator.model_pts if evaluator else None
        )
        
        # Store result
        result['frame'] = frame_data['stem']
        results.append(result)
    
    print(f"\n[6/6] Saving results...")
    
    # Save metrics
    if evaluator is not None:
        summary = evaluator.get_summary()
        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)
        for key, value in summary.items():
            if value is not None:
                print(f"  {key}: {value}")
        
        save_json(summary, output_dir / 'metrics.json')
        print(f"\nSaved metrics to {output_dir / 'metrics.json'}")
        
        # Generate plots
        errors = evaluator.get_errors()
        
        # Summary figure
        create_summary_figure(errors, str(output_dir / 'curves' / 'summary.png'))
        
        # Individual plots
        if errors['add'] is not None and len(errors['add']) > 0:
            plot_error_histogram(
                errors['add'],
                'ADD Error Distribution',
                'ADD Error (m)',
                str(output_dir / 'curves' / 'add_histogram.png')
            )
            plot_success_curve(
                errors['add'],
                0.1,
                'ADD Success Curve',
                'ADD Threshold (m)',
                str(output_dir / 'curves' / 'add_success.png')
            )
    
    print("\n" + "=" * 80)
    print(f"Pipeline complete! Results saved to {output_dir}")
    print("=" * 80)


def main():
    """Main entry point."""
    args = parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config, args.overrides)
    except FileNotFoundError:
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)
    
    # Run pipeline
    try:
        run_pipeline(config)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

