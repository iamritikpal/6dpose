"""
Mesh rendering utilities using pyrender for template generation.
Renders RGB and depth images from multiple viewpoints.
"""

import os
import sys

# Set environment for offscreen rendering BEFORE importing pyrender
# On Linux, use EGL for headless rendering
# On Windows, don't set PYOPENGL_PLATFORM - let pyrender use default OpenGL context
if sys.platform != 'win32':
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
# On Windows, ensure we're not trying to use EGL
elif 'PYOPENGL_PLATFORM' in os.environ:
    del os.environ['PYOPENGL_PLATFORM']

from typing import List, Tuple

import numpy as np
import trimesh
import pyrender


def sample_camera_poses(
    n_views: int,
    azimuth_range: Tuple[float, float] = (0, 360),
    elevation_range: Tuple[float, float] = (0, 60),
    distance: float = 0.7
) -> List[np.ndarray]:
    """
    Sample camera poses on a sphere around the object.
    
    Args:
        n_views: Number of viewpoints to sample
        azimuth_range: Azimuth angle range in degrees (min, max)
        elevation_range: Elevation angle range in degrees (min, max)
        distance: Camera distance from origin
        
    Returns:
        List of 4x4 camera-to-world transformation matrices
    """
    poses = []
    
    # Create grid of viewpoints
    n_az = int(np.sqrt(n_views * (azimuth_range[1] - azimuth_range[0]) / 
                       (elevation_range[1] - elevation_range[0])))
    n_el = int(n_views / n_az)
    
    azimuths = np.linspace(azimuth_range[0], azimuth_range[1], n_az, endpoint=False)
    elevations = np.linspace(elevation_range[0], elevation_range[1], n_el)
    
    for el in elevations:
        for az in azimuths:
            # Convert to radians
            az_rad = np.deg2rad(az)
            el_rad = np.deg2rad(el)
            
            # Spherical to Cartesian (camera position)
            x = distance * np.cos(el_rad) * np.cos(az_rad)
            y = distance * np.cos(el_rad) * np.sin(az_rad)
            z = distance * np.sin(el_rad)
            
            camera_pos = np.array([x, y, z])
            
            # Look-at matrix (camera looks at origin)
            forward = -camera_pos / np.linalg.norm(camera_pos)
            right = np.cross([0, 0, 1], forward)
            if np.linalg.norm(right) < 1e-6:
                # Handle singularity when looking straight down/up
                right = np.array([1, 0, 0])
            right = right / np.linalg.norm(right)
            up = np.cross(forward, right)
            
            # Construct camera-to-world matrix
            T = np.eye(4)
            T[:3, 0] = right
            T[:3, 1] = up
            T[:3, 2] = forward
            T[:3, 3] = camera_pos
            
            poses.append(T)
    
    return poses[:n_views]  # Trim to exact number


def render_mesh_views(
    mesh_path: str,
    height: int,
    width: int,
    fov_deg: float,
    camera_poses: List[np.ndarray],
    ambient_light: float = 0.5
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Render RGB and depth images of a mesh from multiple viewpoints.
    
    Args:
        mesh_path: Path to mesh file (.obj or .ply)
        height: Image height
        width: Image width
        fov_deg: Vertical field of view in degrees
        camera_poses: List of 4x4 camera-to-world matrices
        ambient_light: Ambient light intensity
        
    Returns:
        Tuple of (rgb_images, depth_images, extrinsics)
        - rgb_images: List of RGB images [H, W, 3] uint8
        - depth_images: List of depth maps [H, W] float32 in meters
        - extrinsics: List of 4x4 world-to-camera matrices
    """
    # Load mesh
    mesh = trimesh.load(mesh_path)
    
    # Normalize mesh to unit sphere
    mesh.vertices -= mesh.centroid
    scale = np.max(np.linalg.norm(mesh.vertices, axis=1))
    mesh.vertices /= scale
    
    # Create pyrender mesh
    mesh_pr = pyrender.Mesh.from_trimesh(mesh)
    
    # Create scene
    scene = pyrender.Scene(ambient_light=[ambient_light] * 3)
    scene.add(mesh_pr)
    
    # Create camera
    camera = pyrender.PerspectiveCamera(yfov=np.deg2rad(fov_deg), aspectRatio=width/height)
    
    # Create renderer
    try:
        renderer = pyrender.OffscreenRenderer(width, height)
    except Exception as e:
        if sys.platform == 'win32':
            print("\n" + "="*80)
            print("ERROR: Unable to create offscreen renderer on Windows")
            print("="*80)
            print("\nThis is likely because pyrender needs OpenGL support.")
            print("\nPossible solutions:")
            print("1. Make sure your GPU drivers are up to date")
            print("2. Try running with a visible display")
            print("3. Use pre-rendered templates (if available)")
            print("\nOriginal error:", str(e))
            print("="*80 + "\n")
        raise
    
    rgb_images = []
    depth_images = []
    extrinsics = []
    
    try:
        for cam_pose in camera_poses:
            # Add camera to scene
            cam_node = scene.add(camera, pose=cam_pose)
            
            # Render
            color, depth = renderer.render(scene)
            
            # Store results
            rgb_images.append(color)
            depth_images.append(depth)
            
            # Compute extrinsics (world-to-camera)
            extrinsic = np.linalg.inv(cam_pose)
            extrinsics.append(extrinsic)
            
            # Remove camera for next iteration
            scene.remove_node(cam_node)
    
    finally:
        renderer.delete()
    
    return rgb_images, depth_images, extrinsics


def create_intrinsics_matrix(
    width: int,
    height: int,
    fov_deg: float
) -> np.ndarray:
    """
    Create camera intrinsics matrix from FOV.
    
    Args:
        width: Image width
        height: Image height
        fov_deg: Vertical field of view in degrees
        
    Returns:
        Camera intrinsics matrix [3, 3]
    """
    fov_rad = np.deg2rad(fov_deg)
    fy = height / (2 * np.tan(fov_rad / 2))
    fx = fy  # Assume square pixels
    cx = width / 2
    cy = height / 2
    
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    
    return K


class TemplateRenderer:
    """
    Template renderer for caching and managing rendered views.
    """
    
    def __init__(self, config: dict):
        """
        Initialize renderer with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.templates = None
        self.camera_poses = None
        self.K_render = None
    
    def generate_templates(self, mesh_path: str) -> dict:
        """
        Generate template views from mesh.
        
        Args:
            mesh_path: Path to mesh file
            
        Returns:
            Dictionary containing templates and metadata
        """
        print(f"Generating {self.config['RENDER']['N_VIEWS']} template views...")
        
        # Sample camera poses
        self.camera_poses = sample_camera_poses(
            n_views=self.config['RENDER']['N_VIEWS'],
            azimuth_range=self.config['RENDER']['AZIMUTH_DEG'],
            elevation_range=self.config['RENDER']['ELEVATION_DEG'],
            distance=self.config['RENDER']['DISTANCE']
        )
        
        # Render views
        height, width = self.config['RENDER']['IMG_SIZE']
        rgb_images, depth_images, extrinsics = render_mesh_views(
            mesh_path=mesh_path,
            height=height,
            width=width,
            fov_deg=self.config['RENDER']['FOV_DEG'],
            camera_poses=self.camera_poses
        )
        
        # Create intrinsics
        self.K_render = create_intrinsics_matrix(
            width=width,
            height=height,
            fov_deg=self.config['RENDER']['FOV_DEG']
        )
        
        self.templates = {
            'rgb': rgb_images,
            'depth': depth_images,
            'extrinsics': extrinsics,
            'K': self.K_render,
            'poses': self.camera_poses
        }
        
        print(f"Generated {len(rgb_images)} templates")
        return self.templates
    
    def get_templates(self) -> dict:
        """
        Get cached templates.
        
        Returns:
            Template dictionary
        """
        if self.templates is None:
            raise RuntimeError("Templates not generated. Call generate_templates first.")
        return self.templates

