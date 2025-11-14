"""
Feature extraction and matching using DINOv2 foundation models.
Implements Soft Nearest Neighbor (SNN) matching for cycle-consistent correspondences.
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import timm
from PIL import Image
import torchvision.transforms as transforms


def build_backbone(
    name: str = 'dinov2_vitb14',
    weights: Optional[str] = None,
    device: str = 'cuda'
) -> torch.nn.Module:
    """
    Build DINOv2 backbone model.
    
    Args:
        name: Model name from timm (e.g., 'dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14')
        weights: Path to local weights (None to use pretrained from timm)
        device: Device to load model on
        
    Returns:
        DINOv2 model
    """
    # Check if CUDA is available
    if device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        device = 'cpu'
    
    # Load model from timm
    if weights is None:
        model = timm.create_model(name, pretrained=True, num_classes=0)
    else:
        model = timm.create_model(name, pretrained=False, num_classes=0)
        model.load_state_dict(torch.load(weights, map_location=device))
    
    model = model.to(device)
    model.eval()
    
    return model


def preprocess_image(
    img: np.ndarray,
    size: int = 518
) -> torch.Tensor:
    """
    Preprocess image for DINOv2.
    
    Args:
        img: Input image [H, W, 3] uint8
        size: Target size (DINOv2 uses 518x518 by default)
        
    Returns:
        Preprocessed tensor [1, 3, H, W]
    """
    # Convert to PIL
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    
    # DINOv2 preprocessing
    transform = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(img).unsqueeze(0)
    return img_tensor


@torch.no_grad()
def extract_feats(
    model: torch.nn.Module,
    img_bchw: torch.Tensor,
    return_cls_token: bool = False
) -> torch.Tensor:
    """
    Extract dense features from DINOv2.
    
    Args:
        model: DINOv2 model
        img_bchw: Input images [B, 3, H, W]
        return_cls_token: Whether to return CLS token separately
        
    Returns:
        Dense feature map [B, H_feat, W_feat, C] or tuple if return_cls_token=True
    """
    device = next(model.parameters()).device
    img_bchw = img_bchw.to(device)
    
    # Forward pass
    features = model.forward_features(img_bchw)
    
    # Extract patch tokens (remove CLS token)
    if isinstance(features, dict):
        patch_tokens = features['x_norm_patchtokens']
    else:
        # Assume output is [B, N+1, C] where first token is CLS
        patch_tokens = features[:, 1:, :]
    
    B, N, C = patch_tokens.shape
    
    # Reshape to spatial grid
    H = W = int(np.sqrt(N))
    feat_map = patch_tokens.reshape(B, H, W, C)
    
    if return_cls_token:
        cls_token = features[:, 0, :] if not isinstance(features, dict) else features['x_norm_clstoken']
        return feat_map, cls_token
    
    return feat_map


def normalize_features(feats: torch.Tensor) -> torch.Tensor:
    """
    L2 normalize features.
    
    Args:
        feats: Features [..., C]
        
    Returns:
        Normalized features
    """
    return F.normalize(feats, p=2, dim=-1)


def soft_nearest_neighbors(
    feat_query: torch.Tensor,
    feat_ref: torch.Tensor,
    topk: int = 2000,
    cycle_threshold: float = 0.0
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    """
    Compute Soft Nearest Neighbor (SNN) matches with cycle consistency.
    
    Args:
        feat_query: Query features [H_q, W_q, C]
        feat_ref: Reference features [H_r, W_r, C]
        topk: Maximum number of matches to return
        cycle_threshold: Threshold for cycle consistency (0 for exact cycle)
        
    Returns:
        Tuple of (query_coords, ref_coords, (H_q, W_q, H_r, W_r))
        - query_coords: [N, 2] (u, v) in query feature grid
        - ref_coords: [N, 2] (u, v) in reference feature grid
    """
    device = feat_query.device
    H_q, W_q, C = feat_query.shape
    H_r, W_r, _ = feat_ref.shape
    
    # Flatten spatial dimensions
    feat_query_flat = feat_query.reshape(-1, C)  # [H_q*W_q, C]
    feat_ref_flat = feat_ref.reshape(-1, C)  # [H_r*W_r, C]
    
    # Normalize features
    feat_query_flat = normalize_features(feat_query_flat)
    feat_ref_flat = normalize_features(feat_ref_flat)
    
    # Compute similarity matrix (query -> ref)
    sim_matrix = torch.matmul(feat_query_flat, feat_ref_flat.T)  # [H_q*W_q, H_r*W_r]
    
    # Find nearest neighbors (query -> ref)
    nn_ref_indices = torch.argmax(sim_matrix, dim=1)  # [H_q*W_q]
    nn_ref_scores = sim_matrix[torch.arange(len(nn_ref_indices)), nn_ref_indices]
    
    # Find nearest neighbors (ref -> query) for cycle consistency
    nn_query_indices = torch.argmax(sim_matrix, dim=0)  # [H_r*W_r]
    
    # Check cycle consistency: query -> ref -> query
    cycle_consistent = nn_query_indices[nn_ref_indices] == torch.arange(len(nn_ref_indices), device=device)
    
    # Filter by cycle consistency
    valid_mask = cycle_consistent
    valid_query_indices = torch.where(valid_mask)[0]
    valid_ref_indices = nn_ref_indices[valid_mask]
    valid_scores = nn_ref_scores[valid_mask]
    
    # Sort by score and take top-k
    if len(valid_query_indices) > topk:
        _, top_indices = torch.topk(valid_scores, k=topk)
        valid_query_indices = valid_query_indices[top_indices]
        valid_ref_indices = valid_ref_indices[top_indices]
    
    # Convert flat indices to 2D coordinates
    query_v = (valid_query_indices // W_q).cpu().numpy()
    query_u = (valid_query_indices % W_q).cpu().numpy()
    
    ref_v = (valid_ref_indices // W_r).cpu().numpy()
    ref_u = (valid_ref_indices % W_r).cpu().numpy()
    
    query_coords = np.stack([query_u, query_v], axis=-1)  # [N, 2]
    ref_coords = np.stack([ref_u, ref_v], axis=-1)  # [N, 2]
    
    return query_coords, ref_coords, (H_q, W_q, H_r, W_r)


def upsample_coords(
    coords: np.ndarray,
    feat_size: Tuple[int, int],
    img_size: Tuple[int, int]
) -> np.ndarray:
    """
    Upsample feature coordinates to image coordinates.
    
    Args:
        coords: Coordinates in feature grid [N, 2]
        feat_size: Feature map size (H_feat, W_feat)
        img_size: Image size (H_img, W_img)
        
    Returns:
        Upsampled coordinates [N, 2]
    """
    H_feat, W_feat = feat_size
    H_img, W_img = img_size
    
    scale_u = W_img / W_feat
    scale_v = H_img / H_feat
    
    coords_upsampled = coords.copy()
    coords_upsampled[:, 0] = (coords[:, 0] + 0.5) * scale_u
    coords_upsampled[:, 1] = (coords[:, 1] + 0.5) * scale_v
    
    return coords_upsampled


class FeatureExtractor:
    """
    Feature extractor wrapper for managing DINOv2 model and caching.
    """
    
    def __init__(self, config: dict):
        """
        Initialize feature extractor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = build_backbone(
            name=config['MODEL']['BACKBONE'],
            weights=config['MODEL']['WEIGHTS'],
            device=self.device
        )
        print(f"Loaded {config['MODEL']['BACKBONE']} on {self.device}")
    
    def extract(self, img: np.ndarray) -> torch.Tensor:
        """
        Extract features from image.
        
        Args:
            img: RGB image [H, W, 3] uint8
            
        Returns:
            Feature map [H_feat, W_feat, C]
        """
        img_tensor = preprocess_image(img)
        feat_map = extract_feats(self.model, img_tensor)
        return feat_map[0]  # Remove batch dimension
    
    def match(
        self,
        feat_query: torch.Tensor,
        feat_ref: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray, Tuple]:
        """
        Match features using SNN.
        
        Args:
            feat_query: Query features [H_q, W_q, C]
            feat_ref: Reference features [H_r, W_r, C]
            
        Returns:
            Tuple of (query_coords, ref_coords, metadata)
        """
        return soft_nearest_neighbors(
            feat_query,
            feat_ref,
            topk=self.config['MATCH']['TOPK']
        )
    
    def extract_batch(self, images: List[np.ndarray]) -> List[torch.Tensor]:
        """
        Extract features from batch of images.
        
        Args:
            images: List of RGB images
            
        Returns:
            List of feature maps
        """
        feats = []
        for img in images:
            feat = self.extract(img)
            feats.append(feat)
        return feats

