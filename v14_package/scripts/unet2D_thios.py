"""
ThioS Multiclass Segmentation Model (EfficientNet + U-Net)

This module defines the model architecture and data pipeline for 5-class
ThioS neuropathology segmentation:
- Class 0: Unlabeled (ignored in loss)
- Class 1: Background
- Class 2: Diffuse amyloid plaque
- Class 3: Cored amyloid plaque
- Class 4: Tau tangle

Key features for low-data training:
1. Heavy data augmentation (Mixup, CutMix, geometric, photometric)
2. EfficientNet-B4 pretrained encoder (transfer learning)
3. Class-weighted Focal + Tversky loss
4. Spatial weight maps for boundary emphasis

Author: ThioS Classification Pipeline
Date: January 2026
"""

import os
import json
import csv
import sys
import numpy as np
import torch
import glob
import pytorch_lightning as pl
import torchvision.utils
from torchvision.transforms import ToPILImage

import torchmetrics
from monai.data import list_data_collate
from monai.networks.nets import FlexibleUNet
from monai.networks.layers import Norm
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss, FocalLoss, TverskyLoss

from skimage import io, exposure
from PIL import Image
import torch
import pytorch_lightning as pl

from torch.utils.data import Dataset
import torchmetrics

from monai.transforms import (
    MapTransform,
    Compose,
    CastToTyped,
    MapLabelValued,
    Resized,
    RandFlipd,
    RandAffined,
    ScaleIntensityRanged,
    RandGaussianNoised,
    RandAdjustContrastd,
    RandGaussianSmoothd,
    RandStdShiftIntensityd,
    RandScaleIntensityd,
    RandRotate90d,
    EnsureTyped,
    ScaleIntensityRange,
    EnsureType,
    RandBiasFieldd,
    RandShiftIntensityd,
    RandHistogramShiftd,
    RandZoomd,
    RandGridDistortiond,
    RandCoarseDropoutd,
)

# Class configuration
NUM_CLASSES = 5
CLASS_NAMES = {0: 'unlabeled', 1: 'background', 2: 'diffuse', 3: 'plaque', 4: 'tangle'}
BIOMARKER_CLASSES = [2, 3, 4]
BACKGROUND_CLASS = 1
UNLABELED_CLASS = 0


# =============================================================================
# Attention Gate Module for Skip Connections
# =============================================================================

class AttentionGate(torch.nn.Module):
    """
    Attention Gate for U-Net skip connections.
    
    Based on "Attention U-Net: Learning Where to Look for the Pancreas"
    (Oktay et al., 2018)
    
    The attention gate learns to focus on relevant regions by using
    the decoder (gating) signal to modulate encoder (skip) features.
    
    Args:
        F_g: Number of channels in gating signal (from decoder)
        F_l: Number of channels in skip connection (from encoder)
        F_int: Number of intermediate channels
    """
    
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super(AttentionGate, self).__init__()
        
        # Transform gating signal
        self.W_g = torch.nn.Sequential(
            torch.nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            torch.nn.BatchNorm2d(F_int)
        )
        
        # Transform skip connection
        self.W_x = torch.nn.Sequential(
            torch.nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            torch.nn.BatchNorm2d(F_int)
        )
        
        # Attention coefficients
        self.psi = torch.nn.Sequential(
            torch.nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            torch.nn.BatchNorm2d(1),
            torch.nn.Sigmoid()
        )
        
        self.relu = torch.nn.ReLU(inplace=True)
    
    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            g: Gating signal from decoder (coarser resolution)
            x: Skip connection from encoder (finer resolution)
        
        Returns:
            Attention-weighted skip features
        """
        # Upsample gating signal to match skip connection size
        g_upsampled = torch.nn.functional.interpolate(
            g, size=x.shape[2:], mode='bilinear', align_corners=True
        )
        
        # Transform both signals
        g1 = self.W_g(g_upsampled)
        x1 = self.W_x(x)
        
        # Combine and compute attention
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        # Apply attention to skip connection
        return x * psi


class FlexibleUNetWithAttention(torch.nn.Module):
    """
    Custom U-Net with EfficientNet encoder and Attention Gates on skip connections.
    
    This implementation:
    1. Uses timm's EfficientNet as encoder (pretrained)
    2. Custom decoder with attention-gated skip connections
    3. Full control over skip connection modulation
    
    The attention gates learn to focus on relevant regions (biomarkers)
    while suppressing background features.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 5,
        backbone: str = "efficientnet-b0",
        pretrained: bool = True,
        decoder_channels: tuple = (256, 128, 64, 32, 16),
        spatial_dims: int = 2,
        **kwargs
    ):
        super(FlexibleUNetWithAttention, self).__init__()
        
        self.backbone_name = backbone
        
        # Import timm for EfficientNet encoder
        try:
            import timm
            has_timm = True
        except ImportError:
            has_timm = False
            print("Warning: timm not available, falling back to base FlexibleUNet without attention")
        
        if not has_timm:
            # Fallback: use MONAI's FlexibleUNet without attention
            self.base_unet = FlexibleUNet(
                in_channels=in_channels,
                out_channels=out_channels,
                backbone=backbone,
                pretrained=pretrained,
                decoder_channels=decoder_channels,
                spatial_dims=spatial_dims,
                **kwargs
            )
            self.use_custom = False
            self.attention_gates = torch.nn.ModuleList()
            print("Using base FlexibleUNet (no attention)")
            return
        
        self.use_custom = True
        
        # Create EfficientNet encoder using timm
        timm_backbone = backbone.replace('-', '_')  # efficientnet-b0 -> efficientnet_b0
        self.encoder = timm.create_model(
            timm_backbone,
            pretrained=pretrained,
            features_only=True,  # Extract features at multiple scales
            in_chans=in_channels
        )
        
        # Get encoder feature channels from timm model
        self.encoder_channels = self.encoder.feature_info.channels()
        print(f"Encoder channels: {self.encoder_channels}")
        
        # Decoder blocks with attention gates
        self.decoder_channels = decoder_channels
        
        # Build decoder: each block upsamples and combines with skip connection
        self.decoder_blocks = torch.nn.ModuleList()
        self.attention_gates = torch.nn.ModuleList()
        
        # Input to first decoder block is last encoder output
        prev_channels = self.encoder_channels[-1]
        
        for i, dec_ch in enumerate(decoder_channels):
            # Skip connection channels (from encoder, reversed order)
            skip_idx = len(self.encoder_channels) - 2 - i
            if skip_idx >= 0:
                skip_ch = self.encoder_channels[skip_idx]
            else:
                skip_ch = 0  # No skip connection for deepest levels
            
            # Attention gate for this skip connection
            if skip_ch > 0:
                F_int = max(min(prev_channels, skip_ch) // 4, 8)
                self.attention_gates.append(
                    AttentionGate(F_g=prev_channels, F_l=skip_ch, F_int=F_int)
                )
            else:
                self.attention_gates.append(None)
            
            # Decoder block: upsample + conv
            in_ch = prev_channels + skip_ch if skip_ch > 0 else prev_channels
            
            self.decoder_blocks.append(
                torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(prev_channels, prev_channels, kernel_size=2, stride=2),
                    torch.nn.BatchNorm2d(prev_channels),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Conv2d(in_ch, dec_ch, kernel_size=3, padding=1),
                    torch.nn.BatchNorm2d(dec_ch),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Conv2d(dec_ch, dec_ch, kernel_size=3, padding=1),
                    torch.nn.BatchNorm2d(dec_ch),
                    torch.nn.ReLU(inplace=True),
                )
            )
            prev_channels = dec_ch
        
        # Final segmentation head
        self.seg_head = torch.nn.Conv2d(decoder_channels[-1], out_channels, kernel_size=1)
        
        self.use_attention = True
        print(f"Custom UNet with Attention Gates: {sum(1 for ag in self.attention_gates if ag is not None)} attention modules")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with attention-gated skip connections."""
        
        if not self.use_custom:
            return self.base_unet(x)
        
        input_size = x.shape[2:]
        
        # Encoder forward - get features at each scale
        encoder_features = self.encoder(x)
        
        # Start decoding from deepest features
        dec = encoder_features[-1]
        
        # Decode with attention-gated skip connections
        for i, (decoder_block, attn_gate) in enumerate(zip(self.decoder_blocks, self.attention_gates)):
            # Upsample
            dec = decoder_block[0:3](dec)  # ConvTranspose + BN + ReLU
            
            # Get skip connection
            skip_idx = len(encoder_features) - 2 - i
            if skip_idx >= 0 and attn_gate is not None:
                skip = encoder_features[skip_idx]
                
                # Resize if needed
                if dec.shape[2:] != skip.shape[2:]:
                    dec = torch.nn.functional.interpolate(dec, size=skip.shape[2:], mode='bilinear', align_corners=True)
                
                # Apply attention gate
                if self.use_attention:
                    skip = attn_gate(dec, skip)
                
                # Concatenate
                dec = torch.cat([dec, skip], dim=1)
            
            # Conv blocks
            dec = decoder_block[3:](dec)
        
        # Final upsampling to match input size
        if dec.shape[2:] != input_size:
            dec = torch.nn.functional.interpolate(dec, size=input_size, mode='bilinear', align_corners=True)
        
        # Segmentation head
        return self.seg_head(dec)
    
    def enable_attention(self, enable: bool = True):
        """Enable or disable attention gates."""
        self.use_attention = enable
    
    def parameters(self, recurse: bool = True):
        """Return all parameters."""
        if self.use_custom:
            yield from self.encoder.parameters(recurse)
            yield from self.decoder_blocks.parameters(recurse)
            for ag in self.attention_gates:
                if ag is not None:
                    yield from ag.parameters(recurse)
            yield from self.seg_head.parameters(recurse)
        else:
            yield from self.base_unet.parameters(recurse)


def dynamic_scale(image: np.ndarray) -> np.ndarray:
    """Scales an input image by determining the maximum and minimum pixel values."""
    a_min, a_max = image.min(), image.max()
    transform = ScaleIntensityRange(
        a_min=a_min,
        a_max=a_max,
        b_min=0.0,
        b_max=1.0,
        clip=True,
    )
    return transform(image)


def fixed_scale(image: np.ndarray) -> np.ndarray:
    """Scales an input image by dividing by 255.0 to preserve absolute intensity."""
    return image.astype(np.float32) / 255.0


def compute_boundary_weight_map(label: np.ndarray, boundary_width: int = 3, 
                                boundary_boost: float = 5.0) -> np.ndarray:
    """
    Create weight map that emphasizes pixels near annotation boundaries.
    
    For pathology quantification, boundary precision is critical.
    This gives higher weight to pixels near class transitions.
    
    Args:
        label: (H, W) uint8 label mask with class indices 0-4
        boundary_width: pixels from boundary to boost (default 3)
        boundary_boost: weight multiplier at boundary (default 5.0)
    
    Returns:
        (H, W) float32 weight map, 1.0 for interior, up to boundary_boost at edges
    """
    from scipy import ndimage
    
    weight = np.ones(label.shape[:2], dtype=np.float32)
    
    # For each labeled class, find boundary pixels
    for c in range(1, NUM_CLASSES):  # skip unlabeled
        class_mask = (label == c).astype(np.float32)
        if class_mask.sum() == 0:
            continue
        
        # Erode and subtract to get boundary ring
        eroded = ndimage.binary_erosion(class_mask, iterations=boundary_width).astype(np.float32)
        boundary = class_mask - eroded
        
        # Distance from boundary (smooth falloff)
        if boundary.sum() > 0:
            dist = ndimage.distance_transform_edt(1.0 - boundary)
            # Exponential decay: high weight at boundary, decays with distance
            boundary_weight = boundary_boost * np.exp(-dist / boundary_width)
            # Only apply inside labeled regions
            labeled_mask = (label > 0).astype(np.float32)
            weight = np.maximum(weight, boundary_weight * labeled_mask)
    
    return weight


def compute_boundary_distance_map(label: np.ndarray) -> np.ndarray:
    """
    Compute signed distance transform from class boundaries.
    Used by boundary loss to penalize predictions that don't align with GT edges.
    
    Returns:
        (NUM_CLASSES, H, W) float32 distance map. Negative inside, positive outside.
    """
    from scipy import ndimage
    
    h, w = label.shape[:2]
    dist_maps = np.zeros((NUM_CLASSES, h, w), dtype=np.float32)
    
    for c in range(1, NUM_CLASSES):
        class_mask = (label == c).astype(np.bool_)
        if class_mask.sum() == 0 or class_mask.sum() == class_mask.size:
            continue
        
        # Distance transform: positive outside, negative inside
        pos_dist = ndimage.distance_transform_edt(~class_mask)
        neg_dist = ndimage.distance_transform_edt(class_mask)
        # Signed distance: negative inside GT, positive outside
        dist_maps[c] = pos_dist - neg_dist
    
    return dist_maps


def get_class_weights(labels: torch.Tensor, num_classes: int, device: torch.device) -> torch.Tensor:
    """
    Compute class weights inversely proportional to class frequency in batch.
    
    Only considers labeled pixels (classes 1-4), ignores unlabeled (class 0).
    
    Args:
        labels: Label tensor
        num_classes: Number of classes
        device: Device to place weights tensor
        
    Returns:
        Weight tensor of shape (num_classes,)
    """
    # Flatten labels
    labels_flat = labels.view(-1)
    
    # Only count labeled pixels (exclude class 0)
    labeled_mask = labels_flat > 0
    labels_labeled = labels_flat[labeled_mask]
    
    # Count each class (only among labeled pixels)
    counts = torch.zeros(num_classes, device=device)
    for c in range(1, num_classes):  # Skip class 0
        counts[c] = (labels_labeled == c).sum().float()
    
    # Avoid division by zero
    counts = torch.clamp(counts, min=1.0)
    
    # Inverse frequency weighting (only for classes 1-4)
    weights = torch.zeros(num_classes, device=device)
    for c in range(1, num_classes):
        weights[c] = 1.0 / counts[c]
    
    # Normalize so mean weight of labeled classes is 1
    if weights[1:].sum() > 0:
        weights[1:] = weights[1:] / weights[1:].mean()
    
    # Class 0 weight stays 0 (ignored in loss anyway)
    weights[0] = 0.0
    
    return weights


class ThioSTiffReader(MapTransform):
    """
    Custom TIFF reader for ThioS fluorescent images and multiclass labels.
    V12: Supports boundary-aware weight map enhancement.
    """
    def __init__(
        self, 
        keys=["image", "label"], 
        scaling_method="fixed", 
        contrast_enhance=True,
        boundary_weight_boost=0.0,
        boundary_width=3,
        *args, **kwargs
    ):
        super().__init__(keys, *args, **kwargs)
        self.keys = keys
        self.scaling_method = scaling_method
        self.contrast_enhance = contrast_enhance
        self.boundary_weight_boost = boundary_weight_boost
        self.boundary_width = boundary_width

    def __call__(self, data_dict):
        scale_func = fixed_scale if self.scaling_method == "fixed" else dynamic_scale
        d = dict(data_dict)
        
        # 1. Load Image
        image_path = d[self.keys[0]]
        if isinstance(image_path, str):
            try:
                img = io.imread(image_path)
            except Exception as e:
                print(f"Error reading image {image_path}: {e}")
                img = np.zeros((256, 256, 3), dtype=np.uint8)
        else:
            img = image_path

        # 2. Contrast Enhancement (CLAHE) for fluorescence normalization
        if self.contrast_enhance and img.max() > 0:
            # Normalize to 0-1 for CLAHE
            if img.dtype != np.float32:
                if img.max() > img.min():
                    img = (img.astype(np.float32) - img.min()) / (img.max() - img.min())
                else:
                    img = np.zeros_like(img, dtype=np.float32)
            # Apply CLAHE with gentle clip limit for fluorescence
            img = exposure.equalize_adapthist(img, clip_limit=0.02)
        else:
            img = scale_func(img)

        # 3. Dimension adjustments (ensure Channel First: C, H, W)
        if img.ndim == 2:
            img = img[np.newaxis, ...]
        elif img.ndim == 3:
            # If channels are last (H, W, C) -> (C, H, W)
            if img.shape[-1] <= 4 and img.shape[0] > 4:
                img = np.transpose(img, (2, 0, 1))
        
        d[self.keys[0]] = img.astype(np.float32)

        # 4. Process Label if it exists (multiclass: 0-4)
        if len(self.keys) > 1 and self.keys[1] in d:
            lbl = d[self.keys[1]]
            if isinstance(lbl, str):
                lbl = io.imread(lbl)
            
            # Ensure label is 2D
            if lbl.ndim == 3:
                lbl = lbl[:, :, 0]
            
            # Add channel dim
            if lbl.ndim == 2:
                lbl = lbl[np.newaxis, ...]
            
            d[self.keys[1]] = lbl.astype(np.float32)

        # 5. Process Weight Map if it exists
        if "weight" in self.keys and "weight" in d:
            w_path = d["weight"]
            if isinstance(w_path, str):
                try:
                    weight_map = io.imread(w_path)
                except (FileNotFoundError, OSError):
                    print(f"WARNING: Weight map missing: {w_path}. Using default weights.")
                    current_img = d[self.keys[0]]
                    spatial_shape = current_img.shape[-2:]
                    weight_map = np.ones(spatial_shape, dtype=np.float32)
            else:
                weight_map = w_path
            
            weight_map = weight_map.astype(np.float32)
            
            # V12: Add boundary-aware weight boost
            if self.boundary_weight_boost > 0 and len(self.keys) > 1 and self.keys[1] in d:
                lbl_for_boundary = d[self.keys[1]]
                if isinstance(lbl_for_boundary, np.ndarray):
                    lbl_2d = lbl_for_boundary.squeeze() if lbl_for_boundary.ndim > 2 else lbl_for_boundary
                    boundary_wt = compute_boundary_weight_map(
                        lbl_2d.astype(np.uint8), 
                        boundary_width=self.boundary_width,
                        boundary_boost=self.boundary_weight_boost
                    )
                    # Multiply with existing weight map
                    if weight_map.ndim == 2:
                        weight_map = weight_map * boundary_wt
                    else:
                        weight_map = weight_map * boundary_wt[np.newaxis, ...]
            
            if weight_map.ndim == 2:
                weight_map = weight_map[np.newaxis, ...]
            d["weight"] = weight_map

        return d


class ThioSDataset(Dataset):
    """
    Dataset for ThioS multiclass segmentation with heavy augmentation.
    Includes oversampling of patches containing rare biomarker classes.
    """
    def __init__(self, data_fnames, label_fnames, args, training=False):
        self.training = training
        self.data_fnames = list(data_fnames)
        self.label_fnames = list(label_fnames)
        self.args = args
        
        # Weight file naming: replace /labels/ with /weights/ and _mask_ with _img_
        self.weight_fnames = []
        for f in self.data_fnames:
            weight_path = f.replace('/images/', '/weights/')
            self.weight_fnames.append(weight_path)
        
        # Parameters from config
        self.patch_size = args.get('patch_size', (256, 256))
        self.target_size = tuple(self.patch_size)
        self.input_col = args.get('input_channels', 3)
        self.scaling_method = args.get('scaling_method', 'fixed')
        self.contrast_enhance = args.get('contrast_enhance', True)
        
        # V12: Boundary-aware weight parameters
        self.boundary_weight_boost = args.get('boundary_weight_boost', 0.0)
        self.boundary_width = args.get('boundary_width', 3)
        
        # Augmentation parameters
        self.rotate_range = args.get('rotate_range', 0.1)
        self.translate_range = args.get('translate_range', 0.15)
        self.shear_range = args.get('shear_range', 0.1)
        self.scale_range = args.get('scale_range', 0.15)
        
        # For Mixup/CutMix (handled separately in training_step)
        self.use_mixup = args.get('use_mixup', True)
        self.mixup_alpha = args.get('mixup_alpha', 0.4)
        
        # V13+: Configurable oversampling factors (default = V10/V11/V12 values)
        self.oversample_factors = args.get('oversample_factors', {4: 3, 2: 2, 3: 1})
        # Convert string keys to int if from JSON
        self.oversample_factors = {int(k): v for k, v in self.oversample_factors.items()}
        
        # Oversampling for rare classes (only during training)
        # Class 2=diffuse, 3=plaque, 4=tangle
        if training:
            self._apply_oversampling()
        
        self.Nsamples = len(self.data_fnames)
        
        # Data dictionaries
        self.data_dicts = [
            {
                "image": self.data_fnames[i], 
                "label": self.label_fnames[i],
                "weight": self.weight_fnames[i]
            } 
            for i in range(self.Nsamples)
        ]
        
        self.transform = self._get_transforms()
    
    def _apply_oversampling(self):
        """
        Oversample patches containing rare biomarker classes.
        - Tangle (class 4): 3x oversampling (rarest)
        - Diffuse (class 2): 2x oversampling (rare)
        - Plaque (class 3): 1.5x oversampling (medium rare)
        """
        from skimage import io as skio
        import os
        
        # Oversampling multipliers for each biomarker class (configurable via args)
        oversample_factors = self.oversample_factors
        
        original_count = len(self.data_fnames)
        extra_images = []
        extra_labels = []
        extra_weights = []
        
        print("Analyzing patches for oversampling...")
        for idx, (img_path, lbl_path, wgt_path) in enumerate(zip(
            self.data_fnames, self.label_fnames, self.weight_fnames
        )):
            # Quick check: read label to see which classes present
            try:
                if os.path.exists(lbl_path):
                    lbl = skio.imread(lbl_path)
                    unique_classes = set(np.unique(lbl).tolist())
                    
                    # Find highest priority class in this patch
                    # Priority: tangle > diffuse > plaque
                    extra_copies = 0
                    for class_id in [4, 2, 3]:  # Check in priority order
                        if class_id in unique_classes:
                            extra_copies = oversample_factors[class_id] - 1
                            break
                    
                    # Add extra copies
                    for _ in range(extra_copies):
                        extra_images.append(img_path)
                        extra_labels.append(lbl_path)
                        extra_weights.append(wgt_path)
            except Exception:
                pass
        
        # Add oversampled patches
        self.data_fnames.extend(extra_images)
        self.label_fnames.extend(extra_labels)
        self.weight_fnames.extend(extra_weights)
        
        print(f"Oversampling: {original_count} -> {len(self.data_fnames)} patches")
        print(f"  Added {len(extra_images)} oversampled patches for rare classes")
        
        self.transform = self._get_transforms()

    def __len__(self):
        return self.Nsamples

    def __getitem__(self, idx):
        # Retry logic for corrupted patches
        if self.training:
            for _ in range(10):
                try:
                    data = self.transform(self.data_dicts[idx])
                    if data['image'].max() > 0:
                        return data
                except Exception:
                    pass
                idx = torch.randint(0, len(self.data_dicts), (1,)).item()
        return self.transform(self.data_dicts[idx])

    def _get_transforms(self):
        """Build augmentation pipeline with heavy augmentation for low-data regime."""
        keys_list = ["image", "label", "weight"]
        
        if not self.training:
            # Validation: minimal transforms
            return Compose([
                ThioSTiffReader(
                    keys=keys_list,
                    scaling_method=self.scaling_method,
                    contrast_enhance=self.contrast_enhance,
                    boundary_weight_boost=self.boundary_weight_boost,
                    boundary_width=self.boundary_width,
                ),
                Resized(
                    keys=keys_list, 
                    spatial_size=self.target_size, 
                    mode=['area', 'nearest', 'nearest']
                ),
                CastToTyped(keys=["label"], dtype=torch.long),
                EnsureTyped(keys=keys_list)
            ])
        
        # Training: Heavy augmentation for low-data regime
        return Compose([
            ThioSTiffReader(
                keys=keys_list,
                scaling_method=self.scaling_method,
                contrast_enhance=self.contrast_enhance,
                boundary_weight_boost=self.boundary_weight_boost,
                boundary_width=self.boundary_width,
            ),
            Resized(
                keys=keys_list, 
                spatial_size=self.target_size, 
                mode=['area', 'nearest', 'nearest']
            ),
            
            # === Geometric Augmentations ===
            RandFlipd(keys=keys_list, prob=0.5, spatial_axis=0),
            RandFlipd(keys=keys_list, prob=0.5, spatial_axis=1),
            RandRotate90d(keys=keys_list, prob=0.75, max_k=3),
            
            # Affine transformations
            RandAffined(
                keys=keys_list,
                mode=['bilinear', 'nearest', 'nearest'],
                padding_mode='zeros', 
                prob=0.8,
                spatial_size=self.patch_size,
                rotate_range=[self.rotate_range] * 2,
                translate_range=[int(self.translate_range * s) for s in self.patch_size],
                shear_range=[self.shear_range] * 2,
                scale_range=[self.scale_range] * 2
            ),
            
            # Elastic deformation for bio-realistic variations
            RandGridDistortiond(
                keys=keys_list,
                num_cells=5,
                prob=0.3,
                distort_limit=0.2,
                mode=['bilinear', 'nearest', 'nearest']
            ),
            
            # Random zoom for scale invariance
            RandZoomd(
                keys=keys_list,
                min_zoom=0.85,
                max_zoom=1.15,
                prob=0.4,
                mode=['area', 'nearest', 'nearest']
            ),
            
            # === Photometric Augmentations (image only) ===
            RandAdjustContrastd(keys=["image"], prob=0.6, gamma=(0.7, 1.3)),
            RandGaussianNoised(keys=["image"], prob=0.5, std=0.08),
            RandGaussianSmoothd(
                keys=["image"], 
                prob=0.4, 
                sigma_x=(0.5, 1.5), 
                sigma_y=(0.5, 1.5)
            ),
            RandStdShiftIntensityd(keys=["image"], factors=0.3, prob=0.5),
            RandScaleIntensityd(keys=["image"], factors=0.3, prob=0.5),
            
            # Bias field to simulate uneven illumination (fluorescence artifact)
            RandBiasFieldd(keys=["image"], coeff_range=(0.1, 0.3), prob=0.4),
            
            # Intensity shifts
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.4),
            RandHistogramShiftd(keys=["image"], num_control_points=5, prob=0.3),
            
            # Coarse dropout (cutout) for regularization
            RandCoarseDropoutd(
                keys=["image"],
                holes=8,
                spatial_size=16,
                dropout_holes=True,
                fill_value=0,
                prob=0.3
            ),
            
            CastToTyped(keys=["label"], dtype=torch.long),
            EnsureTyped(keys=keys_list)
        ])


class PredDataset2D(Dataset):
    """Dataset for prediction (inference)."""
    def __init__(self, pred_data_dir, args):
        self.pred_data_dir = pred_data_dir
        self.data_file = self._get_image_paths(pred_data_dir)
        self.input_col = args.get('input_channels', 3)
        self.contrast_enhance = args.get('contrast_enhance', True)

    def _get_image_paths(self, directory):
        """Get all image files."""
        extensions = ['.tiff', '.tif', '.png', '.jpg']
        paths = []
        for ext in extensions:
            paths.extend(glob.glob(os.path.join(directory, f'*{ext}')))
            paths.extend(glob.glob(os.path.join(directory, f'*{ext.upper()}')))
        return sorted(paths)

    def __len__(self):
        return len(self.data_file)

    def __getitem__(self, idx):
        img_path = self.data_file[idx]
        img = np.array(Image.open(img_path))
        
        # Handle dimensions
        if img.ndim == 3 and img.shape[-1] <= 4:
            img = np.transpose(img[..., :3], (2, 0, 1))
        elif img.ndim == 2:
            img = np.expand_dims(img, axis=0)
        
        # Normalize
        img = dynamic_scale(img)
        
        transform = Compose([EnsureType()])
        img = transform(img)
        
        return {"image": img, "image_meta_dict": {"filename_or_obj": img_path}}


class ThioSUnet2D(pl.LightningModule):
    """
    PyTorch Lightning Module for ThioS multiclass segmentation.
    
    Architecture: EfficientNet-B4 encoder + UNet decoder
    Loss: Focal Loss + Tversky Loss (weighted combination)
    """
    def __init__(self, train_ds, val_ds, **kwargs):
        super().__init__()
        
        self.save_hyperparameters(ignore=['train_ds', 'val_ds'])
        self.train_ds = train_ds
        self.val_ds = val_ds
        
        # Set defaults
        self.hparams.setdefault('num_classes', NUM_CLASSES)
        self.hparams.setdefault('input_channels', 3)
        self.hparams.setdefault('pred_patch_size', (256, 256))
        self.hparams.setdefault('batch_size', 16)
        self.hparams.setdefault('lr', 1e-4)
        self.hparams.setdefault('num_workers', 4)
        self.hparams.setdefault('background_index', 1)
        self.hparams.setdefault('spatial_dims', 2)
        
        # Metrics
        self.cnfmat = torchmetrics.ConfusionMatrix(
            num_classes=self.hparams.num_classes,
            task='multiclass',
            normalize=None
        )
        
        # Per-class IoU for detailed monitoring
        self.iou_metric = torchmetrics.JaccardIndex(
            task='multiclass',
            num_classes=self.hparams.num_classes,
            average=None
        )
        
        # Model: Configurable EfficientNet encoder with UNet decoder
        encoder_name = self.hparams.get('encoder_name', 'efficientnet-b4')
        pretrained = self.hparams.get('pretrained', True)
        use_attention = self.hparams.get('use_attention', False)
        dropout = self.hparams.get('dropout', 0.3)  # Configurable dropout (V9+)
        
        if use_attention:
            # Custom UNet with Attention Gates
            print(f"Using FlexibleUNetWithAttention (attention gates enabled)")
            self.model = FlexibleUNetWithAttention(
                in_channels=self.hparams.input_channels,
                out_channels=self.hparams.num_classes,
                backbone=encoder_name,
                pretrained=pretrained,
                decoder_channels=(256, 128, 64, 32, 16),
                spatial_dims=self.hparams.spatial_dims,
            )
        else:
            # Standard MONAI FlexibleUNet
            self.model = FlexibleUNet(
                in_channels=self.hparams.input_channels,
                out_channels=self.hparams.num_classes,
                backbone=encoder_name,
                pretrained=pretrained,
                decoder_channels=(256, 128, 64, 32, 16),
                spatial_dims=self.hparams.spatial_dims,
                norm=("batch", {"eps": 1e-3, "momentum": 0.01}),
                act=("relu", {"inplace": True}),
                dropout=dropout,
                decoder_bias=True,
                upsample="nontrainable",
            )
        
        print(f"ThioS Model: {encoder_name} encoder with {'pretrained' if pretrained else 'random'} weights")
        print(f"Dropout: {dropout}, Weight Decay: {self.hparams.get('weight_decay', 0.01)}, Label Smoothing: {self.hparams.get('label_smoothing', 0.0)}")
        print(f"Output classes: {self.hparams.num_classes}")
        print(f"Trainable params: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

    def forward(self, x):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self.model(x.to(device))

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_ds, 
            batch_size=self.hparams.batch_size, 
            shuffle=True,
            collate_fn=list_data_collate, 
            num_workers=self.hparams.num_workers,
            persistent_workers=True, 
            pin_memory=torch.cuda.is_available()
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_ds, 
            batch_size=self.hparams.batch_size, 
            shuffle=False,
            collate_fn=list_data_collate, 
            num_workers=self.hparams.num_workers,
            persistent_workers=True, 
            pin_memory=torch.cuda.is_available()
        )

    def configure_optimizers(self):
        """AdamW optimizer with configurable scheduler."""
        # Get weight decay from config (default 0.01)
        weight_decay = self.hparams.get('weight_decay', 0.01)
        
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.hparams.lr, 
            weight_decay=weight_decay
        )
        
        # Check if we should use ReduceLROnPlateau (for V9+)
        reduce_lr_patience = self.hparams.get('reduce_lr_patience', None)
        
        if reduce_lr_patience is not None:
            # ReduceLROnPlateau: reduce LR when val_loss plateaus
            reduce_lr_factor = self.hparams.get('reduce_lr_factor', 0.5)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=reduce_lr_factor,
                patience=reduce_lr_patience,
                min_lr=1e-7,
                verbose=True
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss',
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
        else:
            # Default: Cosine annealing with warm restarts
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=30,
                T_mult=2,
                eta_min=1e-6
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def loss_function(self, logits, labels, weight_map=None, class_weights=None):
        """
        Combined Focal Loss + Tversky Loss for multiclass segmentation.
        
        IMPORTANT: Properly ignores unlabeled pixels (class 0) by masking them out.
        Only computes loss on pixels that have actual labels (classes 1-4).
        
        - Focal Loss: Handles class imbalance, focuses on hard examples
        - Tversky Loss: Controls precision/recall tradeoff for rare classes
        - Label Smoothing (V9+): Softens one-hot targets to reduce overconfidence
        """
        if labels.ndim == 3:
            labels = labels.unsqueeze(1)
        
        # Create mask for labeled pixels only (exclude class 0 = unlabeled)
        # This is critical: we completely ignore unlabeled pixels in loss computation
        labeled_mask = (labels > 0).float()  # Shape: (B, 1, H, W)
        
        # Check if there are any labeled pixels
        num_labeled = labeled_mask.sum()
        if num_labeled == 0:
            # No labeled pixels in this batch - return zero loss
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        # Label smoothing (V9+): reduce overconfidence
        label_smoothing = self.hparams.get('label_smoothing', 0.0)
        
        # 1. Focal Loss with class weighting (masked)
        focal_fn = FocalLoss(
            to_onehot_y=True,
            gamma=2.0,
            weight=class_weights,
            reduction='none'
        )
        pixel_loss = focal_fn(logits, labels)  # Shape: (B, 1, H, W)
        
        # Apply ignore mask - zero out loss for unlabeled pixels
        pixel_loss_masked = pixel_loss * labeled_mask
        
        # Apply spatial weight map if provided
        if weight_map is not None:
            if weight_map.ndim == 3:
                weight_map = weight_map.unsqueeze(1)
            weight_map = weight_map.to(pixel_loss.device).type_as(pixel_loss)
            pixel_loss_masked = pixel_loss_masked * weight_map
        
        # Average over labeled pixels only
        focal_term = pixel_loss_masked.sum() / (num_labeled + 1e-7)

        # 2. Tversky Loss - masked version for labeled pixels only
        # We need to compute this manually to properly handle the ignore mask
        
        # Convert labels to one-hot, excluding class 0 predictions
        # For Tversky, we compute per-class overlap, but only on labeled regions
        probs = torch.softmax(logits, dim=1)  # (B, C, H, W)
        
        # One-hot encode labels (B, 1, H, W) -> (B, C, H, W)
        labels_one_hot = torch.zeros_like(probs)
        labels_one_hot.scatter_(1, labels.long(), 1)
        
        # Apply label smoothing to one-hot targets (V9+)
        if label_smoothing > 0:
            # Smooth the one-hot labels: (1 - smoothing) for true class, smoothing/(num_classes-1) for others
            # Only smooth labeled pixels (classes 1-4), not class 0
            num_classes = probs.shape[1]
            labels_one_hot_smoothed = labels_one_hot.clone()
            # For labeled pixels only (where labels > 0)
            labeled_pixels = (labels > 0).expand_as(labels_one_hot)
            labels_one_hot_smoothed = torch.where(
                labeled_pixels,
                labels_one_hot * (1 - label_smoothing) + label_smoothing / (num_classes - 1),
                labels_one_hot
            )
            labels_one_hot = labels_one_hot_smoothed
        
        # Mask both predictions and labels to only include labeled pixels
        # Expand mask to all channels
        labeled_mask_expanded = labeled_mask.expand_as(probs)  # (B, C, H, W)
        
        probs_masked = probs * labeled_mask_expanded
        labels_masked = labels_one_hot * labeled_mask_expanded
        
        # Compute Tversky loss per class (excluding class 0)
        smooth = 1e-5
        
        # V12: Configurable Tversky parameters from hparams
        # Default: alpha >= beta to penalize over-segmentation (boundary precision)
        # Previous versions used alpha < beta which encouraged boundary bleeding
        default_tversky_params = {
            1: {'alpha': 0.5, 'beta': 0.5, 'weight': 1.0},   # background: balanced
            2: {'alpha': 0.6, 'beta': 0.4, 'weight': 3.0},   # diffuse: precision-biased
            3: {'alpha': 0.6, 'beta': 0.4, 'weight': 2.0},   # plaque: precision-biased
            4: {'alpha': 0.6, 'beta': 0.4, 'weight': 4.0},   # tangle: precision-biased
        }
        
        # Override from hparams if provided (V12+)
        class_tversky_params = self.hparams.get('tversky_params', None)
        if class_tversky_params is not None:
            # Convert string keys to int keys if from JSON
            parsed = {}
            for k, v in class_tversky_params.items():
                parsed[int(k)] = v
            class_tversky_params = parsed
        else:
            class_tversky_params = default_tversky_params
        
        tversky_per_class = []
        for c in range(1, NUM_CLASSES):  # Skip class 0 (unlabeled)
            p = probs_masked[:, c]  # (B, H, W)
            g = labels_masked[:, c]  # (B, H, W)
            
            # Get class-specific parameters
            params = class_tversky_params[c]
            alpha = params['alpha']
            beta = params['beta']
            weight = params['weight']
            
            # True positives, false positives, false negatives
            tp = (p * g).sum()
            fp = (p * (1 - g) * labeled_mask.squeeze(1)).sum()  # Only count FP in labeled regions
            fn = ((1 - p) * g).sum()
            
            tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
            # Apply class-specific weight
            weighted_loss = weight * (1 - tversky)
            tversky_per_class.append(weighted_loss)
        
        # Weighted average Tversky loss (normalize by sum of weights)
        total_weight = sum(p['weight'] for p in class_tversky_params.values())
        tversky_term = torch.stack(tversky_per_class).sum() / total_weight

        # 3. Boundary Loss (V12+) - penalize predictions that don't align with GT boundaries
        # Uses precomputed distance maps stored in batch
        boundary_weight = self.hparams.get('boundary_loss_weight', 0.0)
        boundary_term = torch.tensor(0.0, device=logits.device)
        
        if boundary_weight > 0 and hasattr(self, '_current_dist_maps') and self._current_dist_maps is not None:
            dist_maps = self._current_dist_maps  # (B, C, H, W)
            # Boundary loss = integral of softmax * distance_map over labeled region
            # Negative distance inside GT → reward correct predictions
            # Positive distance outside GT → penalize boundary overshoot
            for c in range(1, NUM_CLASSES):
                p_c = probs_masked[:, c]  # (B, H, W)
                d_c = dist_maps[:, c]     # (B, H, W)
                # Normalize distance to [-1, 1] range
                d_max = d_c.abs().max() + 1e-7
                d_norm = d_c / d_max
                boundary_term = boundary_term + (p_c * d_norm * labeled_mask.squeeze(1)).mean()
            boundary_term = boundary_term / (NUM_CLASSES - 1)

        # Combined loss: configurable weights (V12+)
        focal_weight_cfg = self.hparams.get('focal_weight', 0.4)
        tversky_weight_cfg = self.hparams.get('tversky_weight', 0.6)
        
        total_loss = focal_weight_cfg * focal_term + tversky_weight_cfg * tversky_term
        if boundary_weight > 0:
            total_loss = total_loss + boundary_weight * boundary_term
        
        return total_loss

    def training_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        weight_map = batch.get("weight", None)
        
        logits = self.forward(images)
        
        # Compute dynamic class weights from batch
        device = logits.device
        class_weights = get_class_weights(labels, self.hparams.num_classes, device)
        
        # V12: Compute boundary distance maps on-the-fly if boundary loss enabled
        self._current_dist_maps = None
        if self.hparams.get('boundary_loss_weight', 0.0) > 0:
            dist_maps = []
            for b in range(labels.shape[0]):
                lbl_np = labels[b].squeeze().cpu().numpy().astype(np.uint8)
                dm = compute_boundary_distance_map(lbl_np)
                dist_maps.append(dm)
            self._current_dist_maps = torch.from_numpy(np.stack(dist_maps)).to(device)
        
        loss = self.loss_function(logits, labels, weight_map=weight_map, class_weights=class_weights)
        
        self.log("train_loss", loss.item(), on_step=False, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        weight_map = batch.get("weight", None)
        
        logits = self.forward(images)
        
        device = logits.device
        class_weights = get_class_weights(labels, self.hparams.num_classes, device)
        
        # V12: Compute boundary distance maps for validation loss
        self._current_dist_maps = None
        if self.hparams.get('boundary_loss_weight', 0.0) > 0:
            dist_maps = []
            for b in range(labels.shape[0]):
                lbl_np = labels[b].squeeze().cpu().numpy().astype(np.uint8)
                dm = compute_boundary_distance_map(lbl_np)
                dist_maps.append(dm)
            self._current_dist_maps = torch.from_numpy(np.stack(dist_maps)).to(device)
        
        loss = self.loss_function(logits, labels, weight_map=weight_map, class_weights=class_weights)
        
        # Update metrics — apply confidence thresholding if configured
        probs = torch.softmax(logits, dim=1)
        conf_thresh = self.hparams.get('confidence_threshold', 0.0)
        if conf_thresh > 0:
            # Only assign foreground class if max foreground prob > threshold
            # Otherwise assign background (class 1)
            max_fg_prob, _ = probs[:, 1:, :, :].max(dim=1)  # max over classes 1-4
            preds = torch.argmax(probs, dim=1)
            # Where max foreground prob < threshold AND prediction is foreground, force to background
            low_conf_fg = (max_fg_prob < conf_thresh) & (preds > 1)
            preds[low_conf_fg] = 1  # Set to background
        else:
            preds = torch.argmax(logits, dim=1)
        labels_flat = labels.squeeze(1) if labels.ndim == 4 else labels
        
        # IMPORTANT: Exclude class 0 (unlabeled) pixels from metrics
        # Only include pixels where ground truth is labeled (classes 1-4)
        valid_mask = labels_flat != 0
        if valid_mask.any():
            preds_valid = preds[valid_mask]
            labels_valid = labels_flat[valid_mask]
            self.cnfmat(preds_valid.view(-1), labels_valid.view(-1))
            self.iou_metric(preds[valid_mask.unsqueeze(0) if preds.dim() > labels_flat.dim() else valid_mask], 
                           labels_flat[valid_mask])
        
        self.log("val_loss", loss.item(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return {"loss": loss, "logits": logits, "labels": labels}

    def on_validation_epoch_end(self):
        """Compute and log validation metrics."""
        # Confusion matrix stats
        cnfmat = self.cnfmat.compute()
        
        # Since we excluded class 0 from ground truth during accumulation,
        # we work with the submatrix for classes 1-4 only
        # But predictions can still be 0-4, so we need to handle this correctly
        
        # Per-class metrics (for classes 1-4)
        true_pos = torch.diag(cnfmat)
        false_pos = cnfmat.sum(0) - true_pos  # Column sums - diagonal
        false_neg = cnfmat.sum(1) - true_pos  # Row sums - diagonal
        
        # Only use classes 1-4 for all calculations
        valid_classes = list(range(1, self.hparams.num_classes))
        
        # Overall accuracy (only on classes 1-4, ignoring predictions of 0)
        # Accuracy = correctly classified / total labeled pixels
        valid_preds = cnfmat[1:, 1:].sum()  # Predictions 1-4 for labels 1-4
        correct_preds = sum(true_pos[c] for c in valid_classes)
        total_labeled = cnfmat[1:, :].sum()  # All labeled pixels (rows 1-4)
        acc = correct_preds / (total_labeled + 1e-7)
        
        # Macro-averaged precision, recall, IoU (classes 1-4 only)
        eps = 1e-7
        precision_per_class = true_pos / (true_pos + false_pos + eps)
        recall_per_class = true_pos / (true_pos + false_neg + eps)
        iou_per_class = true_pos / (true_pos + false_pos + false_neg + eps)
        
        precision = precision_per_class[valid_classes].mean()
        recall = recall_per_class[valid_classes].mean()
        iou = iou_per_class[valid_classes].mean()
        dice = 2 * true_pos[valid_classes].sum() / (
            2 * true_pos[valid_classes].sum() + 
            false_pos[valid_classes].sum() + 
            false_neg[valid_classes].sum() + eps
        )
        
        # Log metrics
        self.log('val_acc', acc, prog_bar=True, sync_dist=True)
        self.log('val_precision', precision, sync_dist=True)
        self.log('val_recall', recall, sync_dist=True)
        self.log('val_iou', iou, prog_bar=True, sync_dist=True)
        self.log('val_dice', dice, prog_bar=True, sync_dist=True)
        
        # Log per-class IoU
        for c in valid_classes:
            self.log(f'val_iou_{CLASS_NAMES[c]}', iou_per_class[c], sync_dist=True)

        # Rare-class diagnostics (diffuse/plaque/tangle)
        rare_classes = [2, 3, 4]
        rare_iou_mean = iou_per_class[rare_classes].mean()
        rare_recall_mean = recall_per_class[rare_classes].mean()
        rare_precision_mean = precision_per_class[rare_classes].mean()

        # Predicted fraction per rare class among labeled pixels (rows 1-4)
        # Helps detect "collapse to background" despite high overall accuracy.
        total_labeled = cnfmat[1:, :].sum() + eps
        pred_frac_diffuse = cnfmat[:, 2].sum() / total_labeled
        pred_frac_plaque = cnfmat[:, 3].sum() / total_labeled
        pred_frac_tangle = cnfmat[:, 4].sum() / total_labeled

        self.log('val_rare_iou_mean', rare_iou_mean, prog_bar=True, sync_dist=True)
        self.log('val_rare_recall_mean', rare_recall_mean, sync_dist=True)
        self.log('val_rare_precision_mean', rare_precision_mean, sync_dist=True)
        self.log('val_pred_frac_diffuse', pred_frac_diffuse, sync_dist=True)
        self.log('val_pred_frac_plaque', pred_frac_plaque, sync_dist=True)
        self.log('val_pred_frac_tangle', pred_frac_tangle, sync_dist=True)
        
        # Print to console
        print(f"\n[Epoch {self.current_epoch}] Validation Metrics:")
        print(f"  Acc: {acc:.4f} | Prec: {precision:.4f} | Recall: {recall:.4f}")
        print(f"  IoU: {iou:.4f} | Dice: {dice:.4f}")
        print(f"  Rare(mean): IoU={rare_iou_mean:.4f} | Recall={rare_recall_mean:.4f} | Precision={rare_precision_mean:.4f}")
        print(f"  Pred frac: diffuse={pred_frac_diffuse:.4f} plaque={pred_frac_plaque:.4f} tangle={pred_frac_tangle:.4f}")
        print(f"  Per-class IoU: ", end="")
        for c in valid_classes:
            print(f"{CLASS_NAMES[c]}={iou_per_class[c]:.4f} ", end="")
        print()
        
        # Reset metrics
        self.cnfmat.reset()
        self.iou_metric.reset()

    def predict_step(self, batch, batch_idx, use_tta=True):
        """
        Prediction with Test-Time Augmentation (TTA).
        """
        images = batch['image']
        
        if use_tta:
            all_probs = []
            
            # Original
            all_probs.append(torch.softmax(self(images), dim=1))
            
            # Horizontal flip
            flipped_h = torch.flip(images, dims=[3])
            probs_h = torch.softmax(self(flipped_h), dim=1)
            all_probs.append(torch.flip(probs_h, dims=[3]))
            
            # Vertical flip
            flipped_v = torch.flip(images, dims=[2])
            probs_v = torch.softmax(self(flipped_v), dim=1)
            all_probs.append(torch.flip(probs_v, dims=[2]))
            
            # 90° rotation
            rotated_90 = torch.rot90(images, k=1, dims=[2, 3])
            probs_90 = torch.softmax(self(rotated_90), dim=1)
            all_probs.append(torch.rot90(probs_90, k=-1, dims=[2, 3]))
            
            # 180° rotation
            rotated_180 = torch.rot90(images, k=2, dims=[2, 3])
            probs_180 = torch.softmax(self(rotated_180), dim=1)
            all_probs.append(torch.rot90(probs_180, k=-2, dims=[2, 3]))
            
            # 270° rotation
            rotated_270 = torch.rot90(images, k=3, dims=[2, 3])
            probs_270 = torch.softmax(self(rotated_270), dim=1)
            all_probs.append(torch.rot90(probs_270, k=-3, dims=[2, 3]))
            
            probs = torch.stack(all_probs, dim=0).mean(dim=0)
        else:
            probs = torch.softmax(self(images), dim=1)
        
        # Apply confidence thresholding if configured
        conf_thresh = self.hparams.get('confidence_threshold', 0.0)
        if conf_thresh > 0:
            max_fg_prob, _ = probs[:, 1:, :, :].max(dim=1)  # max over classes 1-4
            preds = torch.argmax(probs, dim=1)
            low_conf_fg = (max_fg_prob < conf_thresh) & (preds > 1)
            preds[low_conf_fg] = 1  # Force low-confidence foreground to background
        else:
            preds = torch.argmax(probs, dim=1)
        
        return probs, preds, batch['image_meta_dict']['filename_or_obj']
