"""
VCoMatcher Phase 2: Dataset with Target-Centric Coordinate Transformation
=========================================================================

This module implements the DataLoader for Phase 2 training, with strict adherence
to the coordinate normalization strategy described in Page 13-14.

Key Features:
1. Load Phase 1 .npz data (VGGT pseudo-GT or COLMAP)
2. Online Target-Centric Transformation: Transform all poses and point maps to Target frame
3. Curriculum Learning: Mix COLMAP (easy/standard) and VGGT (hard/extreme) samples
4. Return complete batch structure as per Page 14

Author: VCoMatcher Team
Date: 2025-12-12
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random


def load_and_preprocess_image(
    image_path: Path,
    target_size: int = 518,
) -> torch.Tensor:
    """
    Load and preprocess a single image using OpenCV (consistent with Phase 1 PnP).
    
    CRITICAL FIX (2025-12-21): 
    - Changed from PIL to OpenCV for consistency with Phase 1 PnP refinement
    - Changed from "Pad then Resize" to "Resize then Pad" (standard practice)
    
    This function MUST match VGGT's preprocessing exactly to ensure
    geometric alignment with the 3D points (Pseudo-GT) generated in Phase 1.
    
    Steps (matching VGGT + OpenCV):
    1. Load image with cv2
    2. Handle RGBA → RGB (blend with white background if needed)
    3. Resize LONGER edge to target_size (maintaining aspect ratio)
    4. Center padding shorter edge to square
    5. Convert to tensor [3, H, W] in range [0, 1]
    
    Args:
        image_path: Path to image file
        target_size: Target resolution (default: 518, must match Phase 1)
        
    Returns:
        Preprocessed image tensor [3, H, W] in range [0, 1]
    """
    import cv2
    import numpy as np
    
    # BUGFIX: Validate image path exists before attempting to load
    if not image_path.exists():
        raise FileNotFoundError(f"Image file does not exist: {image_path}")
    
    # Step 1: Load image with OpenCV (BGR format)
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    
    if img is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")
    
    # BUGFIX: Handle grayscale images (2D array)
    if len(img.shape) == 2:
        # Grayscale image - convert to RGB by repeating channels
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # Step 2: Handle RGBA → RGB (only if 4 channels)
    elif img.shape[2] == 4:  # RGBA (OpenCV loads as BGRA)
        # Blend with white background first
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        bgr = img[:, :, :3].astype(np.float32)  # Still in BGR order
        white_bg = np.ones_like(bgr) * 255.0
        blended = (alpha * bgr + (1 - alpha) * white_bg).astype(np.uint8)
        # CRITICAL FIX: Convert BGR to RGB after alpha blending
        img = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
    elif img.shape[2] == 3:
        # BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"Unexpected image format with {img.shape[2]} channels at {image_path}")

    
    height, width = img.shape[:2]
    
    # Step 3: Resize longer edge to target_size (maintaining aspect ratio)
    scale = target_size / max(height, width)
    new_height = int(height * scale)
    new_width = int(width * scale)
    
    # CRITICAL FIX: Use INTER_LINEAR (not CUBIC) for consistency with VGGT
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    # Step 4: Center padding to square
    pad_height = target_size - new_height
    pad_width = target_size - new_width
    
    top = pad_height // 2
    bottom = pad_height - top
    left = pad_width // 2
    right = pad_width - left
    
    # Pad with black (0, 0, 0)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, 
                             cv2.BORDER_CONSTANT, value=(0, 0, 0))
    
    # Step 5: Convert to tensor [3, H, W] in range [0, 1]
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    
    return img_tensor


class VCoMatcherDataset(Dataset):
    """
    VCoMatcher Dataset for Phase 2 training.
    
    Implements:
    - Online coordinate transformation (Page 13-14: Target-Centric Canonicalization)
    - Mixed sampling (COLMAP + VGGT, Page 12-13)
    - Complete batch structure (Page 14)
    """
    
    def __init__(
        self,
        data_paths: List[Path],
        sample_types: List[str] = ["easy", "hard"],
        augmentation: bool = False,
        cache_data: bool = False,  # CRITICAL: Default False to avoid OOM on large datasets
    ):
        """
        Initialize VCoMatcher Dataset.
        
        Args:
            data_paths: List of paths to .npz files from Phase 1
            sample_types: Types of samples to include ['easy', 'hard', 'extreme']
            augmentation: Whether to apply data augmentation
            cache_data: Whether to cache loaded .npz files in memory
        """
        self.data_paths = [Path(p) for p in data_paths]
        self.sample_types = sample_types
        self.augmentation = augmentation
        self.cache_data = cache_data
        
        # Build sample index
        self.samples = []
        self.data_cache = {}
        
        print(f"Building sample index from {len(self.data_paths)} data files...")
        self._build_sample_index()
        print(f"Total samples: {len(self.samples)}")
        
    def _build_sample_index(self):
        """Build index of all valid samples across all data files."""
        for data_path in self.data_paths:
            # CRITICAL FIX: Only load samples for indexing, not full data
            # This prevents OOM when building index
            with np.load(data_path, allow_pickle=True) as data:
                samples = data["samples"]
                scene_name = data_path.stem
                
                # Filter by sample type
                for sample in samples:
                    if sample["sample_type"] in self.sample_types:
                        self.samples.append({
                            "data_path": str(data_path),
                            "scene_name": scene_name,
                            "source_idx": sample["source_idx"],
                            "target_idx": sample["target_idx"],
                            "overlap_score": sample["overlap_score"],
                            "sample_type": sample["sample_type"],
                        })
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def _load_data(self, data_path: str) -> Dict:
        """Load .npz data from disk or cache (lazy loading)."""
        if self.cache_data and data_path in self.data_cache:
            return self.data_cache[data_path]
        else:
            # BUGFIX: Add error handling for corrupted or missing files
            try:
                # BUGFIX: Use allow_pickle=True and keep file open
                # np.load returns NpzFile which is a lazy loader
                # We need to either cache it or load all arrays immediately
                data = np.load(data_path, allow_pickle=True)
                
                if self.cache_data:
                    # Cache the NpzFile object for future use
                    self.data_cache[data_path] = data
                    return data
                else:
                    # BUGFIX: When not caching, we need to load all arrays into memory
                    # because NpzFile closes when going out of scope
                    # Convert to dict to keep data accessible
                    data_dict = {key: data[key] for key in data.files}
                    data.close()
                    return data_dict
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load data from {data_path}\n"
                    f"Error: {str(e)}\n"
                    f"Please verify the file is not corrupted and was generated with Phase 1."
                ) from e
    
    def _compute_target_centric_transform(
        self,
        extrinsic: np.ndarray,
        points_3d: np.ndarray,
        target_idx: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute target-centric coordinate transformation (Page 13-14).
        
        This is the CRITICAL step for Phase 2: Transform all data from world coordinates
        to Target camera coordinates.
        
        Mathematical Definition (Page 13):
        1. M_anchor = (T_target^world)^(-1)  -- World-to-Target transform
        2. T_new^(k) = M_anchor @ T_world^(k)  -- Transform all poses
        3. P_new(u,v) = R_anchor @ P_world(u,v) + t_anchor  -- Transform all points
        
        Args:
            extrinsic: [N, 4, 4] world-to-camera extrinsic matrices
            points_3d: [N, H, W, 3] world coordinate points
            target_idx: Index of target view
            
        Returns:
            Tuple of:
                - extrinsic_new: [N, 4, 4] target-centric extrinsic matrices
                - points_3d_new: [N, H, W, 3] target-centric point maps
                - M_anchor: [4, 4] anchor transformation matrix
        """
        # Step 1: Compute anchor transform (Page 13)
        # M_anchor = inv(T_target^world)
        T_target_world = extrinsic[target_idx]  # [4, 4]
        M_anchor = np.linalg.inv(T_target_world)  # [4, 4]
        
        # Verification checkpoint (Page 13): Target's new pose should be Identity
        T_target_new = M_anchor @ T_target_world
        
        # BUGFIX: Use more lenient tolerance and provide detailed error message
        if not np.allclose(T_target_new, np.eye(4), atol=1e-3):
            max_error = np.abs(T_target_new - np.eye(4)).max()
            print(f"  ⚠️  Warning: Target pose transformation has error {max_error:.6f}")
            print(f"     T_target_new:\n{T_target_new}")
            
            # If error is very large, this is a critical bug
            if max_error > 1e-2:
                raise ValueError(
                    f"Target pose transformation failed! Max error: {max_error:.6f}\n"
                    f"Got:\n{T_target_new}\nExpected Identity"
                )
        
        # Step 2: Transform all poses (Page 13)
        # T_new^(k) = M_anchor @ T_world^(k)
        # 
        # CRITICAL: This formula matches how points are transformed: P_new = M_anchor @ P_world
        # This maintains geometric consistency
        N = extrinsic.shape[0]
        extrinsic_new = np.zeros_like(extrinsic)
        
        for k in range(N):
            extrinsic_new[k] = M_anchor @ extrinsic[k]
        
        # Step 3: Transform all point maps (Page 13)
        # P_new(u,v) = R_anchor @ P_world(u,v) + t_anchor
        R_anchor = M_anchor[:3, :3]  # [3, 3]
        t_anchor = M_anchor[:3, 3]   # [3]
        
        N, H, W, _ = points_3d.shape
        points_3d_flat = points_3d.reshape(N, -1, 3)  # [N, H*W, 3]
        
        # Apply rigid transformation
        points_3d_new_flat = np.einsum('ij,npj->npi', R_anchor, points_3d_flat) + t_anchor  # [N, H*W, 3]
        points_3d_new = points_3d_new_flat.reshape(N, H, W, 3)  # [N, H, W, 3]
        
        return extrinsic_new, points_3d_new, M_anchor
    
    def _sample_source_views(
        self,
        data: Dict,
        target_idx: int,
        primary_source_idx: int,
        min_overlap: float = 0.05,
    ) -> List[int]:
        """
        Sample 3 source views for true multi-view matching.
        
        CRITICAL FIX: Replaces fake multi-view sampling (repeating same source 3x).
        
        Strategy:
        1. First source: primary_source_idx (the hard/extreme sample core)
        2. Additional 2 sources: randomly sampled from views with overlap > min_overlap
        3. Fallback: if insufficient views, repeat primary_source_idx
        
        Args:
            data: Loaded .npz data (dict or NpzFile)
            target_idx: Target view index
            primary_source_idx: Primary source (from hard/extreme sample)
            min_overlap: Minimum overlap threshold for valid sources
            
        Returns:
            List of 3 source view indices
        """
        overlap_matrix = np.array(data["overlap_matrix"])  # [N, N] - ensure it's a copy
        N = overlap_matrix.shape[0]
        
        # BUGFIX: Guard against insufficient images in scene
        if N < 2:
            # Only 1 image - cannot form multi-view sample
            raise ValueError(f"Scene has only {N} images, need at least 2 for matching")
        
        # Get all views with sufficient overlap with target
        overlaps = overlap_matrix[target_idx]  # [N]
        valid_sources = np.where(overlaps > min_overlap)[0]
        
        # Remove target itself
        valid_sources = valid_sources[valid_sources != target_idx]
        
        # BUGFIX: Ensure primary_source_idx is valid and within bounds
        if primary_source_idx >= N:
            raise ValueError(f"primary_source_idx ({primary_source_idx}) >= N ({N})")
        
        if primary_source_idx not in valid_sources:
            # Primary source doesn't meet overlap threshold - force include it
            # This can happen at scene boundaries with hard/extreme samples
            # BUGFIX: Use np.concatenate instead of np.append to ensure 1D array
            valid_sources = np.concatenate([valid_sources, [primary_source_idx]])
        
        # Ensure primary_source_idx is included as 1st source
        # BUGFIX: Ensure all indices are Python int (not numpy types)
        source_views = [int(primary_source_idx)]
        
        # Sample 2 additional sources
        other_sources = valid_sources[valid_sources != primary_source_idx]
        
        if len(other_sources) >= 2:
            # Randomly sample 2 additional views (without replacement)
            additional = np.random.choice(other_sources, size=2, replace=False)
            # BUGFIX: Convert numpy int64 to Python int for JSON serialization compatibility
            source_views.extend([int(x) for x in additional])
        elif len(other_sources) == 1:
            # Only 1 additional view available
            source_views.append(int(other_sources[0]))
            source_views.append(int(primary_source_idx))  # Repeat primary as fallback
        else:
            # No additional views available, repeat primary_source
            source_views.extend([int(primary_source_idx), int(primary_source_idx)])
        
        assert len(source_views) == 3, f"Expected 3 source views, got {len(source_views)}"
        return source_views
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training sample with target-centric coordinate transformation.
        
        Returns batch structure as per Page 14:
        - images: [4, 3, H, W] -- [Target, Source1, Source2, Source3]
        - intrinsic: [4, 3, 3] -- Camera intrinsics
        - extrinsic_rel: [4, 4, 4] -- Target-centric poses (T_new)
        - depth: [4, H, W] -- Depth maps
        - points_3d: [4, H, W, 3] -- Target-centric 3D points
        - depth_conf: [4, H, W] -- Depth confidence (σ_D)
        - points_conf: [4, H, W] -- Point confidence (σ_P)
        - mask_geom: [4, H, W] -- Loose mask for GNN graph construction
        - mask_loss: [4, H, W] -- Strict mask for loss computation
        - uncertainty_map: [4, H, W] -- Uncertainty map (σ_P or σ_D)
        - sample_type: str -- 'easy', 'hard', or 'extreme'
        - overlap_score: float -- Overlap ratio O_ij
        """
        sample = self.samples[idx]
        
        # Load data
        data = self._load_data(sample["data_path"])
        
        # Get source and target indices
        source_idx = sample["source_idx"]
        target_idx = sample["target_idx"]
        
        # CRITICAL FIX: True multi-view sampling instead of fake repetition
        # Sample 3 diverse source views using overlap matrix
        source_views = self._sample_source_views(
            data=data,
            target_idx=target_idx,
            primary_source_idx=source_idx,
        )
        
        # view_indices: [target, source1, source2, source3]
        view_indices = [target_idx] + source_views
        
        # Extract data for selected views
        extrinsic_raw = data["extrinsic"][view_indices]  # [4, 3, 4] or [4, 4, 4]
        intrinsic = data["intrinsic"][view_indices]  # [4, 3, 3]
        depth = data["depth"][view_indices]  # [4, H, W]
        points_3d_world = data["points_3d"][view_indices]  # [4, H, W, 3]
        depth_conf = data["depth_conf"][view_indices]  # [4, H, W]
        points_conf = data["points_conf"][view_indices]  # [4, H, W]
        mask_geom = data["mask_geom"][view_indices]  # [4, H, W]
        mask_loss = data["mask_loss"][view_indices]  # [4, H, W]
        
        # BUGFIX: Make copies to avoid modifying cached data
        extrinsic_raw = np.array(extrinsic_raw)
        intrinsic = np.array(intrinsic)
        depth = np.array(depth)
        points_3d_world = np.array(points_3d_world)
        depth_conf = np.array(depth_conf)
        points_conf = np.array(points_conf)
        mask_geom = np.array(mask_geom)
        mask_loss = np.array(mask_loss)
        
        # Convert extrinsic to homogeneous form if needed (VGGT outputs [R|t] format)
        if extrinsic_raw.shape[-2] == 3:  # [N, 3, 4] format
            # Convert to [N, 4, 4] by adding [0, 0, 0, 1] row
            N = extrinsic_raw.shape[0]
            extrinsic_world = np.zeros((N, 4, 4), dtype=extrinsic_raw.dtype)
            extrinsic_world[:, :3, :] = extrinsic_raw  # Copy [R|t]
            extrinsic_world[:, 3, 3] = 1.0  # Add bottom row [0, 0, 0, 1]
        else:
            extrinsic_world = extrinsic_raw
        
        # ==================== CRITICAL: Target-Centric Transformation ====================
        # Page 13-14: Transform all data from world coordinates to Target camera coordinates
        extrinsic_rel, points_3d_rel, M_anchor = self._compute_target_centric_transform(
            extrinsic=extrinsic_world,
            points_3d=points_3d_world,
            target_idx=0,  # Target is always at index 0
        )
        
        # Verification: Target pose should be Identity
        # BUGFIX: Use more lenient tolerance for float32 precision
        if not np.allclose(extrinsic_rel[0], np.eye(4), atol=1e-3):
            max_error = np.abs(extrinsic_rel[0] - np.eye(4)).max()
            raise ValueError(
                f"Target pose is not Identity after transformation! Max error: {max_error:.6f}\n"
                f"Got:\n{extrinsic_rel[0]}"
            )
        
        # ==================== Prepare Output Batch ====================
        # Convert to torch tensors
        batch = {
            # Geometry
            "intrinsic": torch.from_numpy(intrinsic).float(),  # [4, 3, 3]
            "extrinsic_rel": torch.from_numpy(extrinsic_rel).float(),  # [4, 4, 4]
            "depth": torch.from_numpy(depth).float(),  # [4, H, W]
            "points_3d": torch.from_numpy(points_3d_rel).float(),  # [4, H, W, 3]
            
            # Confidence / Uncertainty
            "depth_conf": torch.from_numpy(depth_conf).float(),  # [4, H, W]
            "points_conf": torch.from_numpy(points_conf).float(),  # [4, H, W]
            "uncertainty_map": torch.from_numpy(points_conf).float(),  # Use σ_P as uncertainty
            
            # Dual Masks (Critical for Phase 2 & 3)
            "mask_geom": torch.from_numpy(mask_geom).bool(),  # [4, H, W] -- For GNN
            "mask_loss": torch.from_numpy(mask_loss).bool(),  # [4, H, W] -- For Loss
            
            # Metadata
            "sample_type": sample["sample_type"],  # 'easy', 'hard', or 'extreme'
            "overlap_score": float(sample["overlap_score"]),
            "scene_name": sample["scene_name"],
            "view_indices": view_indices,
        }
        
        # ==================== CRITICAL: Image Loading ====================
        # Load images using Phase 1's saved image_paths
        # MUST use exact same preprocessing as Phase 1 for geometric alignment
        
        # Check if image_paths exist in .npz file
        if "image_paths" not in data:
            raise ValueError(
                f"No 'image_paths' found in {sample['data_path']}. "
                "Please regenerate Phase 1 data with updated vcomatcher_phase1_data_engine.py"
            )
        
        image_paths_all = data["image_paths"]  # [N] array of paths
        resolution = int(data.get("resolution", 518))  # Get resolution from Phase 1
        
        images_list = []
        for view_idx in view_indices:  # [target, src1, src2, src3]
            img_path = Path(str(image_paths_all[view_idx]))
            
            # Check if image path exists
            if not img_path.exists():
                raise FileNotFoundError(
                    f"Image not found: {img_path}\n"
                    f"Expected from view_indices={view_indices}, current idx={view_idx}"
                )
            
            # Load and preprocess with EXACT VGGT logic
            img_tensor = load_and_preprocess_image(img_path, target_size=resolution)
            images_list.append(img_tensor)
        
        images = torch.stack(images_list)  # [4, 3, H, W]
        batch["images"] = images
        
        return batch


class MixedDataLoader:
    """
    Mixed DataLoader for curriculum learning (Page 12-13).
    
    Dynamically mixes COLMAP (easy/standard) and VGGT (hard/extreme) samples
    based on training epoch.
    """
    
    def __init__(
        self,
        colmap_data_paths: List[Path],
        vggt_data_paths: List[Path],
        batch_size: int = 8,
        num_workers: int = 4,
        curriculum_schedule: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize Mixed DataLoader.
        
        Args:
            colmap_data_paths: Paths to COLMAP data (.npz files)
            vggt_data_paths: Paths to VGGT pseudo-GT data (.npz files)
            batch_size: Batch size
            num_workers: Number of data loading workers
            curriculum_schedule: Schedule for mixing ratios
                Format: {epoch: P_vggt}
                Example: {0: 0.0, 6: 0.0, 20: 0.5, 50: 0.6}
        """
        self.colmap_data_paths = colmap_data_paths
        self.vggt_data_paths = vggt_data_paths
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Default curriculum schedule (Page 13)
        self.curriculum_schedule = curriculum_schedule or {
            0: 0.0,   # Warm-up: 0% VGGT (all COLMAP)
            6: 0.0,   # Warm-up end
            20: 0.5,  # Ramping: 50% VGGT
            50: 0.6,  # Stable: 60% VGGT
        }
        
        # Create datasets
        self.colmap_dataset = VCoMatcherDataset(
            data_paths=colmap_data_paths,
            sample_types=["easy"],  # COLMAP only has 'easy' samples
            cache_data=False,
        )
        
        self.vggt_dataset = VCoMatcherDataset(
            data_paths=vggt_data_paths,
            sample_types=["hard", "extreme"],  # VGGT has 'hard' and 'extreme'
            cache_data=False,
        )
        
        print(f"Mixed DataLoader initialized:")
        print(f"  COLMAP samples: {len(self.colmap_dataset)}")
        print(f"  VGGT samples:   {len(self.vggt_dataset)}")
    
    def get_dataloader(self, epoch: int) -> DataLoader:
        """
        Get DataLoader for a specific epoch with curriculum learning.
        
        Args:
            epoch: Current training epoch
            
        Returns:
            DataLoader with mixed samples
        """
        # Determine P_vggt based on curriculum schedule
        P_vggt = 0.0
        for epoch_threshold in sorted(self.curriculum_schedule.keys()):
            if epoch >= epoch_threshold:
                P_vggt = self.curriculum_schedule[epoch_threshold]
        
        print(f"Epoch {epoch}: P_vggt = {P_vggt:.2%}")
        
        # Create mixed dataset
        mixed_dataset = self._create_mixed_dataset(P_vggt)
        
        # Create DataLoader
        dataloader = DataLoader(
            mixed_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        
        return dataloader
    
    def _create_mixed_dataset(self, P_vggt: float) -> Dataset:
        """
        Create a mixed dataset by sampling from COLMAP and VGGT.
        
        Args:
            P_vggt: Probability of sampling from VGGT dataset
            
        Returns:
            Mixed dataset
        """
        # Simple implementation: Create a wrapper that samples from both datasets
        # A more sophisticated implementation would use torch.utils.data.ConcatDataset
        # with custom sampling logic
        
        class MixedDataset(Dataset):
            def __init__(self, colmap_ds, vggt_ds, p_vggt):
                self.colmap_ds = colmap_ds
                self.vggt_ds = vggt_ds
                self.p_vggt = p_vggt
                
                # Estimate total length
                self.length = len(colmap_ds) + len(vggt_ds)
                
                # BUGFIX: Validate datasets are not both empty
                if len(colmap_ds) == 0 and len(vggt_ds) == 0:
                    raise ValueError("Both COLMAP and VGGT datasets are empty!")
            
            def __len__(self):
                return self.length
            
            def __getitem__(self, idx):
                # BUGFIX: Handle edge cases where one dataset is empty
                colmap_available = len(self.colmap_ds) > 0
                vggt_available = len(self.vggt_ds) > 0
                
                # Randomly choose dataset based on P_vggt
                if vggt_available and (random.random() < self.p_vggt or not colmap_available):
                    # Sample from VGGT
                    vggt_idx = random.randint(0, len(self.vggt_ds) - 1)
                    return self.vggt_ds[vggt_idx]
                elif colmap_available:
                    # Sample from COLMAP
                    colmap_idx = random.randint(0, len(self.colmap_ds) - 1)
                    return self.colmap_ds[colmap_idx]
                else:
                    # This should never happen due to __init__ check
                    raise RuntimeError("No data available in either dataset")
        
        return MixedDataset(self.colmap_dataset, self.vggt_dataset, P_vggt)


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for VCoMatcher.
    
    Handles variable-size inputs and metadata.
    """
    # Stack tensors
    batch_dict = {}
    
    for key in batch[0].keys():
        if isinstance(batch[0][key], torch.Tensor):
            batch_dict[key] = torch.stack([b[key] for b in batch])
        elif isinstance(batch[0][key], (int, float)):
            batch_dict[key] = torch.tensor([b[key] for b in batch])
        else:
            # Keep as list for strings and other types
            batch_dict[key] = [b[key] for b in batch]
    
    return batch_dict


def compute_source_aware_weights(
    batch: Dict[str, torch.Tensor],
    tau_min: float = 1.0,   # VGGT minimum (1 + exp(x))
    tau_max: float = 20.0,  # CRITICAL: Must match Phase 1 tau_uncertainty!
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """
    Compute source-aware weights W_src as per Page 9.
    
    CRITICAL PARAMETER ALIGNMENT (2025-12-22):
    ==========================================
    tau_max MUST match Phase 1's tau_uncertainty filter threshold!
    
    Phase 1 typically uses tau_uncertainty=8.0~15.0 to filter unreliable data.
    This means uncertainty values in Phase 2 data are bounded: σ ∈ [1.0, ~15.0]
    
    ❌ Previous tau_max=100.0 (WRONG):
       - unc=1.0  → weight=1.0   ✓
       - unc=15.0 → weight=0.86  ✗ (too high! noise not suppressed)
       - Weight range: [0.86, 1.0] (no discrimination)
    
    ✅ Current tau_max=20.0 (CORRECT):
       - unc=1.0  → weight=1.0   ✓ (perfect data)
       - unc=5.0  → weight=0.79  ✓ (good data)
       - unc=10.0 → weight=0.53  ✓ (moderate data)
       - unc=15.0 → weight=0.26  ✓ (suppress noise)
       - unc=20.0 → weight=0.0   ✓ (reject)
       - Weight range: [0, 1.0] (full discrimination)
    
    ⚠️ Configuration Requirement:
       Set tau_max = Phase1_tau_uncertainty * 1.3~1.5
       Example: If Phase 1 uses tau_uncertainty=15.0, use tau_max=20.0
    
    Formula:
        W_src(u) = {
            1.0                                     if source is COLMAP
            1 - (σ(u) - τ_min) / (τ_max - τ_min)  if source is VGGT
        }
    
    Args:
        batch: Batch dictionary from dataloader
        tau_min: Minimum uncertainty (σ_min) = 1.0 for VGGT's 1+exp(x) activation
        tau_max: Maximum uncertainty (σ_max) = 20.0
                 ⚠️ MUST align with Phase 1's tau_uncertainty!
        epsilon: Numerical stability constant
        
    Returns:
        W_src: [B, 4, H, W] source-aware weights in range [0, 1]
    """
    # BUGFIX: Handle both batched and single-sample cases
    sample_types = batch["sample_type"]
    if isinstance(sample_types, str):
        # Single sample case
        B = 1
        sample_types = [sample_types]
    else:
        B = len(sample_types)
    
    # Get spatial dimensions from depth tensor
    depth_shape = batch["depth"].shape
    if len(depth_shape) == 4:  # [B, 4, H, W]
        H, W = depth_shape[2], depth_shape[3]
    elif len(depth_shape) == 3:  # [4, H, W] single sample
        H, W = depth_shape[1], depth_shape[2]
    else:
        raise ValueError(f"Unexpected depth shape: {depth_shape}")
    
    # Get device from batch tensors
    # BUGFIX: Handle case where tensors might be on CPU
    device = batch["depth"].device if torch.is_tensor(batch["depth"]) else torch.device("cpu")
    
    W_src = torch.zeros(B, 4, H, W, dtype=torch.float32, device=device)
    
    for b in range(B):
        sample_type = sample_types[b]
        
        # Handle both batched and single-sample indexing
        if B == 1 and len(batch["mask_loss"].shape) == 3:
            mask_loss = batch["mask_loss"]  # [4, H, W]
            uncertainty = batch["uncertainty_map"]  # [4, H, W]
        else:
            mask_loss = batch["mask_loss"][b]  # [4, H, W]
            uncertainty = batch["uncertainty_map"][b]  # [4, H, W]
        
        if sample_type == "easy":
            # COLMAP samples: W_src = 1.0 (gold standard)
            W_src[b] = 1.0
        else:
            # CRITICAL FIX: Inverted uncertainty → weight mapping
            # VGGT's points_conf is UNCERTAINTY [1, ∞): higher = less reliable
            # We need: low uncertainty → high weight (1.0), high uncertainty → low weight (0.0)
            
            # Step 1: Min-max normalize uncertainty to [0, 1]
            normalized_uncertainty = (
                (uncertainty - tau_min) / (tau_max - tau_min + epsilon)
            )
            normalized_uncertainty = torch.clamp(normalized_uncertainty, 0.0, 1.0)
            
            # Step 2: INVERT to get weight (1.0 - uncertainty)
            # Apply to valid regions only (mask_loss=True)
            W_src[b] = mask_loss.float() * (1.0 - normalized_uncertainty)
    
    return W_src


# ==================== Testing and Validation ====================

def test_dataset():
    """Test the dataset implementation."""
    print("Testing VCoMatcher Dataset...")
    
    # Create dummy data paths (assuming Phase 1 has been run)
    data_dir = Path("./data/vcomatcher_phase1_test")
    if not data_dir.exists():
        print(f"Error: {data_dir} does not exist. Run Phase 1 first.")
        return False
    
    data_paths = list(data_dir.glob("*.npz"))
    if len(data_paths) == 0:
        print(f"Error: No .npz files found in {data_dir}")
        return False
    
    print(f"Found {len(data_paths)} data files")
    
    # Create dataset
    dataset = VCoMatcherDataset(
        data_paths=data_paths,
        sample_types=["easy", "hard", "extreme"],
        cache_data=True,
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    # Test single sample
    if len(dataset) > 0:
        print("\nTesting single sample...")
        sample = dataset[0]
        
        print("Sample keys:", sample.keys())
        print("\nShapes:")
        for key, val in sample.items():
            if isinstance(val, torch.Tensor):
                print(f"  {key:20s}: {tuple(val.shape)}")
            else:
                print(f"  {key:20s}: {val}")
        
        # Verify target-centric transformation
        extrinsic_rel = sample["extrinsic_rel"]  # [4, 4, 4]
        target_pose = extrinsic_rel[0]  # [4, 4]
        
        print(f"\nTarget pose (should be Identity):")
        print(target_pose.numpy())
        
        is_identity = torch.allclose(target_pose, torch.eye(4), atol=1e-4)
        print(f"Is Identity: {is_identity}")
        
        if is_identity:
            print("✓ Target-centric transformation is correct!")
        else:
            print("✗ Target-centric transformation failed!")
            return False
    
    print("\n" + "="*80)
    print("✓ Dataset test passed!")
    print("="*80)
    return True


if __name__ == "__main__":
    test_dataset()
