"""
VCoMatcher Phase 1: Data Engine with VGGT-based Pseudo-GT Generation
=====================================================================

This module implements the data generation pipeline described in VCoMatcher documentation (Page 2-5).

Key Features:
1. VGGT Model Loading: Loads pretrained VGGT model for inference
2. Dual Masking Strategy (Critical):
   - mask_geom: Loose mask (removes only invalid depths) for GNN graph construction
   - mask_loss: Strict mask (adds uncertainty σ and consistency checks) for loss computation
3. Sample Binning: Categorizes samples by overlap ratio into easy/hard/extreme bins
4. Data Storage: Saves processed data as .npz files with metadata

Author: VCoMatcher Team
Date: 2025-12-12
"""

import os
import sys
import glob
import argparse
import numpy as np
import cv2  # For PnP pose refinement and morphological operations
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

# Add VGGT to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "vggt-main"))

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map


class VGGTDataEngine:
    """
    Data engine for generating pseudo-ground-truth using VGGT.
    
    Implements dual masking strategy and overlap computation as per VCoMatcher spec.
    """
    
    def __init__(
        self,
        device: str = "cuda",
        dtype: torch.dtype = None,
        resolution: int = 518,
        tau_min: float = 0.1,
        tau_max: float = 100.0,
        tau_uncertainty: float = 15.0,  # v1.3 FINAL (proven optimal)
        epsilon_consist: float = 0.15,  # v1.3 FINAL (proven optimal)
        epsilon_numerical: float = 1e-6,
        pnp_tau: float = 6.0,  # BUGFIX: PnP confidence threshold (elite points only)
        delta0_consist: float = 0.15,  # Absolute consistency tolerance (meters)
    ):
        """
        Initialize VGGT Data Engine.
        
        Args:
            device: Computing device ('cuda' or 'cpu')
            dtype: Data type for computation (default: bfloat16 on A100+, else float16)
            resolution: VGGT input resolution (default: 518)
            tau_min: Minimum depth threshold for near plane (default: 0.1)
            tau_max: Maximum depth threshold for far plane/sky (default: 100.0)
            tau_uncertainty: Uncertainty threshold for σ_P (default: 15.0, FINAL)
            epsilon_consist: Consistency check tolerance (default: 0.15, FINAL)
            epsilon_numerical: Numerical stability epsilon (default: 1e-6)
            pnp_tau: PnP confidence threshold for elite points (default: 6.0)
        """
        self.device = device
        # BUGFIX: Handle CPU mode where cuda.get_device_capability() is not available
        if dtype is None:
            if device == "cuda" and torch.cuda.is_available():
                self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
            else:
                self.dtype = torch.float32  # Use float32 for CPU
        else:
            self.dtype = dtype
        self.resolution = resolution
        
        # Thresholds (from Page 2-4 of documentation)
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.tau_uncertainty = tau_uncertainty
        self.epsilon_consist = epsilon_consist
        self.epsilon_numerical = epsilon_numerical
        self.pnp_tau = pnp_tau  # BUGFIX: Store PnP threshold
        
        # Load VGGT model
        print(f"Loading VGGT model on {device} with dtype {self.dtype}")
        self.model = self._load_vggt_model()
        print("VGGT model loaded successfully")
        
    def _load_vggt_model(self) -> VGGT:
        """Load pretrained VGGT model from HuggingFace."""
        model = VGGT()
        _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
        model.eval()
        model = model.to(self.device)
        return model
    
    def create_valid_region_mask(
        self,
        original_coords: torch.Tensor,
        resolution: int
    ) -> np.ndarray:
        """
        创建有效区域掩膜，标记非 padding 区域
        
        修复 Padding Offset Bug: 过滤掉 padding 区域（黑色边缘）
        这些区域没有真实的图像内容，不应该用于训练
        
        Args:
            original_coords: [N, 6] = [x1, y1, x2, y2, orig_w, orig_h]
                x1, y1: 原始图像在 target space 中的左上角
                x2, y2: 原始图像在 target space 中的右下角
                orig_w, orig_h: 原始图像尺寸
            resolution: 目标分辨率（518）
        
        Returns:
            valid_mask: [N, H, W] bool array, True 表示有效区域
        
        示例:
            原始图像 (480, 640) padding 到 (640, 640) 再 resize 到 (518, 518)
            - scale = 518 / 640 = 0.809
            - new_h = 480 * 0.809 = 388
            - pad_y = (518 - 388) / 2 = 65
            - 有效区域: y ∈ [65, 453], x ∈ [0, 518]
        """
        N = original_coords.shape[0]
        valid_mask = np.zeros((N, resolution, resolution), dtype=bool)
        
        for i in range(N):
            x1, y1, x2, y2 = original_coords[i, :4].cpu().numpy()
            
            # 转换为整数索引（向内收缩以避免边界问题）
            x1_int = int(np.ceil(x1))
            y1_int = int(np.ceil(y1))
            x2_int = int(np.floor(x2))
            y2_int = int(np.floor(y2))
            
            # 确保在有效范围内
            x1_int = max(0, x1_int)
            y1_int = max(0, y1_int)
            x2_int = min(resolution, x2_int)
            y2_int = min(resolution, y2_int)
            
            # 标记有效区域
            valid_mask[i, y1_int:y2_int, x1_int:x2_int] = True
        
        return valid_mask
    
    def refine_camera_pose_pnp(
        self,
        points_3d: np.ndarray,
        points_conf: np.ndarray,
        intrinsic: np.ndarray,
        extrinsic: np.ndarray,
        valid_region_mask: Optional[np.ndarray] = None,
        n_samples: int = 5000,
    ) -> np.ndarray:
        """
        Refine camera poses using PnP to fix systematic reprojection error.
        
        Problem: VGGT's predicted extrinsics and 3D points are not perfectly aligned,
        causing ~3px systematic error.
        
        Solution: Use high-confidence 3D points to optimize each camera's extrinsic
        matrix via cv2.solvePnP, treating the 3D points as fixed "world structure".
        
        Args:
            points_3d: [N, H, W, 3] 3D world points from VGGT
            points_conf: [N, H, W] point confidence (lower = better, range [1, ∞))
            intrinsic: [N, 3, 3] camera intrinsic matrices
            extrinsic: [N, 4, 4] camera extrinsic matrices (to be refined)
            valid_region_mask: [N, H, W] optional mask to exclude padding regions
            n_samples: Number of points to sample for PnP (default: 5000)
            
        Returns:
            extrinsic_refined: [N, 4, 4] refined extrinsic matrices
        """
        N, H, W, _ = points_3d.shape
        extrinsic_refined = extrinsic.copy()
        
        # Handle tau_uncertainty (same logic as compute_dual_masks)
        # VGGT's points_conf range: [1, ∞), lower = better
        if self.tau_uncertainty < 1.0:
            effective_tau = 5.0
            print(f"  ⚠️  Old tau_uncertainty={self.tau_uncertainty:.1f} detected, using {effective_tau:.1f}")
        else:
            effective_tau = self.tau_uncertainty
        
        print(f"\nRefining camera poses using PnP (sampling {n_samples} points, tau={effective_tau:.1f})...")
        
        for i in tqdm(range(N), desc="PnP refinement"):
            # ============================================================
            # CRITICAL: Decouple PnP threshold from global mask threshold
            # ============================================================
            # PnP needs STRICT threshold (only elite points) for accurate pose
            # Masks need RELAXED threshold (include more data) for training
            # Global self.tau_uncertainty = 12.0 is for masks, NOT for PnP!
            # ============================================================
            # BUGFIX: Make PnP threshold configurable (previously hardcoded)
            pnp_tau = getattr(self, 'pnp_tau', 6.0)  # Default 6.0 if not set
            
            # Sample high-confidence points (low uncertainty)
            conf_flat = points_conf[i].flatten()
            valid_mask = conf_flat < pnp_tau  # Use strict PnP threshold, NOT global tau
            
            # Apply valid_region_mask if provided (exclude padding regions)
            if valid_region_mask is not None:
                valid_region_flat = valid_region_mask[i].flatten()
                valid_mask = valid_mask & valid_region_flat  # Combine both conditions
            
            if valid_mask.sum() < 100:
                # Not enough points for PnP, apply SVD orthogonality fix instead
                print(f"  Warning: Camera {i} has only {valid_mask.sum()} valid points, applying orthogonality fix")
                
                # Use SVD to project rotation matrix to SO(3) manifold
                R_init = extrinsic[i, :3, :3]
                U, _, Vt = np.linalg.svd(R_init)
                R_ortho = U @ Vt
                
                # Ensure det(R) = +1 (not -1)
                if np.linalg.det(R_ortho) < 0:
                    Vt[-1, :] *= -1
                    R_ortho = U @ Vt
                
                # BUGFIX: Preserve translation (only fix rotation)
                extrinsic_refined[i, :3, :3] = R_ortho
                extrinsic_refined[i, :3, 3] = extrinsic[i, :3, 3]  # Keep original translation
                continue
            
            # Get indices of valid points
            valid_indices = np.where(valid_mask)[0]
            
            # Sample n_samples points (or all if less than n_samples)
            n_actual = min(n_samples, len(valid_indices))
            sampled_indices = np.random.choice(valid_indices, n_actual, replace=False)
            
            # Get 3D points and 2D pixel coordinates
            points_3d_flat = points_3d[i].reshape(-1, 3)
            object_points = points_3d_flat[sampled_indices]  # [n_actual, 3]
            
            # Compute 2D pixel coordinates from indices
            y_coords = sampled_indices // W
            x_coords = sampled_indices % W
            # CRITICAL: +0.5 to move from pixel corner to center
            image_points = np.stack([x_coords, y_coords], axis=1).astype(np.float32) + 0.5  # [n_actual, 2]
            
            # BUGFIX: Conditional memory copy - only if not already contiguous
            # Avoids unnecessary memory allocation for large point sets
            if not object_points.flags['C_CONTIGUOUS']:
                object_points = np.ascontiguousarray(object_points, dtype=np.float64)
            else:
                object_points = object_points.astype(np.float64, copy=False)
            
            if not image_points.flags['C_CONTIGUOUS']:
                image_points = np.ascontiguousarray(image_points, dtype=np.float64)
            else:
                image_points = image_points.astype(np.float64, copy=False)
            
            # Prepare camera matrix and distortion coefficients
            K = intrinsic[i].astype(np.float64)
            dist_coeffs = np.zeros(4, dtype=np.float64)  # Assume no distortion
            
            # ============================================================
            # CRITICAL BUG FIX: VGGT extrinsic format clarification
            # ============================================================
            # VGGT outputs World-to-Camera (w2c) format: [R_w2c | t_w2c]
            # Source: vggt/utils/pose_enc.py Line 22, 88: "camera from world transformation"
            # Source: vggt/utils/geometry.py Line 59: "cam from world"
            #
            # OpenCV PnP also expects w2c format: [R_w2c | t_w2c]
            # Therefore: NO CONVERSION NEEDED!
            # ============================================================
            pose_w2c = extrinsic[i]  # [4, 4] - Already in w2c format!
            R_w2c = pose_w2c[:3, :3]  # [3, 3]
            t_w2c = pose_w2c[:3, 3]   # [3]
            
            # Convert to rodrigues for OpenCV (no conversion, direct use)
            rvec_init, _ = cv2.Rodrigues(R_w2c)
            tvec_init = t_w2c.reshape(3, 1)
            
            # ============================================================
            # Projection-based Outlier Rejection (v1.3 FINAL)
            # ============================================================
            try:
                # Step 1: Project all points using initial VGGT pose
                projected_points, _ = cv2.projectPoints(
                    object_points, rvec_init, tvec_init, K, dist_coeffs
                )
                projected_points = projected_points.squeeze()
                
                # Compute initial reprojection errors
                initial_errors = np.linalg.norm(projected_points - image_points, axis=1)
                initial_mean_error = initial_errors.mean()
                
                # Filter: keep only points with error < 10.0 px
                # BUGFIX: Handle NaN in errors
                if not np.isfinite(initial_mean_error):
                    print(f"  Warning: Camera {i} has invalid reprojection errors, keeping original pose")
                    continue
                
                median_error = np.median(initial_errors)
                dynamic_threshold = min(10.0, max(2.0, median_error*3.0))
                inlier_mask = initial_errors < dynamic_threshold
                inlier_indices = np.where(inlier_mask)[0]
                
                if len(inlier_indices) < 4:
                    print(f"  Warning: Camera {i} has only {len(inlier_indices)} inliers, keeping original pose")
                    continue
                
                # Step 2: Refine pose using filtered inliers
                success, rvec_refined, tvec_refined = cv2.solvePnP(
                    objectPoints=object_points[inlier_indices],
                    imagePoints=image_points[inlier_indices],
                    cameraMatrix=K,
                    distCoeffs=dist_coeffs,
                    rvec=rvec_init,
                    tvec=tvec_init,
                    useExtrinsicGuess=True,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                
                if not success:
                    print(f"  Warning: Camera {i} PnP refinement failed")
                    continue
                
                # Step 3: Safety check
                projected_refined, _ = cv2.projectPoints(
                    object_points[inlier_indices], rvec_refined, tvec_refined, K, dist_coeffs
                )
                refined_errors = np.linalg.norm(projected_refined.squeeze() - image_points[inlier_indices], axis=1)
                refined_mean_error = refined_errors.mean()
                
                if refined_mean_error < initial_mean_error:
                    # Accept refined pose - PnP output is already in w2c format (same as VGGT)
                    R_w2c_refined, _ = cv2.Rodrigues(rvec_refined)
                    t_w2c_refined = tvec_refined.flatten()
                    
                    # Store refined pose directly (no conversion needed)
                    extrinsic_refined[i, :3, :3] = R_w2c_refined
                    extrinsic_refined[i, :3, 3] = t_w2c_refined
                else:
                    print(f"  Warning: Camera {i} refinement increased error, rollback")
                    
            except cv2.error as e:
                print(f"  Error in PnP for camera {i}: {e}")
        
        print("  ✓ PnP refinement completed")
        return extrinsic_refined
    
    @torch.no_grad()
    def run_vggt_inference(
        self, 
        images: torch.Tensor,
        original_coords: torch.Tensor = None,
        use_sliding_window: bool = True,
        window_size: int = 32,
        overlap: int = 8,
    ) -> Dict[str, torch.Tensor]:
        """
        Run VGGT inference on a sequence of images with optional sliding window.
        
        Implements VCoMatcher.md lines 160-202 for large-scale datasets.
        
        Args:
            images: Input images [N, 3, H, W] in range [0, 1]
            original_coords: [N, 6] coordinates in padded space
            use_sliding_window: Enable sliding window for N > window_size
            window_size: Frames per window (default: 32 for 80GB A100)
            overlap: Overlapping frames (default: 8)
            
        Returns:
            Dictionary with extrinsic, intrinsic, depth, points_3d, etc.
        """
        N, _, H, W = images.shape
        assert H == self.resolution and W == self.resolution
        
        # Small datasets: direct inference
        if not use_sliding_window or N <= window_size:
            return self._run_vggt_single_window(images, original_coords)
        
        # Large datasets: sliding window + Umeyama alignment
        print(f"\n[Sliding Window] Processing {N} images with window_size={window_size}, overlap={overlap}")
        
        from vcomatcher_sliding_window import SlidingWindowProcessor
        processor = SlidingWindowProcessor(window_size, overlap)
        windows = processor.create_windows(N)
        
        # Process windows
        all_poses = []
        all_intrinsics = []
        all_depths = []
        all_points_3d = []
        all_points_conf = []
        all_depth_conf = []
        
        for win_idx, (start, end) in enumerate(windows):
            print(f"  Window {win_idx+1}/{len(windows)}: frames [{start}, {end})")
            
            window_images = images[start:end]
            window_coords = original_coords[start:end] if original_coords is not None else None
            
            result = self._run_vggt_single_window(window_images, window_coords)
            
            # First window: use as-is (global frame)
            if win_idx == 0:
                all_poses.append(result["extrinsic"])
                all_intrinsics.append(result["intrinsic"])
                all_depths.append(result["depth"])
                all_points_3d.append(result["points_3d"])
                all_points_conf.append(result["points_conf"])
                all_depth_conf.append(result["depth_conf"])
            else:
                # CRITICAL FIX: Align to PREVIOUS window, not first window
                # Concatenate all previous poses to get the accumulated global trajectory
                accumulated_poses = np.concatenate(all_poses, axis=0)
                
                # Overlap is at the END of accumulated trajectory
                overlap_start_in_prev = len(accumulated_poses) - overlap
                
                # BUGFIX: Validate overlap region exists
                if overlap_start_in_prev < 0:
                    print(f"  ⚠️  Warning: Accumulated trajectory ({len(accumulated_poses)}) < overlap ({overlap})")
                    print(f"     Skipping alignment for this window")
                    # Just append without alignment
                    all_poses.append(result["extrinsic"])
                    all_intrinsics.append(result["intrinsic"])
                    all_depths.append(result["depth"])
                    all_points_3d.append(result["points_3d"])
                    all_points_conf.append(result["points_conf"])
                    all_depth_conf.append(result["depth_conf"])
                    continue
                
                # Call align_and_stitch with points parameter
                aligned_poses, blended_overlap, aligned_points_3d = processor.align_and_stitch(
                    window_a_poses=accumulated_poses,  # All accumulated poses
                    window_b_poses=result["extrinsic"],
                    window_b_points=result["points_3d"],  # Pass points to transform
                    overlap_start_in_a=overlap_start_in_prev,
                    overlap_start_in_b=0,
                    overlap_count=overlap,
                )
                
                # Update the last overlap frames in accumulated list
                # We need to replace the overlap in the LAST window we added
                last_window_size = all_poses[-1].shape[0]
                overlap_start_in_last_window = last_window_size - overlap
                
                # BUGFIX: Guard against negative indexing when last window < overlap
                if overlap_start_in_last_window < 0:
                    print(f"  ⚠️  Warning: Last window size ({last_window_size}) < overlap ({overlap})")
                    print(f"     Adjusting overlap count to {last_window_size}")
                    overlap_start_in_last_window = 0
                    # Only blend the available frames
                    blended_overlap = blended_overlap[:last_window_size]
                
                all_poses[-1][overlap_start_in_last_window:] = blended_overlap
                
                # Append NEW non-overlap part from current window
                all_poses.append(aligned_poses[overlap:])
                all_intrinsics.append(result["intrinsic"][overlap:])
                all_depths.append(result["depth"][overlap:])
                all_points_3d.append(aligned_points_3d[overlap:])  # Use transformed points!
                all_points_conf.append(result["points_conf"][overlap:])
                all_depth_conf.append(result["depth_conf"][overlap:])
        
        # Concatenate all windows
        final_results = {
            "extrinsic": np.concatenate(all_poses, axis=0),
            "intrinsic": np.concatenate(all_intrinsics, axis=0),
            "depth": np.concatenate(all_depths, axis=0),
            "points_3d": np.concatenate(all_points_3d, axis=0),
            "points_conf": np.concatenate(all_points_conf, axis=0),
            "depth_conf": np.concatenate(all_depth_conf, axis=0),
        }
        
        if original_coords is not None:
            valid_mask = self.create_valid_region_mask(original_coords, self.resolution)
            final_results["valid_region_mask"] = valid_mask
        
        return final_results
    
    def _run_vggt_single_window(
        self,
        images: torch.Tensor,  # [N, 3, H, W]
        original_coords: torch.Tensor = None,
    ) -> Dict[str, np.ndarray]:
        """Single-window VGGT inference (original logic)."""
        N, _, H, W = images.shape
        
        with torch.amp.autocast('cuda', dtype=self.dtype):
            images_batch = images.unsqueeze(0)  # [1, N, 3, H, W]
            
            # Forward pass
            aggregated_tokens_list, ps_idx = self.model.aggregator(images_batch)
            
            # Predict cameras
            pose_enc = self.model.camera_head(aggregated_tokens_list)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(
                pose_enc, 
                images_batch.shape[-2:]
            )
            
            # Predict depth
            depth, depth_conf = self.model.depth_head(
                aggregated_tokens_list, images_batch, ps_idx
            )
            
            # Predict 3D points
            points_3d, points_conf = self.model.point_head(
                aggregated_tokens_list, images_batch, ps_idx
            )
        
        # Convert to numpy with shape validation
        extrinsic_np = extrinsic.squeeze(0).cpu().numpy()  # [N, 3, 4] or [N, 4, 4]
        intrinsic_np = intrinsic.squeeze(0).cpu().numpy()  # [N, 3, 3]
        
        # BUGFIX: Convert extrinsic from [N, 3, 4] to [N, 4, 4] if needed
        if extrinsic_np.shape[-2] == 3:
            # VGGT outputs [R|t] format, need to add [0, 0, 0, 1] row
            N_frames = extrinsic_np.shape[0]
            extrinsic_4x4 = np.zeros((N_frames, 4, 4), dtype=extrinsic_np.dtype)
            extrinsic_4x4[:, :3, :] = extrinsic_np
            extrinsic_4x4[:, 3, 3] = 1.0
            extrinsic_np = extrinsic_4x4
        
        # BUGFIX: Handle depth shape variations
        # VGGT depth output could be [1, N, H, W, 1] or [1, N, H, W]
        depth_np = depth.squeeze(0).cpu().numpy()  # Remove batch dim
        if depth_np.ndim == 4 and depth_np.shape[-1] == 1:
            depth_np = depth_np.squeeze(-1)  # [N, H, W]
        elif depth_np.ndim == 3:
            pass  # Already [N, H, W]
        else:
            raise ValueError(f"Unexpected depth shape: {depth_np.shape}")
        
        # Handle depth_conf shape
        depth_conf_np = depth_conf.squeeze(0).cpu().numpy()
        if depth_conf_np.ndim == 4 and depth_conf_np.shape[-1] == 1:
            depth_conf_np = depth_conf_np.squeeze(-1)
        
        # Handle points_conf shape
        points_conf_np = points_conf.squeeze(0).cpu().numpy()
        if points_conf_np.ndim == 4 and points_conf_np.shape[-1] == 1:
            points_conf_np = points_conf_np.squeeze(-1)
        
        # BUGFIX: Handle points_3d shape variations
        # VGGT points_3d output could be [1, N, H, W, 3] or [1, N, 3, H, W]
        points_3d_np = points_3d.squeeze(0).cpu().numpy()  # Remove batch dim
        if points_3d_np.ndim == 4:
            # Check if shape is [N, 3, H, W] (channel-first) instead of [N, H, W, 3]
            if points_3d_np.shape[1] == 3 and points_3d_np.shape[1] != points_3d_np.shape[2]:
                # Transpose from [N, 3, H, W] to [N, H, W, 3]
                points_3d_np = points_3d_np.transpose(0, 2, 3, 1)
        
        results = {
            "extrinsic": extrinsic_np,  # [N, 4, 4]
            "intrinsic": intrinsic_np,  # [N, 3, 3]
            "depth": depth_np,  # [N, H, W]
            "depth_conf": depth_conf_np,  # [N, H, W]
            "points_3d": points_3d_np,  # [N, H, W, 3]
            "points_conf": points_conf_np,  # [N, H, W]
        }
        
        if original_coords is not None:
            valid_region_mask = self.create_valid_region_mask(
                original_coords, self.resolution
            )
            results["valid_region_mask"] = valid_region_mask
        
        return results
    
    def compute_dual_masks(
        self,
        depth: np.ndarray,
        points_3d: np.ndarray,
        points_conf: np.ndarray,
        extrinsic: np.ndarray,
        intrinsic: np.ndarray,
        valid_region_mask: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute dual masking strategy as per Page 2 and Page 4.
        
        Dual Mask Philosophy:
        - mask_geom (M_geom): Loose filtering - keeps physical regions for GNN graph
        - mask_loss (I_valid): Strict filtering - ensures reliable supervision for loss
        
        Args:
            depth: [N, H, W] depth maps
            points_3d: [N, H, W, 3] 3D world points
            points_conf: [N, H, W] point confidence (σ_P)
            extrinsic: [N, 4, 4] extrinsic matrices
            intrinsic: [N, 3, 3] intrinsic matrices
            
        Returns:
            Tuple of:
                - mask_geom: [N, H, W] loose mask for geometry (bool)
                - mask_loss: [N, H, W] strict mask for loss computation (bool)
        """
        N, H, W = depth.shape
        
        # ==================== MASK_GEOM (Loose) ====================
        # Page 4: M_geom(u) = (D_i(u) > τ_min) ∧ (D_i(u) < τ_max) ∧ ¬isnan(D_i(u))
        # Purpose: Remove only invalid depths, keep physical regions for GNN
        
        mask_depth_valid = (depth > self.tau_min) & (depth < self.tau_max) & (~np.isnan(depth))
        mask_geom = mask_depth_valid  # [N, H, W]
        
        # ==================== MASK_LOSS (Strict) ====================
        # Page 2: I_valid(u) = M_valid(u) ∧ M_depth(u) ∧ M_consist(u)
        # Purpose: Ensure reliable supervision by removing uncertain regions
        
        # 1. Uncertainty filtering (M_valid) - Page 2
        # M_valid(u) = I(σ_P(u) > τ)
        # 
        # ⚠️  VGGT 的 points_conf 定义（重要！）:
        #     points_conf = 1 + exp(x), 值域 [1, ∞)
        #     值越大 = 越不确定（这是 uncertainty，不是 confidence！）
        # 
        # 修正逻辑: 保留低 uncertainty（高 confidence）的区域
        #     points_conf < tau_uncertainty (不是 >)
        # 
        # 阈值调整: tau_uncertainty 应该在 [1, 10] 范围
        #     1-3:   非常严格（只保留最确定的区域）
        #     3-5:   标准（推荐）
        #     5-10:  宽松
        #     >10:   几乎不过滤
        
        # 如果 tau_uncertainty < 1，说明是旧代码，使用默认值
        if self.tau_uncertainty < 1.0:
            # 旧阈值 0.3 → 新阈值 5.0（自动转换）
            effective_tau = 5.0
            print(f"  ⚠️  检测到旧阈值 tau_uncertainty={self.tau_uncertainty:.1f}")
            print(f"     自动调整为: {effective_tau:.1f} (VGGT 的 conf 范围是 [1, ∞))")
        else:
            effective_tau = self.tau_uncertainty
        
        mask_uncertainty = points_conf < effective_tau  # [N, H, W] ← 反转逻辑！
        
        # 2. Depth thresholding (M_depth) - Same as mask_geom
        mask_depth = mask_depth_valid
        
        # 3. Consistency check (M_consist) - Page 2
        # Compare depth from depth map vs depth derived from point map
        # M_consist(u) = |D_i(u) - d_P| < ε * D_i(u)
        
        # Compute depth from points: X_cam = R @ X_world + t
        R = extrinsic[:, :3, :3]  # [N, 3, 3]
        t = extrinsic[:, :3, 3:4]  # [N, 3, 1]
        
        # Transform points to camera space
        points_flat = points_3d.reshape(N, -1, 3)  # [N, H*W, 3]
        points_cam = np.einsum('nij,npj->npi', R, points_flat) + t.transpose(0, 2, 1)  # [N, H*W, 3]
        points_cam = points_cam.reshape(N, H, W, 3)  # [N, H, W, 3]
        
        # Extract Z component (depth)
        depth_from_points = points_cam[..., 2]  # [N, H, W]
        
        # Consistency check
        # BUGFIX: Ensure numerical stability by filtering invalid depth_from_points
        # depth_from_points could be NaN or negative if points_3d is corrupted
        depth_diff = np.abs(depth - depth_from_points)
        delta0_consist = 0.15  # v1.3 FINAL (proven optimal)
        
        # Guard against NaN/Inf in depth_diff (can occur if depth_from_points is invalid)
        depth_diff_valid = np.isfinite(depth_diff) & (depth_from_points > 0)
        mask_consistency = depth_diff_valid & (depth_diff < (self.epsilon_consist * depth + delta0_consist))
        
        # BUGFIX: Warn if consistency check filters out too much data
        consistency_ratio = mask_consistency.sum() / mask_consistency.size
        if consistency_ratio < 0.1:
            print(f"  ⚠️  Warning: Consistency check is very strict ({consistency_ratio*100:.1f}% pass)")
            print(f"     This may indicate misalignment between depth and points_3d")
        
        # Combine all strict conditions for mask_loss
        mask_loss = mask_uncertainty & mask_depth & mask_consistency  # [N, H, W]
        
        # 修复 Padding Offset Bug: 过滤 padding 区域
        if valid_region_mask is not None:
            mask_geom = mask_geom & valid_region_mask
            mask_loss = mask_loss & valid_region_mask
        
        return mask_geom, mask_loss
    
    def compute_overlap_matrix(
        self,
        points_3d: np.ndarray,
        depth: np.ndarray,
        extrinsic: np.ndarray,
        intrinsic: np.ndarray,
        mask_loss: np.ndarray,
        alpha: float = 0.03,      # Reverted to v1.3
        gamma: float = 0.04,      # Noise lower bound: 4% of d_obs  
        delta0: float = 0.15,     # Reverted to v1.3
        batch_size: int = 10,     # BUGFIX: Batch processing to prevent GPU OOM
    ) -> np.ndarray:
        """
        Compute overlap matrix O_ij with robust geometry validation.
        
        UPGRADES:
        1. **Bilinear Interpolation**: Uses torch.nn.functional.grid_sample for sub-pixel sampling
        2. **Asymmetric Dynamic Occlusion**: diff ∈ [-(γ·d_obs + δ0), α·d_obs + δ0]
        3. **Dynamic 3D Consistency**: threshold = 0.03 · d_obs + 0.1
        
        Overlap Score Definition (Page 3):
        O_ij = (Σ_u∈valid(I_i) I_vis(u)) / N_valid(I_i)
        
        Visibility Checks:
        1. FoV check: Projected point within image bounds
        2. Occlusion check (Asymmetric): Lower ≤ (d_proj - d_obs) ≤ Upper
           - Upper = α · d_obs + δ0 (strict, prevents occluded points)
           - Lower = -(γ · d_obs + δ0) (loose, tolerates interpolation noise)
        3. 3D consistency (Dynamic): ||X_proj - X_obs|| < (0.03 · d_obs + 0.1)
        
        Args:
            points_3d: [N, H, W, 3] 3D world points
            depth: [N, H, W] depth maps
            extrinsic: [N, 4, 4] extrinsic matrices
            intrinsic: [N, 3, 3] intrinsic matrices
            mask_loss: [N, H, W] valid mask for counting
            alpha: Occlusion upper bound coefficient (default: 0.03 = 3%)
            gamma: Noise lower bound coefficient (default: 0.04 = 4%)
            delta0: Absolute tolerance in meters (default: 0.15 = 15cm)
            
        Returns:
            overlap_matrix: [N, N] overlap scores in range [0, 1]
        """
        N, H, W, _ = points_3d.shape
        
        # Convert to torch tensors on GPU (avoid repeated CPU-GPU transfers)
        points_3d_th = torch.from_numpy(points_3d).float().to(self.device)  # [N, H, W, 3]
        depth_th = torch.from_numpy(depth).float().to(self.device)  # [N, H, W]
        extrinsic_th = torch.from_numpy(extrinsic).float().to(self.device)  # [N, 4, 4]
        intrinsic_th = torch.from_numpy(intrinsic).float().to(self.device)  # [N, 3, 3]
        mask_loss_th = torch.from_numpy(mask_loss).bool().to(self.device)  # [N, H, W]
        
        # Prepare depth and points_3d for grid_sample
        # grid_sample expects [N, C, H, W]
        depth_th_4d = depth_th.unsqueeze(1)  # [N, 1, H, W]
        points_3d_th_4d = points_3d_th.permute(0, 3, 1, 2)  # [N, 3, H, W]
        
        overlap_matrix = np.zeros((N, N), dtype=np.float32)
        
        # BUGFIX: Warn about potential GPU memory usage for large scenes
        if N > 50:
            print(f"\n⚠️  Large scene detected (N={N})")
            print(f"   Using batched processing (batch_size={batch_size}) to prevent GPU OOM")
            print(f"   Estimated GPU memory: ~{N * 518 * 518 * 3 * 4 / 1e9:.2f} GB")
        
        print("Computing overlap matrix with robust geometry validation...")
        for i in tqdm(range(N), desc="Source images"):
            # Get valid pixels in source image i
            valid_mask_i = mask_loss_th[i]  # [H, W]
            if not valid_mask_i.any():
                continue
            
            n_valid_i = valid_mask_i.sum().item()
            
            # Get 3D points from source image i
            X_world = points_3d_th[i]  # [H, W, 3]
            
            # BUGFIX: Process targets in batches to avoid GPU OOM
            for j_batch_start in range(0, N, batch_size):
                j_batch_end = min(j_batch_start + batch_size, N)
                
                for j in range(j_batch_start, j_batch_end):
                    if i == j:
                        overlap_matrix[i, j] = 1.0
                        continue
                    
                    # === Step 1: Geometric Reprojection ===
                    # Transform to camera j coordinate system
                    R_j = extrinsic_th[j, :3, :3]  # [3, 3]
                    t_j = extrinsic_th[j, :3, 3]  # [3]
                    K_j = intrinsic_th[j]  # [3, 3]
                    
                    # Transform world points to camera j
                    X_world_flat = X_world.reshape(-1, 3)  # [H*W, 3]
                    X_cam_j = (R_j @ X_world_flat.T).T + t_j  # [H*W, 3]
                    X_cam_j = X_cam_j.reshape(H, W, 3)
                    
                    # Project to image j
                    X_proj_homog = (K_j @ X_cam_j.reshape(-1, 3).T).T  # [H*W, 3]
                    X_proj_homog = X_proj_homog.reshape(H, W, 3)
                    
                    # Perspective division
                    z_proj = X_proj_homog[..., 2]  # [H, W] - projected depth (d_proj)
                    u_proj = X_proj_homog[..., 0] / (z_proj + self.epsilon_numerical)  # [H, W]
                    v_proj = X_proj_homog[..., 1] / (z_proj + self.epsilon_numerical)  # [H, W]
                    
                    # === Step 2: Visibility Checks ===
                    
                    # 1. FoV check
                    check_fov = (u_proj >= 0) & (u_proj < W) & (v_proj >= 0) & (v_proj < H) & (z_proj > 0)
                    
                    # 2. Bilinear sampling of d_obs and X_obs using grid_sample
                    # Normalize coordinates to [-1, 1] for grid_sample
                    # BUGFIX: Explicit clamping for numerical stability at boundaries
                    # Handle edge case where W=1 or H=1 (single pixel dimension)
                    w_norm = max(W - 1, 1)
                    h_norm = max(H - 1, 1)
                    grid_x = torch.clamp((u_proj / w_norm) * 2 - 1, -1.0, 1.0)  # [H, W]
                    grid_y = torch.clamp((v_proj / h_norm) * 2 - 1, -1.0, 1.0)  # [H, W]
                    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # [1, H, W, 2]
                    
                    # Sample depth from target image j (bilinear interpolation)
                    d_obs = F.grid_sample(
                        depth_th_4d[j:j+1],  # [1, 1, H, W]
                        grid,  # [1, H, W, 2]
                        mode='bilinear',
                        padding_mode='border',
                        align_corners=True
                    ).squeeze()  # [H, W]
                    
                    # Sample 3D points from target image j (bilinear interpolation)
                    X_obs = F.grid_sample(
                        points_3d_th_4d[j:j+1],  # [1, 3, H, W]
                        grid,  # [1, H, W, 2]
                        mode='bilinear',
                        padding_mode='border',
                        align_corners=True
                    ).squeeze(0).permute(1, 2, 0)  # [H, W, 3]
                    
                    # 3. Asymmetric Dynamic Occlusion Check
                    # diff = d_proj - d_obs
                    diff = z_proj - d_obs
                    
                    # Upper bound: α · d_obs + δ0 (strict, prevent occlusion)
                    upper = alpha * d_obs + delta0
                    
                    # Lower bound: -(γ · d_obs + δ0) (loose, tolerate noise)
                    lower = -(gamma * d_obs + delta0)
                    
                    # Check: Lower ≤ diff ≤ Upper
                    check_occ = (diff >= lower) & (diff <= upper)
                    
                    # 4. Dynamic 3D Consistency Check
                    # threshold = 0.03 · d_obs + 0.1
                    dist_3d = torch.norm(X_world - X_obs, dim=-1)  # [H, W]
                    thresh_3d = 0.03 * d_obs + 0.1
                    check_3d_consist = dist_3d < thresh_3d
                    
                    # Combine all visibility checks
                    visibility = check_fov & check_occ & check_3d_consist  # [H, W]
                    
                    # === Step 3: Compute Overlap Score ===
                    visible_and_valid = visibility & valid_mask_i
                    n_visible = visible_and_valid.sum().item()
                    
                    overlap_matrix[i, j] = n_visible / (n_valid_i + self.epsilon_numerical)
                
                # BUGFIX: Clear GPU cache after each batch to prevent memory accumulation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        return overlap_matrix
    
    def bin_samples_by_overlap(
        self,
        overlap_matrix: np.ndarray,
    ) -> List[Dict]:
        """
        Bin image pairs by overlap ratio into easy/hard/extreme categories (Page 12).
        
        Binning Strategy:
        - Easy/Standard: 0.4 < O_ij ≤ 0.7  (COLMAP supervision)
        - Hard: 0.1 < O_ij ≤ 0.4           (VGGT pseudo-GT)
        - Extreme: 0.05 < O_ij ≤ 0.1       (VGGT exploration)
        
        Args:
            overlap_matrix: [N, N] overlap scores
            
        Returns:
            List of sample dictionaries with metadata
        """
        N = overlap_matrix.shape[0]
        samples = []
        
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                
                o_ij = overlap_matrix[i, j]
                
                # Determine sample type based on overlap
                if 0.4 < o_ij <= 0.7:
                    sample_type = "easy"
                elif 0.1 < o_ij <= 0.4:
                    sample_type = "hard"
                elif 0.05 < o_ij <= 0.1:
                    sample_type = "extreme"
                else:
                    continue  # Skip samples outside valid ranges
                
                samples.append({
                    "source_idx": i,
                    "target_idx": j,
                    "overlap_score": float(o_ij),
                    "sample_type": sample_type,
                })
        
        # Statistics
        n_easy = sum(1 for s in samples if s["sample_type"] == "easy")
        n_hard = sum(1 for s in samples if s["sample_type"] == "hard")
        n_extreme = sum(1 for s in samples if s["sample_type"] == "extreme")
        
        print(f"\nSample Statistics:")
        print(f"  Easy (0.4-0.7):     {n_easy:6d} pairs")
        print(f"  Hard (0.1-0.4):     {n_hard:6d} pairs")
        print(f"  Extreme (0.05-0.1): {n_extreme:6d} pairs")
        print(f"  Total:              {len(samples):6d} pairs")
        
        # BUGFIX: Warn if no valid samples found
        if len(samples) == 0:
            print(f"\n⚠️  WARNING: No valid samples found!")
            print(f"   All overlap scores are outside valid ranges [0.05, 0.7]")
            print(f"   This may indicate:")
            print(f"     - Camera poses are incorrect")
            print(f"     - Scene has very low overlap")
            print(f"     - Depth/geometry estimation failed")
        
        return samples
    
    def process_scene(
        self,
        scene_dir: Path,
        output_dir: Path,
    ) -> bool:
        """
        Process a single scene: VGGT inference + dual masking + overlap computation.
        
        Args:
            scene_dir: Path to scene directory containing 'images/' subfolder
            output_dir: Path to output directory for .npz files
            
        Returns:
            True if successful, False otherwise
        """
        print(f"\n{'='*80}")
        print(f"Processing scene: {scene_dir}")
        print(f"{'='*80}")
        
        # Load images
        image_dir = scene_dir / "images"
        if not image_dir.exists():
            print(f"Error: {image_dir} does not exist")
            return False
        
        image_paths = sorted(glob.glob(str(image_dir / "*")))
        if len(image_paths) == 0:
            print(f"Error: No images found in {image_dir}")
            return False
        
        print(f"Found {len(image_paths)} images")
        
        # Load and preprocess images
        images, original_coords = load_and_preprocess_images_square(image_paths, self.resolution)
        images = images.to(self.device)
        print(f"Loaded images with shape: {images.shape}")
        
        # BUGFIX: Validate minimum number of images
        N = images.shape[0]
        if N < 2:
            print(f"Error: Need at least 2 images for matching, got {N}")
            return False
        
        # Run VGGT inference
        print("\nRunning VGGT inference...")
        vggt_results = self.run_vggt_inference(images, original_coords)
        
        # === NEW: Refine camera poses using PnP ===
        # Fix systematic ~3px reprojection error by optimizing extrinsics
        vggt_results["extrinsic"] = self.refine_camera_pose_pnp(
            points_3d=vggt_results["points_3d"],
            points_conf=vggt_results["points_conf"],
            intrinsic=vggt_results["intrinsic"],
            extrinsic=vggt_results["extrinsic"],
            valid_region_mask=vggt_results.get("valid_region_mask"),  # Exclude padding
            n_samples=5000,
        )
        
        # Compute dual masks
        print("\nComputing dual masks (mask_geom, mask_loss)...")
        mask_geom, mask_loss = self.compute_dual_masks(
            depth=vggt_results["depth"],
            points_3d=vggt_results["points_3d"],
            points_conf=vggt_results["points_conf"],
            extrinsic=vggt_results["extrinsic"],
            intrinsic=vggt_results["intrinsic"],
            valid_region_mask=vggt_results.get("valid_region_mask"),
        )
        
        print(f"  mask_geom: {mask_geom.sum() / mask_geom.size * 100:.1f}% valid pixels")
        print(f"  mask_loss: {mask_loss.sum() / mask_loss.size * 100:.1f}% valid pixels")
        
        # Compute overlap matrix
        print("\nComputing overlap matrix...")
        overlap_matrix = self.compute_overlap_matrix(
            points_3d=vggt_results["points_3d"],
            depth=vggt_results["depth"],
            extrinsic=vggt_results["extrinsic"],
            intrinsic=vggt_results["intrinsic"],
            mask_loss=mask_loss,
        )
        
        # Bin samples
        samples = self.bin_samples_by_overlap(overlap_matrix)
        
        # Save results
        output_dir.mkdir(parents=True, exist_ok=True)
        scene_name = scene_dir.name
        output_path = output_dir / f"{scene_name}.npz"
        
        print(f"\nSaving results to {output_path}...")
        
        # Prepare save dict
        save_dict = {
            # Geometry
            "extrinsic": vggt_results["extrinsic"],
            "intrinsic": vggt_results["intrinsic"],
            "depth": vggt_results["depth"],
            "points_3d": vggt_results["points_3d"],
            # Confidence
            "depth_conf": vggt_results["depth_conf"],
            "points_conf": vggt_results["points_conf"],
            # Dual masks
            "mask_geom": mask_geom,
            "mask_loss": mask_loss,
            # Overlap and samples
            "overlap_matrix": overlap_matrix,
            "samples": samples,
            # Metadata
            "image_paths": np.array(image_paths, dtype=object),
            "resolution": self.resolution,
            "tau_min": self.tau_min,
            "tau_max": self.tau_max,
            "tau_uncertainty": self.tau_uncertainty,
        }
        
        # Add valid_region_mask if it exists (Padding fix)
        if "valid_region_mask" in vggt_results:
            save_dict["valid_region_mask"] = vggt_results["valid_region_mask"]
            print(f"  ✓ Saving valid_region_mask (Padding fix)")
        
        np.savez_compressed(output_path, **save_dict)
        
        print(f"✓ Scene processed successfully!")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="VCoMatcher Phase 1: Data Engine with VGGT-based Pseudo-GT Generation"
    )
    parser.add_argument(
        "--scene_dir",
        type=str,
        required=True,
        help="Path to scene directory (containing 'images/' subfolder)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/vcomatcher_phase1",
        help="Output directory for processed .npz files",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Computing device (cuda or cpu)",
    )
    parser.add_argument(
        "--tau_min",
        type=float,
        default=0.1,
        help="Minimum depth threshold (near plane)",
    )
    parser.add_argument(
        "--tau_max",
        type=float,
        default=100.0,
        help="Maximum depth threshold (far plane/sky)",
    )
    parser.add_argument(
        "--tau_uncertainty",
        type=float,
        default=8.0,
        help="Uncertainty threshold for σ_P filtering (VGGT range: [1, ∞), lower=better)",
    )
    parser.add_argument(
        "--pnp_tau",
        type=float,
        default=6.0,
        help="PnP confidence threshold for elite points (default: 6.0, strict)",
    )
    
    args = parser.parse_args()
    
    # Initialize data engine
    engine = VGGTDataEngine(
        device=args.device,
        tau_min=args.tau_min,
        tau_max=args.tau_max,
        tau_uncertainty=args.tau_uncertainty,
        pnp_tau=args.pnp_tau,
    )
    
    # Process scene
    scene_dir = Path(args.scene_dir)
    output_dir = Path(args.output_dir)
    
    success = engine.process_scene(scene_dir, output_dir)
    
    if success:
        print("\n" + "="*80)
        print("✓ Phase 1 data generation completed successfully!")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("✗ Phase 1 data generation failed")
        print("="*80)
        sys.exit(1)


if __name__ == "__main__":
    main()
