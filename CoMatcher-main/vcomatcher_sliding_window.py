"""
VCoMatcher Sliding Window Implementation
=========================================

Implements sliding window + Umeyama alignment + global stitching + linear smoothing
as per VCoMatcher.md lines 160-202.

For 80GB A100 optimization.

Author: VCoMatcher Team
Date: 2025-12-21
"""

import numpy as np
import torch
from typing import Dict, List, Tuple


def umeyama_alignment(
    source_points: np.ndarray,  # [N, 3] - Batch B positions
    target_points: np.ndarray,  # [N, 3] - Batch A positions (global)
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute Sim3 (similarity) transformation using Umeyama algorithm.
    
    Finds scale s, rotation R, translation t such that:
        target ≈ s * R * source + t
    
    Based on VCoMatcher.md line 186-187.
    
    Args:
        source_points: Points in local coordinate system [N, 3]
        target_points: Points in global coordinate system [N, 3]
    
    Returns:
        s: scale factor
        R: rotation matrix [3, 3]
        t: translation vector [3]
    """
    assert source_points.shape == target_points.shape
    assert source_points.shape[1] == 3
    N = source_points.shape[0]
    
    # BUGFIX: Check minimum point count for stable alignment
    if N < 3:
        raise ValueError(f"Umeyama requires at least 3 points, got {N}")
    
    # Center the point sets
    centroid_src = np.mean(source_points, axis=0)
    centroid_tgt = np.mean(target_points, axis=0)
    
    src_centered = source_points - centroid_src
    tgt_centered = target_points - centroid_tgt
    
    # Compute cross-covariance matrix
    H = src_centered.T @ tgt_centered  # [3, 3]
    
    # SVD decomposition
    U, S, Vt = np.linalg.svd(H)
    
    # Compute rotation (ensure det(R) = +1)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Compute scale (Umeyama algorithm)
    # BUGFIX v1.8: Correct scale formula!
    # Standard Umeyama uses: scale = sum(singular_values) / sum(source_variance)
    # where source_variance is the total variance, not mean variance
    var_src = np.sum(src_centered ** 2)  # Total variance (sum of squared norms)
    
    # BUGFIX: Handle degenerate case where all source points are identical
    if var_src < 1e-9:
        print(f"  ⚠️  Warning: Source points are nearly identical (var={var_src:.2e})")
        print(f"     Using scale=1.0 and identity rotation")
        # Return identity transformation
        return 1.0, np.eye(3), centroid_tgt - centroid_src
    
    scale = np.sum(S) / var_src
    
    # BUGFIX: Validate scale is reasonable (prevent degenerate cases)
    if scale < 0.1 or scale > 10.0:
        print(f"  ⚠️  Warning: Extreme scale factor detected: {scale:.3f}")
        print(f"     This may indicate misaligned point clouds")
        scale = np.clip(scale, 0.5, 2.0)  # Clamp to reasonable range
    
    # Compute translation
    t = centroid_tgt - scale * R @ centroid_src
    
    return scale, R, t


def apply_sim3_to_pose(
    pose: np.ndarray,  # [4, 4]
    scale: float,
    R_align: np.ndarray,  # [3, 3]
    t_align: np.ndarray,  # [3]
) -> np.ndarray:
    """
    Apply Sim3 transformation to a camera pose matrix.
    
    CRITICAL COORDINATE SYSTEM HANDLING:
    When world coordinates transform as: P_new = s*R*P_old + t
    The w2c pose must be transformed to maintain: P_cam = T_w2c * P_world
    
    For w2c poses: T_w2c_new = T_w2c_old @ inv(s*R) @ translate(-t)
    
    Derivation:
        P_cam = T_w2c_old @ P_world_old
        P_cam = T_w2c_new @ P_world_new
        P_cam = T_w2c_new @ (s*R*P_world_old + t)
        => T_w2c_new = T_w2c_old @ inv(s*R) - T_w2c_new @ t
    
    Expanded form:
        R_new = R_cam @ R_align^T
        t_new = (t_cam - R_cam @ R_align^T @ t_align) / scale
    
    Args:
        pose: Camera pose [4, 4] in local coordinate system (w2c format)
        scale: Scale factor from Umeyama
        R_align: Rotation from Umeyama
        t_align: Translation from Umeyama
    
    Returns:
        transformed_pose: [4, 4] in global coordinate system
    """
    R_cam = pose[:3, :3]
    t_cam = pose[:3, 3]
    
    # BUGFIX v1.8: Correct w2c transformation formula (verified with debug_sim3.py)
    # When world coords scale by s, the w2c matrix must scale inversely to maintain P_cam
    # 
    # Mathematical verification:
    #   P_cam = R_w2c * P_world + t_w2c  (original)
    #   P_cam = R_new * P_world_new + t_new  (after transform)
    #   P_cam = R_new * (s*R*P_world + t) + t_new
    # 
    # For P_cam to remain constant:
    #   R_new = (R_cam @ R^T) / s
    #   t_new = t_cam - R_new @ t
    # 
    # Note: R_new is not a pure rotation matrix (det != 1), but it's the correct
    # transformation that preserves projection after world coordinate scaling
    R_new = (R_cam @ R_align.T) / scale  # Inverse rotation AND scale
    t_new = t_cam - R_new @ t_align  # Inverse translation
    
    pose_new = np.eye(4, dtype=pose.dtype)
    pose_new[:3, :3] = R_new
    pose_new[:3, 3] = t_new
    
    return pose_new


def apply_sim3_to_points(
    points_3d: np.ndarray,  # [N, H, W, 3] or [H, W, 3]
    scale: float,
    R_align: np.ndarray,  # [3, 3]
    t_align: np.ndarray,  # [3]
) -> np.ndarray:
    """
    Apply Sim3 transformation to 3D point cloud.
    
    CRITICAL: This MUST be called alongside apply_sim3_to_pose to keep
    camera and points in the same coordinate system!
    
    Args:
        points_3d: 3D points in local coordinate system
        scale: Scale factor from Umeyama
        R_align: Rotation from Umeyama
        t_align: Translation from Umeyama
    
    Returns:
        transformed_points: 3D points in global coordinate system
    """
    original_shape = points_3d.shape
    original_dtype = points_3d.dtype
    
    # Reshape to [N, 3] for transformation
    if points_3d.ndim == 4:  # [N, H, W, 3]
        N, H, W, _ = points_3d.shape
        points_flat = points_3d.reshape(-1, 3)
    elif points_3d.ndim == 3:  # [H, W, 3]
        H, W, _ = points_3d.shape
        points_flat = points_3d.reshape(-1, 3)
    else:
        raise ValueError(f"Unexpected points_3d shape: {points_3d.shape}")
    
    # Apply Sim3: P_global = s * R * P_local + t
    points_transformed = scale * (points_flat @ R_align.T) + t_align
    
    # Reshape back to original shape and preserve dtype
    points_transformed = points_transformed.reshape(original_shape).astype(original_dtype)
    
    return points_transformed


def linear_blend_poses(
    pose_a: np.ndarray,  # [4, 4]
    pose_b: np.ndarray,  # [4, 4]
    weight_a: float,     # 0 to 1
) -> np.ndarray:
    """
    Linear blending of two poses in overlap region.
    
    Based on VCoMatcher.md line 194-201.
    
    Args:
        pose_a: Pose from Window A
        pose_b: Pose from Window B (already aligned to A)
        weight_a: Weight for pose_a (1.0 = full A, 0.0 = full B)
    
    Returns:
        blended_pose: [4, 4]
    """
    weight_b = 1.0 - weight_a
    
    # Blend translation (linear interpolation)
    t_blend = weight_a * pose_a[:3, 3] + weight_b * pose_b[:3, 3]
    
    # Blend rotation (SLERP approximation via matrix)
    # For small differences, linear blend is acceptable
    R_blend = weight_a * pose_a[:3, :3] + weight_b * pose_b[:3, :3]
    
    # Re-orthogonalize (SVD projection to SO(3))
    U, _, Vt = np.linalg.svd(R_blend)
    R_blend = U @ Vt
    if np.linalg.det(R_blend) < 0:
        Vt[-1, :] *= -1
        R_blend = U @ Vt
    
    # BUGFIX: Preserve dtype from input poses
    pose_blend = np.eye(4, dtype=pose_a.dtype)
    pose_blend[:3, :3] = R_blend
    pose_blend[:3, 3] = t_blend
    
    return pose_blend


class SlidingWindowProcessor:
    """
    Sliding window processor for large-scale VGGT inference.
    
    Implements VCoMatcher.md "注意事项" section (lines 160-202).
    """
    
    def __init__(
        self,
        window_size: int = 32,
        overlap_size: int = 8,
    ):
        """
        Args:
            window_size: Number of frames per window (default: 32 for 80GB A100)
            overlap_size: Number of overlapping frames between windows (default: 8)
        """
        self.window_size = window_size
        self.overlap_size = overlap_size
    
    def create_windows(self, total_frames: int) -> List[Tuple[int, int]]:
        """
        Create sliding window indices.
        
        Args:
            total_frames: Total number of frames
        
        Returns:
            List of (start_idx, end_idx) tuples
        """
        # BUGFIX: Validate input
        if total_frames <= 0:
            raise ValueError(f"total_frames must be positive, got {total_frames}")
        
        windows = []
        stride = self.window_size - self.overlap_size
        
        # BUGFIX: Ensure stride is positive
        if stride <= 0:
            raise ValueError(
                f"Invalid window configuration: window_size ({self.window_size}) "
                f"must be greater than overlap_size ({self.overlap_size})"
            )
        
        start = 0
        while start < total_frames:
            end = min(start + self.window_size, total_frames)
            windows.append((start, end))
            
            if end >= total_frames:
                break
            
            start += stride
        
        return windows
    
    def align_and_stitch(
        self,
        window_a_poses: np.ndarray,  # [N_a, 4, 4]
        window_b_poses: np.ndarray,  # [N_b, 4, 4]
        window_b_points: np.ndarray,  # [N_b, H, W, 3] - NEW!
        overlap_start_in_a: int,
        overlap_start_in_b: int,
        overlap_count: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Align Window B to Window A using Umeyama on overlap region,
        then apply linear blending. Also transforms 3D points.
        
        Args:
            window_a_poses: Poses from Window A (global frame)
            window_b_poses: Poses from Window B (local frame)
            window_b_points: 3D points from Window B (local frame) - NEW!
            overlap_start_in_a: Start index of overlap in Window A
            overlap_start_in_b: Start index of overlap in Window B (usually 0)
            overlap_count: Number of overlapping frames
        
        Returns:
            aligned_b_poses: Window B poses transformed to global frame
            blended_overlap: Blended poses for overlap region
            aligned_b_points: Window B points transformed to global frame - NEW!
        """
        # Extract overlap translations for Umeyama
        overlap_a_trans = window_a_poses[overlap_start_in_a:overlap_start_in_a+overlap_count, :3, 3]
        overlap_b_trans = window_b_poses[overlap_start_in_b:overlap_start_in_b+overlap_count, :3, 3]
        
        # Compute Sim3 alignment
        scale, R, t = umeyama_alignment(overlap_b_trans, overlap_a_trans)
        
        # Apply transformation to ALL Window B poses
        aligned_b_poses = np.zeros_like(window_b_poses)
        for i in range(window_b_poses.shape[0]):
            aligned_b_poses[i] = apply_sim3_to_pose(window_b_poses[i], scale, R, t)
        
        # CRITICAL: Apply SAME transformation to ALL Window B points
        aligned_b_points = apply_sim3_to_points(window_b_points, scale, R, t)
        
        # Linear blending in overlap region
        blended_overlap = []
        
        # BUGFIX: Guard against edge cases
        if overlap_count == 0:
            # No overlap - return empty blend
            return aligned_b_poses, np.array(blended_overlap), aligned_b_points
        elif overlap_count == 1:
            # Single frame overlap - use 50/50 blend
            pose_a = window_a_poses[overlap_start_in_a]
            pose_b = aligned_b_poses[overlap_start_in_b]
            blended = linear_blend_poses(pose_a, pose_b, 0.5)
            blended_overlap.append(blended)
        else:
            # Multiple frames - linear interpolation
            for i in range(overlap_count):
                # Weight decreases from 1.0 (start) to 0.0 (end)
                weight_a = 1.0 - (i / (overlap_count - 1))
                
                pose_a = window_a_poses[overlap_start_in_a + i]
                pose_b = aligned_b_poses[overlap_start_in_b + i]
                
                blended = linear_blend_poses(pose_a, pose_b, weight_a)
                blended_overlap.append(blended)
        
        return aligned_b_poses, np.array(blended_overlap), aligned_b_points
