"""
Dataset Structure Verification Tool
====================================

Validates dataset structure before running batch processing.
Checks for common issues and provides recommendations.

Usage:
    python verify_dataset_structure.py --dataset_root /path/to/dataset --dataset_name scannet
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict


class DatasetVerifier:
    """Verify dataset structure and identify potential issues."""
    
    def __init__(self, dataset_root: Path, dataset_name: str):
        self.dataset_root = dataset_root
        self.dataset_name = dataset_name
        self.issues = []
        self.warnings = []
        self.stats = defaultdict(int)
    
    def verify(self) -> bool:
        """
        Run all verification checks.
        
        Returns:
            True if all checks pass, False otherwise
        """
        print("="*80)
        print(f"Verifying {self.dataset_name.upper()} Dataset Structure")
        print("="*80)
        print(f"Dataset root: {self.dataset_root}")
        print()
        
        # Check 1: Root directory exists
        if not self._check_root_exists():
            return False
        
        # Check 2: Discover scenes
        scenes = self._discover_scenes()
        if len(scenes) == 0:
            self.issues.append("No scenes found in dataset")
            return False
        
        print(f"✓ Found {len(scenes)} potential scenes")
        print()
        
        # Check 3: Verify scene structure
        valid_scenes = self._verify_scenes(scenes)
        
        # Check 4: Analyze scene statistics
        self._analyze_scenes(valid_scenes)
        
        # Print summary
        self._print_summary(len(scenes), len(valid_scenes))
        
        return len(self.issues) == 0
    
    def _check_root_exists(self) -> bool:
        """Check if dataset root directory exists."""
        if not self.dataset_root.exists():
            self.issues.append(f"Dataset root does not exist: {self.dataset_root}")
            return False
        
        if not self.dataset_root.is_dir():
            self.issues.append(f"Dataset root is not a directory: {self.dataset_root}")
            return False
        
        print(f"✓ Dataset root exists")
        return True
    
    def _discover_scenes(self) -> List[Path]:
        """Discover all potential scene directories."""
        # Find all 'images' directories
        image_dirs = list(self.dataset_root.glob("*/images"))
        
        # Get parent directories (scene directories)
        scene_dirs = [img_dir.parent for img_dir in image_dirs]
        
        return sorted(scene_dirs)
    
    def _verify_scenes(self, scenes: List[Path]) -> List[Path]:
        """
        Verify each scene and identify issues.
        
        Returns:
            List of valid scene directories
        """
        print("Verifying scenes...")
        print("-"*80)
        
        valid_scenes = []
        
        for scene_dir in scenes:
            scene_name = scene_dir.name
            issues = []
            
            # Check 1: images/ directory exists
            image_dir = scene_dir / "images"
            if not image_dir.exists():
                issues.append("'images/' directory missing")
                self.stats['missing_images_dir'] += 1
                continue
            
            # Check 2: Find image files
            image_files = []
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                image_files.extend(list(image_dir.glob(f"*{ext}")))
            
            num_images = len(image_files)
            
            if num_images == 0:
                issues.append("No images found")
                self.stats['empty_scenes'] += 1
                continue
            
            if num_images == 1:
                issues.append(f"Only 1 image found (need at least 2)")
                self.stats['single_image_scenes'] += 1
                continue
            
            # Check 3: Image file sizes
            suspicious_files = []
            for img_path in image_files[:10]:  # Check first 10
                if img_path.stat().st_size < 1024:  # < 1KB
                    suspicious_files.append(img_path.name)
            
            if len(suspicious_files) > 0:
                self.warnings.append(
                    f"{scene_name}: Suspicious small images: {', '.join(suspicious_files)}"
                )
                self.stats['suspicious_images'] += len(suspicious_files)
            
            # Scene is valid
            valid_scenes.append(scene_dir)
            self.stats['valid_scenes'] += 1
            self.stats['total_images'] += num_images
            
            # Print progress every 50 scenes
            if len(valid_scenes) % 50 == 0:
                print(f"  Processed {len(valid_scenes)} valid scenes...")
        
        print(f"✓ Verification complete")
        print()
        
        return valid_scenes
    
    def _analyze_scenes(self, scenes: List[Path]):
        """Analyze scene statistics."""
        if len(scenes) == 0:
            return
        
        print("Analyzing scene statistics...")
        print("-"*80)
        
        image_counts = []
        
        for scene_dir in scenes:
            # BUG FIX v1.2.1: Support both 'images' and 'imgs' directories
            image_dir = scene_dir / "images"
            if not image_dir.exists():
                image_dir = scene_dir / "imgs"
            
            image_files = []
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                image_files.extend(list(image_dir.glob(f"*{ext}")))
            
            image_counts.append(len(image_files))
        
        # Compute statistics
        import numpy as np
        image_counts = np.array(image_counts)
        
        min_images = int(image_counts.min())
        max_images = int(image_counts.max())
        mean_images = float(image_counts.mean())
        median_images = float(np.median(image_counts))
        
        # Categorize scenes by size
        small_scenes = np.sum(image_counts < 20)
        medium_scenes = np.sum((image_counts >= 20) & (image_counts < 60))
        large_scenes = np.sum(image_counts >= 60)
        
        self.stats['min_images'] = min_images
        self.stats['max_images'] = max_images
        self.stats['mean_images'] = mean_images
        self.stats['median_images'] = median_images
        self.stats['small_scenes'] = small_scenes
        self.stats['medium_scenes'] = medium_scenes
        self.stats['large_scenes'] = large_scenes
        
        print(f"Image count per scene:")
        print(f"  Min:    {min_images}")
        print(f"  Max:    {max_images}")
        print(f"  Mean:   {mean_images:.1f}")
        print(f"  Median: {median_images:.0f}")
        print()
        print(f"Scene size distribution:")
        print(f"  Small (<20 images):   {small_scenes:4d} scenes")
        print(f"  Medium (20-60):       {medium_scenes:4d} scenes")
        print(f"  Large (>60):          {large_scenes:4d} scenes")
        print()
        
        # Warnings for large scenes
        if large_scenes > 0:
            self.warnings.append(
                f"{large_scenes} scenes have >60 images. "
                f"Sliding window will be automatically used (may take longer)."
            )
    
    def _print_summary(self, total_scenes: int, valid_scenes: int):
        """Print verification summary."""
        print("="*80)
        print("VERIFICATION SUMMARY")
        print("="*80)
        
        print(f"Total scenes discovered: {total_scenes}")
        print(f"Valid scenes:            {valid_scenes}")
        print(f"Invalid scenes:          {total_scenes - valid_scenes}")
        print()
        
        if self.stats['valid_scenes'] > 0:
            print(f"Total images:            {self.stats['total_images']}")
            print(f"Average images/scene:    {self.stats['total_images'] / self.stats['valid_scenes']:.1f}")
            print()
        
        # Print issues
        if len(self.issues) > 0:
            print("❌ CRITICAL ISSUES:")
            for issue in self.issues:
                print(f"  - {issue}")
            print()
        
        # Print warnings
        if len(self.warnings) > 0:
            print("⚠️  WARNINGS:")
            for warning in self.warnings:
                print(f"  - {warning}")
            print()
        
        # Print recommendations
        if self.stats['empty_scenes'] > 0:
            print(f"ℹ️  {self.stats['empty_scenes']} scenes have no images (will be skipped)")
        
        if self.stats['single_image_scenes'] > 0:
            print(f"ℹ️  {self.stats['single_image_scenes']} scenes have only 1 image (need at least 2, will be skipped)")
        
        if self.stats['large_scenes'] > 0:
            print(f"ℹ️  {self.stats['large_scenes']} large scenes detected. Processing will use sliding window.")
        
        print()
        
        # Estimate processing time
        if valid_scenes > 0:
            self._estimate_processing_time(valid_scenes)
        
        # Final verdict
        print("="*80)
        if len(self.issues) == 0:
            print("✅ VERIFICATION PASSED")
            print("   Dataset structure is valid and ready for batch processing.")
        else:
            print("❌ VERIFICATION FAILED")
            print("   Please fix the issues above before proceeding.")
        print("="*80)
    
    def _estimate_processing_time(self, num_scenes: int):
        """Estimate total processing time."""
        # Rough estimates based on A100
        avg_time_per_scene = self.stats.get('mean_images', 40) * 3  # ~3 seconds per image
        total_seconds = num_scenes * avg_time_per_scene
        total_hours = total_seconds / 3600
        
        print("ESTIMATED PROCESSING TIME:")
        print(f"  Scenes:          {num_scenes}")
        print(f"  Avg time/scene:  {avg_time_per_scene:.0f} seconds")
        print(f"  Total time:      ~{total_hours:.1f} hours")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Verify dataset structure before batch processing"
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Path to dataset root directory",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        choices=['scannet', 'megadepth'],
        help="Name of dataset (scannet or megadepth)",
    )
    
    args = parser.parse_args()
    
    # Run verification
    verifier = DatasetVerifier(
        dataset_root=Path(args.dataset_root),
        dataset_name=args.dataset_name,
    )
    
    success = verifier.verify()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

