"""
VCoMatcher Phase 1: Production-Grade Batch Processing Script
=============================================================

Automatically processes ScanNet and MegaDepth datasets on 80GB A100 GPU.

Features:
- Auto-discovery of scenes in dataset directories
- Sliding window processing for large scenes (OOM prevention)
- Resume capability (skip already processed scenes)
- Parallel processing with error isolation
- Comprehensive logging and progress tracking
- Quality validation after each scene
- Final summary report with statistics

Author: VCoMatcher Team
Date: 2025-12-25
Version: 1.2.1
"""

import os
import sys
import json
import time
import argparse
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
import numpy as np
import torch
from tqdm import tqdm
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "vggt-main"))

from vcomatcher_phase1_data_engine import VGGTDataEngine


@dataclass
class SceneResult:
    """Result of processing a single scene."""
    scene_name: str
    dataset: str  # 'scannet' or 'megadepth'
    status: str  # 'success', 'failed', 'skipped'
    num_images: int = 0
    processing_time: float = 0.0
    output_path: str = ""
    error_message: str = ""
    timestamp: str = ""
    
    # Quality metrics
    num_samples: int = 0
    num_easy: int = 0
    num_hard: int = 0
    num_extreme: int = 0
    mask_loss_coverage: float = 0.0
    mask_geom_coverage: float = 0.0


@dataclass
class DatasetConfig:
    """Configuration for a specific dataset."""
    name: str
    root_dir: Path
    output_dir: Path
    scene_dirs: List[Path]
    tau_min: float
    tau_max: float
    tau_uncertainty: float
    pnp_tau: float


class BatchProcessor:
    """
    Production-grade batch processor for VCoMatcher Phase 1.
    """
    
    def __init__(
        self,
        device: str = "cuda",
        log_dir: Path = Path("./logs"),
        resume: bool = True,
        validate_quality: bool = True,
        max_workers: int = 1,  # Number of parallel scenes (set to 1 for A100)
    ):
        """
        Initialize batch processor.
        
        Args:
            device: Computing device ('cuda' or 'cpu')
            log_dir: Directory for logs and checkpoints
            resume: Skip already processed scenes
            validate_quality: Run quality validation after each scene
            max_workers: Number of parallel workers (1 for single GPU)
        """
        self.device = device
        self.log_dir = Path(log_dir)
        self.resume = resume
        self.validate_quality = validate_quality
        self.max_workers = max_workers
        
        # Create directories
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Results tracking
        self.results: List[SceneResult] = []
        self.checkpoint_path = self.log_dir / "checkpoint.json"
        self.processed_scenes = self._load_checkpoint()
        
        # Statistics
        self.start_time = None
        self.total_scenes = 0
        self.successful_scenes = 0
        self.failed_scenes = 0
        self.skipped_scenes = 0
        
    def _setup_logging(self):
        """Setup comprehensive logging with tqdm compatibility."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"batch_processing_{timestamp}.log"
        
        # FIXED v1.2: Configure logging to be compatible with tqdm progress bars
        # Use file handler for detailed logs, minimal console output
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file),
            ]
        )
        
        # Add console handler with higher level (only warnings and errors)
        # This prevents log spam that conflicts with tqdm progress bars
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.WARNING)
        console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(console_handler)
        
        # Initial info (to file only)
        self.logger.info("="*80)
        self.logger.info("VCoMatcher Phase 1 Batch Processing Started")
        self.logger.info("="*80)
        self.logger.info(f"Log file: {log_file}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Resume mode: {self.resume}")
        self.logger.info(f"Quality validation: {self.validate_quality}")
        
        # Print to console (outside logger to avoid tqdm conflict)
        print("="*80)
        print("VCoMatcher Phase 1 Batch Processing Started")
        print("="*80)
        print(f"Log file: {log_file}")
        print(f"Device: {self.device}")
        print(f"Resume mode: {self.resume}")
        print(f"Quality validation: {self.validate_quality}")
        print(f"\n💡 Detailed logs are written to: {log_file}")
        print(f"   Console shows only progress bar and critical messages\n")
        
    def _load_checkpoint(self) -> set:
        """Load checkpoint of already processed scenes."""
        if not self.checkpoint_path.exists():
            return set()
        
        try:
            with open(self.checkpoint_path, 'r') as f:
                checkpoint = json.load(f)
                processed = set(checkpoint.get('processed_scenes', []))
                self.logger.info(f"Loaded checkpoint: {len(processed)} scenes already processed")
                return processed
        except Exception as e:
            self.logger.warning(f"Failed to load checkpoint: {e}")
            return set()
    
    def _save_checkpoint(self):
        """Save checkpoint with list of processed scenes."""
        checkpoint = {
            'processed_scenes': list(self.processed_scenes),
            'timestamp': datetime.now().isoformat(),
            'total_processed': len(self.processed_scenes),
        }
        
        with open(self.checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
    
    def discover_scenes(
        self,
        dataset_root: Path,
        dataset_name: str,
        max_depth: int = 10,  # Prevent infinite recursion
    ) -> List[Path]:
        """
        Automatically discover scene directories in a dataset.
        
        CRITICAL FIX v1.2: Support both 'images' and 'imgs' folder names
        - ScanNet typically uses: scene/images/
        - MegaDepth typically uses: scene/imgs/ or scene/images/
        
        Also supports deep nesting like:
            root/Phoenix/S6/zl548/MegaDepth_v1/0019/dense0/imgs/
        
        Args:
            dataset_root: Root directory of dataset
            dataset_name: Name of dataset ('scannet' or 'megadepth')
            max_depth: Maximum directory depth to search (default: 10)
            
        Returns:
            List of scene directories (parent of 'images/' or 'imgs/' folder)
        """
        self.logger.info(f"\nDiscovering scenes in {dataset_name}: {dataset_root}")
        
        if not dataset_root.exists():
            self.logger.error(f"Dataset root does not exist: {dataset_root}")
            return []
        
        # FIXED v1.2: Search for both 'images' and 'imgs' patterns
        patterns = ["**/images", "**/imgs"]
        self.logger.info(f"Scanning directory structure (patterns: {patterns}, max_depth: {max_depth})...")
        
        # Find all image directories (supports deep nesting)
        image_dirs = []
        try:
            for pattern in patterns:
                for img_dir in dataset_root.glob(pattern):
                    # Check depth to avoid infinite recursion
                    depth = len(img_dir.relative_to(dataset_root).parts)
                    if depth <= max_depth and img_dir.is_dir():
                        # Check if directory has actual images
                        has_images = any(
                            f.suffix.lower() in ['.jpg', '.jpeg', '.png'] 
                            for f in img_dir.iterdir() 
                            if f.is_file()
                        )
                        if has_images:
                            image_dirs.append(img_dir)
        except Exception as e:
            self.logger.error(f"Error during scene discovery: {e}")
            return []
        
        # Get parent directories (scene directories)
        scene_dirs = [img_dir.parent for img_dir in image_dirs]
        scene_dirs = sorted(set(scene_dirs))  # Remove duplicates
        
        self.logger.info(f"Found {len(scene_dirs)} scenes in {dataset_name}")
        
        # Filter out already processed scenes if resume mode
        if self.resume:
            original_count = len(scene_dirs)
            scene_dirs = [
                s for s in scene_dirs 
                if f"{dataset_name}_{s.name}" not in self.processed_scenes
            ]
            skipped = original_count - len(scene_dirs)
            if skipped > 0:
                self.logger.info(f"Skipping {skipped} already processed scenes")
        
        return scene_dirs
    
    def create_dataset_config(
        self,
        dataset_name: str,
        dataset_root: Path,
        output_root: Path,
        **kwargs
    ) -> DatasetConfig:
        """
        Create configuration for a specific dataset.
        
        Args:
            dataset_name: Name of dataset ('scannet' or 'megadepth')
            dataset_root: Root directory of dataset
            output_root: Root output directory
            **kwargs: Additional parameters (tau_min, tau_max, etc.)
            
        Returns:
            DatasetConfig object
        """
        # Discover scenes
        scene_dirs = self.discover_scenes(dataset_root, dataset_name)
        
        # Dataset-specific defaults
        if dataset_name.lower() == 'scannet':
            defaults = {
                'tau_min': 0.1,
                'tau_max': 10.0,  # Indoor scenes have shorter depth range
                'tau_uncertainty': 15.0,
                'pnp_tau': 6.0,
            }
        elif dataset_name.lower() == 'megadepth':
            defaults = {
                'tau_min': 0.5,
                'tau_max': 100.0,  # Outdoor scenes have longer depth range
                'tau_uncertainty': 15.0,
                'pnp_tau': 6.0,
            }
        else:
            defaults = {
                'tau_min': 0.1,
                'tau_max': 100.0,
                'tau_uncertainty': 15.0,
                'pnp_tau': 6.0,
            }
        
        # Override with user-provided values
        defaults.update(kwargs)
        
        # BUG FIX v1.2.1: Validate parameters
        if defaults['tau_min'] >= defaults['tau_max']:
            raise ValueError(
                f"tau_min ({defaults['tau_min']}) must be less than "
                f"tau_max ({defaults['tau_max']})"
            )
        if defaults['tau_uncertainty'] <= 0:
            raise ValueError(
                f"tau_uncertainty ({defaults['tau_uncertainty']}) must be positive"
            )
        if defaults['pnp_tau'] <= 0:
            raise ValueError(
                f"pnp_tau ({defaults['pnp_tau']}) must be positive"
            )
        
        # Create output directory with permission check
        output_dir = output_root / dataset_name
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            # BUG FIX v1.2.1: Verify write permissions
            test_file = output_dir / ".write_test"
            test_file.touch()
            test_file.unlink()
        except PermissionError as e:
            raise PermissionError(
                f"No write permission for output directory: {output_dir}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to create output directory: {output_dir}"
            ) from e
        
        return DatasetConfig(
            name=dataset_name,
            root_dir=dataset_root,
            output_dir=output_dir,
            scene_dirs=scene_dirs,
            **defaults
        )
    
    def process_scene(
        self,
        scene_dir: Path,
        config: DatasetConfig,
        engine: VGGTDataEngine,
    ) -> SceneResult:
        """
        Process a single scene with comprehensive error handling.
        
        Args:
            scene_dir: Path to scene directory
            config: Dataset configuration
            engine: VGGTDataEngine instance
            
        Returns:
            SceneResult object with processing outcome
        """
        scene_name = scene_dir.name
        scene_id = f"{config.name}_{scene_name}"
        timestamp = datetime.now().isoformat()
        
        # FIXED v1.2: Log to file only, avoid console spam that conflicts with tqdm
        self.logger.info("\n" + "="*80)
        self.logger.info(f"Processing: {scene_id}")
        self.logger.info("="*80)
        
        result = SceneResult(
            scene_name=scene_name,
            dataset=config.name,
            status='failed',
            timestamp=timestamp,
        )
        
        try:
            # Check if already processed
            if self.resume and scene_id in self.processed_scenes:
                self.logger.info(f"Skipping (already processed): {scene_id}")
                result.status = 'skipped'
                return result
            
            # BUG FIX v1.2.1: Check for both 'images' and 'imgs' directories
            # Must match the discover_scenes logic
            image_dir = scene_dir / "images"
            if not image_dir.exists():
                image_dir = scene_dir / "imgs"
                if not image_dir.exists():
                    raise FileNotFoundError(
                        f"Neither 'images' nor 'imgs' directory found in {scene_dir}"
                    )
            
            # BUG FIX v1.2.1: Use explicit glob patterns to avoid non-image files
            image_files = []
            for pattern in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                image_files.extend(list(image_dir.glob(pattern)))
            result.num_images = len(image_files)
            
            if result.num_images < 2:
                raise ValueError(f"Not enough images: {result.num_images} (need at least 2)")
            
            self.logger.info(f"Found {result.num_images} images")
            
            # Process scene
            start_time = time.time()
            
            success = engine.process_scene(
                scene_dir=scene_dir,
                output_dir=config.output_dir,
            )
            
            elapsed_time = time.time() - start_time
            result.processing_time = elapsed_time
            
            if not success:
                raise RuntimeError("Scene processing failed (returned False)")
            
            # Check output file
            output_path = config.output_dir / f"{scene_name}.npz"
            if not output_path.exists():
                raise FileNotFoundError(f"Output file not created: {output_path}")
            
            result.output_path = str(output_path)
            
            # CRITICAL FIX v1.2: .npz files are ALREADY lazy-loaded by default!
            # np.savez_compressed creates compressed .npz files, which:
            # 1. Are incompatible with mmap_mode (will raise ValueError or fail silently)
            # 2. Only load arrays when accessed (lazy loading is built-in)
            # 
            # Solution: Use default np.load() without mmap_mode
            # - File header is read (lightweight)
            # - Arrays are only decompressed when accessed via data['key']
            # - No need for explicit mmap or file handle management
            with np.load(output_path, allow_pickle=True) as data:
                # Extract quality metrics (these are small, quick to compute)
                # Only these arrays will be decompressed, not the large ones (points_3d, depth)
                result.mask_loss_coverage = float(data['mask_loss'].sum() / data['mask_loss'].size * 100)
                result.mask_geom_coverage = float(data['mask_geom'].sum() / data['mask_geom'].size * 100)
                
                samples = data['samples'].item()  # Dict
                result.num_samples = len(samples)
                result.num_easy = sum(1 for s in samples.values() if s['difficulty'] == 'easy')
                result.num_hard = sum(1 for s in samples.values() if s['difficulty'] == 'hard')
                result.num_extreme = sum(1 for s in samples.values() if s['difficulty'] == 'extreme')
                
                # Validate quality if enabled (data context is still open)
                if self.validate_quality:
                    self._validate_scene_quality_fast(data, result)
            
            # Mark as successful
            result.status = 'success'
            self.processed_scenes.add(scene_id)
            self._save_checkpoint()
            
            self.logger.info(f"✓ Scene processed successfully in {elapsed_time:.1f}s")
            self.logger.info(f"  - Samples: {result.num_samples} (Easy: {result.num_easy}, Hard: {result.num_hard}, Extreme: {result.num_extreme})")
            self.logger.info(f"  - mask_loss coverage: {result.mask_loss_coverage:.1f}%")
            self.logger.info(f"  - mask_geom coverage: {result.mask_geom_coverage:.1f}%")
            
        except Exception as e:
            result.status = 'failed'
            result.error_message = str(e)
            self.logger.error(f"✗ Scene processing failed: {scene_id}")
            self.logger.error(f"  Error: {e}")
            self.logger.debug(traceback.format_exc())
            
        return result
    
    def _validate_scene_quality_fast(self, data: np.lib.npyio.NpzFile, result: SceneResult):
        """
        Quick quality validation using lazy-loaded .npz data.
        
        Because .npz files from np.savez_compressed are lazy-loaded by default,
        only the arrays we actually access will be decompressed.
        
        Args:
            data: Already-opened np.load() context manager
            result: SceneResult object to update with quality metrics
        """
        try:
            # Check minimum coverage (already computed in result)
            if result.mask_loss_coverage < 30:
                self.logger.warning(f"  ⚠️  Low mask_loss coverage: {result.mask_loss_coverage:.1f}% (expected >60%)")
            
            # Check sample distribution (already computed in result)
            if result.num_samples == 0:
                self.logger.warning(f"  ⚠️  No valid samples found (overlap out of range)")
            
            # Check depth validity (only if critical, as this requires decompressing depth array)
            # OPTIMIZATION: Skip depth check if mask_loss is reasonable (avoids loading large depth array)
            if result.mask_loss_coverage < 30:
                # This will decompress the depth array, but only if needed
                depth = data['depth']
                valid_depth_ratio = np.sum((depth > 0.1) & (depth < 100)) / depth.size * 100
                if valid_depth_ratio < 50:
                    self.logger.warning(f"  ⚠️  Low valid depth ratio: {valid_depth_ratio:.1f}%")
                
        except Exception as e:
            self.logger.warning(f"  Quality validation failed: {e}")
    
    def process_dataset(
        self,
        config: DatasetConfig,
    ) -> List[SceneResult]:
        """
        Process all scenes in a dataset.
        
        Args:
            config: DatasetConfig object
            
        Returns:
            List of SceneResult objects
        """
        self.logger.info("\n" + "="*80)
        self.logger.info(f"Processing Dataset: {config.name.upper()}")
        self.logger.info("="*80)
        self.logger.info(f"Root: {config.root_dir}")
        self.logger.info(f"Output: {config.output_dir}")
        self.logger.info(f"Scenes: {len(config.scene_dirs)}")
        self.logger.info(f"Parameters:")
        self.logger.info(f"  - tau_min: {config.tau_min}")
        self.logger.info(f"  - tau_max: {config.tau_max}")
        self.logger.info(f"  - tau_uncertainty: {config.tau_uncertainty}")
        self.logger.info(f"  - pnp_tau: {config.pnp_tau}")
        
        if len(config.scene_dirs) == 0:
            self.logger.warning(f"No scenes to process in {config.name}")
            return []
        
        # Initialize data engine
        self.logger.info(f"\nInitializing VGGT Data Engine...")
        engine = VGGTDataEngine(
            device=self.device,
            tau_min=config.tau_min,
            tau_max=config.tau_max,
            tau_uncertainty=config.tau_uncertainty,
            pnp_tau=config.pnp_tau,
        )
        
        # Process scenes
        dataset_results = []
        
        # FIXED v1.2: Use tqdm with better formatting to avoid log conflicts
        progress_bar = tqdm(
            config.scene_dirs,
            desc=f"Processing {config.name}",
            position=0,
            leave=True,
            ncols=100,  # Fixed width
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        )
        
        for scene_dir in progress_bar:
            result = self.process_scene(scene_dir, config, engine)
            dataset_results.append(result)
            self.results.append(result)
            
            # Update progress bar with current scene info
            progress_bar.set_postfix_str(f"Last: {scene_dir.name[:20]}", refresh=False)
            
            # Update statistics
            self.total_scenes += 1
            if result.status == 'success':
                self.successful_scenes += 1
            elif result.status == 'failed':
                self.failed_scenes += 1
            elif result.status == 'skipped':
                self.skipped_scenes += 1
            
            # Periodic progress report
            if self.total_scenes % 10 == 0:
                self._print_progress()
            
            # OPTIMIZATION FIX: Smart GPU memory management
            # Clear cache based on memory usage and scene size, not fixed interval
            if self.device == 'cuda':
                self._smart_gpu_cleanup(result)
        
        return dataset_results
    
    def _smart_gpu_cleanup(self, result: SceneResult):
        """
        OPTIMIZATION FIX: Intelligent GPU memory management.
        
        Instead of fixed-interval cleanup (every N scenes), use dynamic strategy:
        1. Always cleanup after large scenes (>60 images) or Extreme samples
        2. Cleanup when GPU memory usage exceeds threshold (>70%)
        3. Periodic cleanup as fallback (every 5 scenes)
        
        This prevents OOM on MegaDepth Extreme samples while avoiding
        unnecessary overhead on small scenes.
        
        Args:
            result: SceneResult from just-completed scene
        """
        should_cleanup = False
        reason = ""
        
        # Strategy 1: Large scenes leave memory fragmentation
        if result.num_images > 60:
            should_cleanup = True
            reason = f"large scene ({result.num_images} images)"
        
        # Strategy 2: Extreme samples may have used sliding window
        elif result.num_extreme > 0:
            should_cleanup = True
            reason = f"extreme samples ({result.num_extreme})"
        
        # Strategy 3: Check GPU memory usage (if available)
        # BUG FIX v1.2.1: Add proper error handling for GPU checks
        elif torch.cuda.is_available():
            try:
                # Get current GPU memory usage
                gpu_mem_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                gpu_mem_reserved = torch.cuda.memory_reserved() / 1024**3    # GB
                gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                
                usage_ratio = gpu_mem_reserved / gpu_mem_total
                
                # Cleanup if >70% reserved (indicates fragmentation)
                if usage_ratio > 0.7:
                    should_cleanup = True
                    reason = f"high GPU memory ({usage_ratio*100:.1f}% reserved)"
            except RuntimeError as e:
                # CUDA error (e.g., driver issue)
                self.logger.warning(f"Failed to check GPU memory: {e}")
                pass
            except Exception as e:
                # Unexpected error, log but don't fail
                self.logger.debug(f"GPU memory check failed: {e}")
                pass
        
        # Strategy 4: Periodic cleanup as fallback
        if not should_cleanup and self.total_scenes % 5 == 0:
            should_cleanup = True
            reason = "periodic cleanup"
        
        # Execute cleanup
        if should_cleanup:
            torch.cuda.empty_cache()
            self.logger.debug(f"  🧹 GPU cache cleared: {reason}")
    
    def _print_progress(self):
        """Print current progress statistics."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        success_rate = self.successful_scenes / max(1, self.total_scenes - self.skipped_scenes) * 100
        
        self.logger.info("\n" + "-"*80)
        self.logger.info(f"PROGRESS REPORT")
        self.logger.info("-"*80)
        self.logger.info(f"Elapsed time: {elapsed/3600:.1f} hours")
        self.logger.info(f"Total processed: {self.total_scenes}")
        self.logger.info(f"  ✓ Successful: {self.successful_scenes}")
        self.logger.info(f"  ✗ Failed: {self.failed_scenes}")
        self.logger.info(f"  ⊘ Skipped: {self.skipped_scenes}")
        self.logger.info(f"Success rate: {success_rate:.1f}%")
        self.logger.info("-"*80)
    
    def generate_report(self, output_path: Path = None):
        """
        Generate comprehensive processing report.
        
        Args:
            output_path: Path to save report (default: log_dir/report.json)
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.log_dir / f"report_{timestamp}.json"
        
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        
        # Aggregate statistics by dataset
        datasets_stats = {}
        for result in self.results:
            if result.dataset not in datasets_stats:
                datasets_stats[result.dataset] = {
                    'total': 0,
                    'successful': 0,
                    'failed': 0,
                    'skipped': 0,
                    'total_images': 0,
                    'total_samples': 0,
                    'total_time': 0.0,
                }
            
            stats = datasets_stats[result.dataset]
            stats['total'] += 1
            if result.status == 'success':
                stats['successful'] += 1
                stats['total_images'] += result.num_images
                stats['total_samples'] += result.num_samples
                stats['total_time'] += result.processing_time
            elif result.status == 'failed':
                stats['failed'] += 1
            elif result.status == 'skipped':
                stats['skipped'] += 1
        
        # Create report
        report = {
            'summary': {
                'total_scenes': self.total_scenes,
                'successful_scenes': self.successful_scenes,
                'failed_scenes': self.failed_scenes,
                'skipped_scenes': self.skipped_scenes,
                'success_rate': self.successful_scenes / max(1, self.total_scenes - self.skipped_scenes) * 100,
                'total_time_hours': elapsed_time / 3600,
                'average_time_per_scene': elapsed_time / max(1, self.total_scenes - self.skipped_scenes),
            },
            'datasets': datasets_stats,
            'results': [asdict(r) for r in self.results],
            'timestamp': datetime.now().isoformat(),
        }
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info("\n" + "="*80)
        self.logger.info("FINAL REPORT")
        self.logger.info("="*80)
        self.logger.info(f"Total scenes: {self.total_scenes}")
        self.logger.info(f"  ✓ Successful: {self.successful_scenes}")
        self.logger.info(f"  ✗ Failed: {self.failed_scenes}")
        self.logger.info(f"  ⊘ Skipped: {self.skipped_scenes}")
        self.logger.info(f"Success rate: {report['summary']['success_rate']:.1f}%")
        self.logger.info(f"Total time: {report['summary']['total_time_hours']:.1f} hours")
        self.logger.info(f"Average time per scene: {report['summary']['average_time_per_scene']:.1f} seconds")
        self.logger.info("\nDataset Statistics:")
        for dataset_name, stats in datasets_stats.items():
            self.logger.info(f"\n{dataset_name.upper()}:")
            self.logger.info(f"  Successful: {stats['successful']}/{stats['total']}")
            self.logger.info(f"  Total images: {stats['total_images']}")
            self.logger.info(f"  Total samples: {stats['total_samples']}")
            self.logger.info(f"  Total time: {stats['total_time']/3600:.1f} hours")
        
        self.logger.info(f"\nFull report saved to: {output_path}")
        self.logger.info("="*80)
        
        return report
    
    def run(
        self,
        datasets: List[DatasetConfig],
    ):
        """
        Run batch processing on multiple datasets.
        
        Args:
            datasets: List of DatasetConfig objects
        """
        self.start_time = time.time()
        
        self.logger.info("\n" + "="*80)
        self.logger.info("BATCH PROCESSING START")
        self.logger.info("="*80)
        self.logger.info(f"Datasets: {len(datasets)}")
        self.logger.info(f"Total scenes: {sum(len(d.scene_dirs) for d in datasets)}")
        
        # Process each dataset
        for dataset_config in datasets:
            try:
                self.process_dataset(dataset_config)
            except Exception as e:
                self.logger.error(f"Dataset processing failed: {dataset_config.name}")
                self.logger.error(f"Error: {e}")
                self.logger.debug(traceback.format_exc())
        
        # Generate final report
        self.generate_report()
        
        self.logger.info("\n" + "="*80)
        self.logger.info("BATCH PROCESSING COMPLETE")
        self.logger.info("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="VCoMatcher Phase 1: Batch Processing for ScanNet and MegaDepth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process both ScanNet and MegaDepth
  python batch_process_datasets.py \\
      --scannet_root /path/to/scannet \\
      --megadepth_root /path/to/megadepth \\
      --output_root ./data/vcomatcher_phase1

  # Process only ScanNet
  python batch_process_datasets.py \\
      --scannet_root /path/to/scannet \\
      --output_root ./data/vcomatcher_phase1

  # Resume from checkpoint
  python batch_process_datasets.py \\
      --scannet_root /path/to/scannet \\
      --output_root ./data/vcomatcher_phase1 \\
      --resume
        """
    )
    
    # Dataset paths
    parser.add_argument(
        "--scannet_root",
        type=str,
        default=None,
        help="Path to ScanNet dataset root directory",
    )
    parser.add_argument(
        "--megadepth_root",
        type=str,
        default=None,
        help="Path to MegaDepth dataset root directory",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Root output directory for processed data",
    )
    
    # Processing options
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Computing device (cuda or cpu)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint (skip already processed scenes)",
    )
    parser.add_argument(
        "--no_validation",
        action="store_true",
        help="Disable quality validation after each scene",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./logs/batch_processing",
        help="Directory for logs and checkpoints",
    )
    
    # Dataset-specific parameters
    parser.add_argument(
        "--scannet_tau_min",
        type=float,
        default=0.1,
        help="ScanNet: Minimum depth threshold (default: 0.1)",
    )
    parser.add_argument(
        "--scannet_tau_max",
        type=float,
        default=10.0,
        help="ScanNet: Maximum depth threshold (default: 10.0)",
    )
    parser.add_argument(
        "--scannet_tau_uncertainty",
        type=float,
        default=15.0,
        help="ScanNet: Uncertainty threshold (default: 15.0)",
    )
    parser.add_argument(
        "--scannet_pnp_tau",
        type=float,
        default=6.0,
        help="ScanNet: PnP confidence threshold (default: 6.0)",
    )
    
    parser.add_argument(
        "--megadepth_tau_min",
        type=float,
        default=0.5,
        help="MegaDepth: Minimum depth threshold (default: 0.5)",
    )
    parser.add_argument(
        "--megadepth_tau_max",
        type=float,
        default=100.0,
        help="MegaDepth: Maximum depth threshold (default: 100.0)",
    )
    parser.add_argument(
        "--megadepth_tau_uncertainty",
        type=float,
        default=15.0,
        help="MegaDepth: Uncertainty threshold (default: 15.0)",
    )
    parser.add_argument(
        "--megadepth_pnp_tau",
        type=float,
        default=6.0,
        help="MegaDepth: PnP confidence threshold (default: 6.0)",
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.scannet_root is None and args.megadepth_root is None:
        parser.error("At least one dataset root must be specified (--scannet_root or --megadepth_root)")
    
    # Initialize batch processor
    processor = BatchProcessor(
        device=args.device,
        log_dir=Path(args.log_dir),
        resume=args.resume,
        validate_quality=not args.no_validation,
    )
    
    # Create dataset configurations
    datasets = []
    
    if args.scannet_root:
        scannet_config = processor.create_dataset_config(
            dataset_name='scannet',
            dataset_root=Path(args.scannet_root),
            output_root=Path(args.output_root),
            tau_min=args.scannet_tau_min,
            tau_max=args.scannet_tau_max,
            tau_uncertainty=args.scannet_tau_uncertainty,
            pnp_tau=args.scannet_pnp_tau,
        )
        datasets.append(scannet_config)
    
    if args.megadepth_root:
        megadepth_config = processor.create_dataset_config(
            dataset_name='megadepth',
            dataset_root=Path(args.megadepth_root),
            output_root=Path(args.output_root),
            tau_min=args.megadepth_tau_min,
            tau_max=args.megadepth_tau_max,
            tau_uncertainty=args.megadepth_tau_uncertainty,
            pnp_tau=args.megadepth_pnp_tau,
        )
        datasets.append(megadepth_config)
    
    # Run batch processing
    processor.run(datasets)


if __name__ == "__main__":
    main()

