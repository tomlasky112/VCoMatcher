"""
Batch Processing Progress Monitor
==================================

Real-time monitoring tool for batch processing jobs.
Displays current status, progress, and estimated completion time.

Usage:
    # Monitor with auto-refresh
    python monitor_batch_progress.py --log_dir ./logs/batch_processing

    # One-time check
    python monitor_batch_progress.py --log_dir ./logs/batch_processing --once

    # Custom refresh interval
    python monitor_batch_progress.py --log_dir ./logs/batch_processing --interval 30
"""

import argparse
import json
import time
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional


class ProgressMonitor:
    """Monitor batch processing progress."""
    
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.checkpoint_path = log_dir / "checkpoint.json"
        self.report_pattern = "report_*.json"
    
    def load_checkpoint(self) -> Optional[Dict]:
        """Load current checkpoint."""
        if not self.checkpoint_path.exists():
            return None
        
        try:
            with open(self.checkpoint_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error: Checkpoint file is corrupted: {e}")
            return None
        except FileNotFoundError:
            # Normal case: file doesn't exist yet
            return None
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return None
    
    def load_latest_report(self) -> Optional[Dict]:
        """Load the most recent report file."""
        report_files = sorted(self.log_dir.glob(self.report_pattern))
        
        if len(report_files) == 0:
            return None
        
        latest_report = report_files[-1]
        
        try:
            with open(latest_report, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error: Report file is corrupted: {e}")
            return None
        except Exception as e:
            print(f"Error loading report: {e}")
            return None
    
    def find_latest_log(self) -> Optional[Path]:
        """Find the most recent log file."""
        log_files = sorted(self.log_dir.glob("batch_processing_*.log"))
        
        if len(log_files) == 0:
            return None
        
        return log_files[-1]
    
    def parse_log_for_current_scene(self, log_path: Path) -> Optional[str]:
        """Parse log file to find currently processing scene."""
        try:
            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            # Search backwards for most recent "Processing:" line
            for line in reversed(lines[-100:]):  # Check last 100 lines
                if "Processing:" in line:
                    # Extract scene name
                    parts = line.split("Processing:")
                    if len(parts) > 1:
                        scene_name = parts[1].strip()
                        return scene_name
            
            return None
        except Exception as e:
            return None
    
    def estimate_completion_time(
        self,
        checkpoint: Dict,
        total_scenes: int,
    ) -> Optional[datetime]:
        """Estimate completion time based on current progress."""
        try:
            processed = checkpoint.get('total_processed', 0)
            timestamp_str = checkpoint.get('timestamp', None)
            
            if processed == 0 or timestamp_str is None:
                return None
            
            # Parse checkpoint timestamp
            checkpoint_time = datetime.fromisoformat(timestamp_str)
            
            # Calculate elapsed time (rough estimate)
            now = datetime.now()
            
            # If we have a report, use its data
            report = self.load_latest_report()
            if report and 'summary' in report:
                elapsed_seconds = report['summary'].get('total_time_hours', 0) * 3600
                avg_time_per_scene = report['summary'].get('average_time_per_scene', 0)
            else:
                # Estimate from checkpoint age
                elapsed_seconds = (now - checkpoint_time).total_seconds()
                avg_time_per_scene = elapsed_seconds / max(1, processed)
            
            # Estimate remaining time
            remaining_scenes = total_scenes - processed
            estimated_remaining_seconds = remaining_scenes * avg_time_per_scene
            
            completion_time = now + timedelta(seconds=estimated_remaining_seconds)
            
            return completion_time
            
        except Exception as e:
            return None
    
    def display_status(self, clear_screen: bool = True):
        """Display current processing status."""
        if clear_screen and os.name != 'nt':  # Clear screen on Unix-like systems
            os.system('clear')
        elif clear_screen and os.name == 'nt':  # Clear screen on Windows
            os.system('cls')
        
        print("="*80)
        print("VCoMatcher Batch Processing - Progress Monitor")
        print("="*80)
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Log directory: {self.log_dir}")
        print()
        
        # Load checkpoint
        checkpoint = self.load_checkpoint()
        
        if checkpoint is None:
            print("❌ No checkpoint found. Processing may not have started yet.")
            print()
            print("Waiting for batch processing to begin...")
            return
        
        # Load report
        report = self.load_latest_report()
        
        # Display checkpoint info
        processed_scenes = checkpoint.get('total_processed', 0)
        checkpoint_time = checkpoint.get('timestamp', 'Unknown')
        
        print(f"📊 CHECKPOINT STATUS")
        print(f"─"*80)
        print(f"Processed scenes:  {processed_scenes}")
        print(f"Last update:       {checkpoint_time}")
        print()
        
        # Display detailed stats from report
        if report and 'summary' in report:
            summary = report['summary']
            
            total = summary.get('total_scenes', 0)
            successful = summary.get('successful_scenes', 0)
            failed = summary.get('failed_scenes', 0)
            skipped = summary.get('skipped_scenes', 0)
            success_rate = summary.get('success_rate', 0)
            
            elapsed_hours = summary.get('total_time_hours', 0)
            avg_time = summary.get('average_time_per_scene', 0)
            
            print(f"📈 PROCESSING STATISTICS")
            print(f"─"*80)
            print(f"Total scenes:      {total}")
            print(f"  ✓ Successful:    {successful}")
            print(f"  ✗ Failed:        {failed}")
            print(f"  ⊘ Skipped:       {skipped}")
            print(f"Success rate:      {success_rate:.1f}%")
            print()
            print(f"Elapsed time:      {elapsed_hours:.2f} hours")
            print(f"Avg time/scene:    {avg_time:.1f} seconds")
            print()
            
            # Progress bar
            if total > 0:
                progress_pct = processed_scenes / total * 100
                bar_length = 50
                filled_length = int(bar_length * processed_scenes / total)
                bar = '█' * filled_length + '░' * (bar_length - filled_length)
                
                print(f"Progress:          [{bar}] {progress_pct:.1f}%")
                print()
            
            # Estimate completion time
            completion_time = self.estimate_completion_time(checkpoint, total)
            if completion_time:
                remaining = completion_time - datetime.now()
                remaining_hours = remaining.total_seconds() / 3600
                
                print(f"⏱️  TIME ESTIMATE")
                print(f"─"*80)
                print(f"Remaining time:    ~{remaining_hours:.1f} hours")
                print(f"Est. completion:   {completion_time.strftime('%Y-%m-%d %H:%M:%S')}")
                print()
            
            # Dataset breakdown
            if 'datasets' in report:
                print(f"📂 DATASET BREAKDOWN")
                print(f"─"*80)
                
                for dataset_name, stats in report['datasets'].items():
                    successful_ds = stats.get('successful', 0)
                    total_ds = stats.get('total', 0)
                    total_images = stats.get('total_images', 0)
                    total_samples = stats.get('total_samples', 0)
                    
                    print(f"{dataset_name.upper()}:")
                    print(f"  Processed:       {successful_ds}/{total_ds} scenes")
                    print(f"  Images:          {total_images}")
                    print(f"  Samples:         {total_samples}")
                print()
        
        # Find current scene
        log_path = self.find_latest_log()
        if log_path:
            current_scene = self.parse_log_for_current_scene(log_path)
            if current_scene:
                print(f"🔄 CURRENTLY PROCESSING")
                print(f"─"*80)
                print(f"Scene: {current_scene}")
                print()
        
        # Recent failures
        if report and 'results' in report:
            recent_failures = [
                r for r in report['results'][-20:]  # Last 20 results
                if r['status'] == 'failed'
            ]
            
            if len(recent_failures) > 0:
                print(f"⚠️  RECENT FAILURES ({len(recent_failures)})")
                print(f"─"*80)
                for result in recent_failures[-5:]:  # Show last 5 failures
                    scene_name = result['scene_name']
                    error = result.get('error_message', 'Unknown error')
                    # Truncate long errors
                    if len(error) > 60:
                        error = error[:60] + "..."
                    print(f"  {scene_name}: {error}")
                print()
        
        print("="*80)
        print("Press Ctrl+C to exit")
        print("="*80)
    
    def run(self, interval: int = 10, once: bool = False):
        """
        Run monitoring loop.
        
        Args:
            interval: Refresh interval in seconds
            once: If True, run once and exit
        """
        if once:
            self.display_status(clear_screen=False)
            return
        
        print(f"Monitoring batch processing (refresh every {interval}s)...")
        print()
        
        try:
            while True:
                self.display_status()
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped.")


def main():
    parser = argparse.ArgumentParser(
        description="Monitor VCoMatcher batch processing progress"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./logs/batch_processing",
        help="Path to log directory (default: ./logs/batch_processing)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=10,
        help="Refresh interval in seconds (default: 10)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Display status once and exit (no auto-refresh)",
    )
    
    args = parser.parse_args()
    
    log_dir = Path(args.log_dir)
    
    if not log_dir.exists():
        print(f"Error: Log directory does not exist: {log_dir}")
        print()
        print("Make sure batch processing has started and the log directory is correct.")
        sys.exit(1)
    
    monitor = ProgressMonitor(log_dir)
    monitor.run(interval=args.interval, once=args.once)


if __name__ == "__main__":
    main()

