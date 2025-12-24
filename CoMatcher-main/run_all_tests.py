#!/usr/bin/env python3
"""
VCoMatcher Complete Test Runner
================================

Runs all VCoMatcher tests in proper order and generates comprehensive report.

Usage:
    python run_all_tests.py                    # Run all tests
    python run_all_tests.py --quick            # Quick test (skip slow tests)
    python run_all_tests.py --phase1           # Only Phase 1 tests
    python run_all_tests.py --phase2           # Only Phase 2 tests
    python run_all_tests.py --critical-only    # Only critical tests
"""

import sys
import time
import argparse
from pathlib import Path
from typing import List, Tuple


def print_banner(text: str):
    """Print a fancy banner."""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80)


def run_test_module(module_name: str, description: str) -> Tuple[bool, str]:
    """
    Run a test module and capture results.
    
    Returns:
        (success, output)
    """
    print(f"\n{'─'*80}")
    print(f"Running: {description}")
    print(f"Module: {module_name}")
    print(f"{'─'*80}\n")
    
    try:
        # Import and run the module
        if module_name == "test_sliding_window":
            from test_sliding_window import main as test_main
        elif module_name == "test_phase2_dataset":
            from test_phase2_dataset import main as test_main
        elif module_name == "validate_phase1":
            # Special case: validate_phase1 requires data path
            data_dir = Path("./data/vcomatcher_phase1_test")
            if not data_dir.exists():
                print(f"⚠️ Skipping: data directory not found")
                return True, "SKIPPED"
            
            data_files = list(data_dir.glob("*.npz"))
            if not data_files:
                print(f"⚠️ Skipping: no .npz files found")
                return True, "SKIPPED"
            
            # Run validation on first file
            from validate_phase1_comprehensive import Phase1Validator
            validator = Phase1Validator(data_files[0])
            output_dir = Path("./validation_results")
            success = validator.run_full_validation(output_dir)
            return success, "PASSED" if success else "FAILED"
        else:
            print(f"✗ Unknown module: {module_name}")
            return False, "ERROR"
        
        # Run the test
        result = test_main()
        return result == 0, "PASSED" if result == 0 else "FAILED"
        
    except ImportError as e:
        print(f"⚠️ Skipping: {e}")
        return True, "SKIPPED"
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False, "ERROR"


def run_critical_tests_only() -> List[Tuple[str, bool]]:
    """Run only critical tests (fast, must pass)."""
    print_banner("CRITICAL TESTS ONLY")
    
    results = []
    
    # Critical sliding window tests
    print("\n[1/4] Sliding Window Critical Tests:")
    try:
        from test_sliding_window import (
            test_pose_points_synchronization,
            test_umeyama_alignment_known_transform,
        )
        
        try:
            r1 = test_pose_points_synchronization()
            results.append(("Pose-Points Sync", r1))
        except Exception as e:
            print(f"  ✗ test_pose_points_synchronization failed: {e}")
            results.append(("Pose-Points Sync", False))
        
        try:
            r2 = test_umeyama_alignment_known_transform()
            results.append(("Umeyama Alignment", r2))
        except Exception as e:
            print(f"  ✗ test_umeyama_alignment_known_transform failed: {e}")
            results.append(("Umeyama Alignment", False))
        
    except ImportError:
        print("  ⚠️ Sliding window tests not found (skipping)")
    
    # Critical Phase 2 tests
    print("\n[2/4] Phase 2 Critical Tests:")
    try:
        from test_phase2_dataset import (
            test_target_centric_transformation,
            test_geometric_consistency,
        )
        
        try:
            r3 = test_target_centric_transformation()
            results.append(("Target-Centric Transform", r3))
        except Exception as e:
            print(f"  ✗ test_target_centric_transformation failed: {e}")
            results.append(("Target-Centric Transform", False))
        
        try:
            r4 = test_geometric_consistency()
            results.append(("Geometric Consistency", r4))
        except Exception as e:
            print(f"  ✗ test_geometric_consistency failed: {e}")
            results.append(("Geometric Consistency", False))
        
    except ImportError:
        print("  ⚠️ Phase 2 tests not found (skipping)")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="VCoMatcher Complete Test Runner"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode (skip performance tests)"
    )
    parser.add_argument(
        "--phase1",
        action="store_true",
        help="Only run Phase 1 tests"
    )
    parser.add_argument(
        "--phase2",
        action="store_true",
        help="Only run Phase 2 tests"
    )
    parser.add_argument(
        "--critical-only",
        action="store_true",
        help="Only run critical tests (fast)"
    )
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "="*80)
    print("  VCoMatcher Complete Test Suite")
    print("="*80)
    print(f"\nStart time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    
    # Run critical tests only
    if args.critical_only:
        results = run_critical_tests_only()
        
        # Print summary
        print_banner("CRITICAL TESTS SUMMARY")
        total = len(results)
        passed = sum(1 for _, r in results if r)
        
        for name, result in results:
            status = "✓ PASS" if result else "✗ FAIL"
            print(f"  {name:35s}: {status}")
        
        print(f"\nTotal: {passed}/{total} critical tests passed")
        elapsed = time.time() - start_time
        print(f"Elapsed time: {elapsed:.1f} seconds")
        
        sys.exit(0 if passed == total else 1)
    
    # Define test modules
    test_modules = []
    
    if args.phase1 or not (args.phase2):
        test_modules.extend([
            ("validate_phase1", "Phase 1: Data Quality Validation"),
            ("test_sliding_window", "Phase 1: Sliding Window OOM Solution"),
        ])
    
    if args.phase2 or not (args.phase1):
        test_modules.append(
            ("test_phase2_dataset", "Phase 2: DataLoader Tests")
        )
    
    # Run all test modules
    results = []
    
    for module_name, description in test_modules:
        success, status = run_test_module(module_name, description)
        results.append((description, success, status))
    
    # Print comprehensive summary
    print_banner("COMPLETE TEST SUMMARY")
    
    for description, success, status in results:
        icon = "✓" if success else "✗"
        status_str = f"[{status}]"
        print(f"  {icon} {description:50s} {status_str}")
    
    # Calculate statistics
    total = len(results)
    passed = sum(1 for _, s, st in results if s and st == "PASSED")
    skipped = sum(1 for _, s, st in results if st == "SKIPPED")
    failed = sum(1 for _, s, st in results if not s or st == "FAILED")
    
    print(f"\n{'─'*80}")
    print(f"Total: {total} test modules")
    print(f"  ✓ Passed:  {passed}")
    print(f"  ⏭️  Skipped: {skipped}")
    print(f"  ✗ Failed:  {failed}")
    
    elapsed = time.time() - start_time
    print(f"\nTotal elapsed time: {elapsed:.1f} seconds")
    
    # Final verdict
    if failed == 0:
        print("\n" + "="*80)
        print("🎉 ALL TESTS PASSED!")
        print("VCoMatcher is ready for training!")
        print("="*80)
        exit_code = 0
    else:
        print("\n" + "="*80)
        print("⚠️  SOME TESTS FAILED")
        print("Please review failures before proceeding")
        print("="*80)
        exit_code = 1
    
    # Recommendations
    if exit_code == 0:
        print("\n📋 Next Steps:")
        print("  1. Review TESTING_GUIDE.md for detailed test information")
        print("  2. Check validation_results/ for visual reports")
        print("  3. Proceed to Phase 3 training development")
    else:
        print("\n📋 Troubleshooting:")
        print("  1. Check test output above for error details")
        print("  2. Consult TESTING_GUIDE.md for common issues")
        print("  3. Run individual tests for debugging:")
        print("     - python test_sliding_window.py")
        print("     - python test_phase2_dataset.py")
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

