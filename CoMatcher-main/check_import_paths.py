"""
Import Path Verification Script
================================

Verifies that VGGT can be imported correctly from all batch processing scripts.

Usage:
    python check_import_paths.py
"""

import sys
from pathlib import Path


def check_vggt_import():
    """Verify VGGT import paths."""
    
    print("="*80)
    print("VGGT Import Path Verification")
    print("="*80)
    print()
    
    # Get project structure
    current_file = Path(__file__)
    comatcher_dir = current_file.parent
    vcomatcher_dir = comatcher_dir.parent
    vggt_dir = vcomatcher_dir / "vggt-main"
    
    print("📁 Directory Structure:")
    print(f"  Current file:     {current_file}")
    print(f"  CoMatcher-main:   {comatcher_dir}")
    print(f"  VCoMatcher:       {vcomatcher_dir}")
    print(f"  vggt-main:        {vggt_dir}")
    print()
    
    # Check if vggt-main exists
    print("🔍 Checking VGGT directory...")
    if not vggt_dir.exists():
        print(f"  ❌ ERROR: vggt-main not found at {vggt_dir}")
        print(f"  Expected structure:")
        print(f"    VCoMatcher/")
        print(f"    ├── vggt-main/")
        print(f"    └── CoMatcher-main/")
        return False
    else:
        print(f"  ✓ vggt-main exists at {vggt_dir}")
    
    # Check if vggt module exists
    vggt_module = vggt_dir / "vggt"
    if not vggt_module.exists():
        print(f"  ❌ ERROR: vggt module not found at {vggt_module}")
        return False
    else:
        print(f"  ✓ vggt module exists at {vggt_module}")
    print()
    
    # Add VGGT to path
    print("📦 Adding VGGT to sys.path...")
    sys.path.insert(0, str(vggt_dir))
    print(f"  ✓ Added: {vggt_dir}")
    print()
    
    # Try to import VGGT
    print("🔄 Testing VGGT import...")
    try:
        from vggt.models.vggt import VGGT
        print(f"  ✓ Successfully imported VGGT model")
        print(f"  ✓ Module location: {VGGT.__module__}")
        print()
        return True
    except ImportError as e:
        print(f"  ❌ ERROR: Failed to import VGGT")
        print(f"  Error: {e}")
        print()
        return False


def check_batch_scripts():
    """Check import paths in batch processing scripts."""
    
    print("="*80)
    print("Batch Processing Scripts Path Check")
    print("="*80)
    print()
    
    scripts = [
        "batch_process_datasets.py",
        "vcomatcher_phase1_data_engine.py",
    ]
    
    current_dir = Path(__file__).parent
    
    all_correct = True
    
    for script_name in scripts:
        script_path = current_dir / script_name
        
        print(f"📝 Checking: {script_name}")
        
        if not script_path.exists():
            print(f"  ⚠️  Script not found: {script_path}")
            continue
        
        # Read script and find sys.path.insert line
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find the vggt-main path line
        import re
        pattern = r'sys\.path\.insert\(0,\s*str\(Path\(__file__\)((?:\.parent)*)\s*/\s*["\']vggt-main["\']\)\)'
        matches = re.findall(pattern, content)
        
        if not matches:
            print(f"  ⚠️  No sys.path.insert for vggt-main found")
            all_correct = False
            continue
        
        # Count .parent occurrences
        parent_count = matches[0].count('.parent')
        
        print(f"  Found: Path(__file__){matches[0]} / 'vggt-main'")
        print(f"  .parent count: {parent_count}")
        
        # Should be 2 (.parent.parent)
        if parent_count == 2:
            print(f"  ✓ Correct path (expected 2 .parent)")
        else:
            print(f"  ❌ INCORRECT path (expected 2 .parent, got {parent_count})")
            all_correct = False
        
        print()
    
    return all_correct


def main():
    """Run all checks."""
    
    print()
    print("🔧 VCoMatcher Import Path Verification")
    print()
    
    # Check VGGT import
    vggt_ok = check_vggt_import()
    
    # Check batch scripts
    scripts_ok = check_batch_scripts()
    
    # Final result
    print("="*80)
    print("FINAL RESULT")
    print("="*80)
    
    if vggt_ok and scripts_ok:
        print("✅ All checks passed!")
        print("   VGGT can be imported correctly from all scripts.")
        return 0
    else:
        print("❌ Some checks failed!")
        if not vggt_ok:
            print("   - VGGT import failed")
        if not scripts_ok:
            print("   - Batch script paths are inconsistent")
        print()
        print("💡 Expected directory structure:")
        print("   VCoMatcher/")
        print("   ├── vggt-main/")
        print("   │   └── vggt/")
        print("   │       └── models/")
        print("   │           └── vggt.py")
        print("   └── CoMatcher-main/")
        print("       ├── batch_process_datasets.py")
        print("       └── vcomatcher_phase1_data_engine.py")
        return 1


if __name__ == "__main__":
    exit(main())

