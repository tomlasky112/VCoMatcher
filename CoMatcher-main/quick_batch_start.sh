#!/bin/bash
################################################################################
# VCoMatcher Phase 1: Quick Batch Processing Launcher
################################################################################
#
# Quick start templates for batch processing ScanNet and MegaDepth datasets
#
# Usage:
#   1. Edit the paths below to match your environment
#   2. Choose a template (uncomment the corresponding section)
#   3. Run: bash quick_batch_start.sh
#
################################################################################

# ============================================================================
# CONFIGURATION - EDIT THESE PATHS
# ============================================================================

# Dataset paths (MODIFY THESE!)
SCANNET_ROOT="/path/to/scannet"       # 改成你的 ScanNet 路径
MEGADEPTH_ROOT="/path/to/megadepth"   # 改成你的 MegaDepth 路径

# Output path
OUTPUT_ROOT="./data/vcomatcher_phase1" # 改成你的输出路径

# Log directory
LOG_DIR="./logs/batch_processing"

# ============================================================================
# TEMPLATE 1: Process Both ScanNet and MegaDepth (Recommended)
# ============================================================================

# Uncomment to use:
python batch_process_datasets.py \
    --scannet_root "${SCANNET_ROOT}" \
    --megadepth_root "${MEGADEPTH_ROOT}" \
    --output_root "${OUTPUT_ROOT}" \
    --log_dir "${LOG_DIR}" \
    --resume

# ============================================================================
# TEMPLATE 2: Process Only ScanNet (Indoor Scenes)
# ============================================================================

# Uncomment to use:
# python batch_process_datasets.py \
#     --scannet_root "${SCANNET_ROOT}" \
#     --output_root "${OUTPUT_ROOT}" \
#     --log_dir "${LOG_DIR}" \
#     --scannet_tau_min 0.1 \
#     --scannet_tau_max 10.0 \
#     --scannet_tau_uncertainty 15.0 \
#     --scannet_pnp_tau 6.0 \
#     --resume

# ============================================================================
# TEMPLATE 3: Process Only MegaDepth (Outdoor Scenes)
# ============================================================================

# Uncomment to use:
# python batch_process_datasets.py \
#     --megadepth_root "${MEGADEPTH_ROOT}" \
#     --output_root "${OUTPUT_ROOT}" \
#     --log_dir "${LOG_DIR}" \
#     --megadepth_tau_min 0.5 \
#     --megadepth_tau_max 100.0 \
#     --megadepth_tau_uncertainty 15.0 \
#     --megadepth_pnp_tau 6.0 \
#     --resume

# ============================================================================
# TEMPLATE 4: Debug Mode (Test on Small Subset) 调试模式
# ============================================================================

# For testing, create a small subset first:
#   mkdir -p /tmp/scannet_test
#   cp -r /path/to/scannet/scene0000_00 /tmp/scannet_test/
#   cp -r /path/to/scannet/scene0000_01 /tmp/scannet_test/

# Then uncomment:
# python batch_process_datasets.py \
#     --scannet_root "/tmp/scannet_test" \
#     --output_root "./data/phase1_test" \
#     --log_dir "./logs/test" \
#     --resume

# ============================================================================
# TEMPLATE 5: High Quality Mode (Strict Filtering) 高质量模式
# ============================================================================

# Use this for maximum quality, lower coverage:
# python batch_process_datasets.py \
#     --scannet_root "${SCANNET_ROOT}" \
#     --megadepth_root "${MEGADEPTH_ROOT}" \
#     --output_root "${OUTPUT_ROOT}" \
#     --log_dir "${LOG_DIR}" \
#     --scannet_tau_uncertainty 10.0 \
#     --scannet_pnp_tau 8.0 \
#     --megadepth_tau_uncertainty 10.0 \
#     --megadepth_pnp_tau 8.0 \
#     --resume

# ============================================================================
# TEMPLATE 6: Robust Mode (Relaxed Filtering for Difficult Scenes) 鲁棒模式
# ============================================================================

# Use this for challenging scenes with weak textures:
# python batch_process_datasets.py \
#     --scannet_root "${SCANNET_ROOT}" \
#     --megadepth_root "${MEGADEPTH_ROOT}" \
#     --output_root "${OUTPUT_ROOT}" \
#     --log_dir "${LOG_DIR}" \
#     --scannet_tau_uncertainty 20.0 \
#     --scannet_pnp_tau 5.0 \
#     --megadepth_tau_uncertainty 20.0 \
#     --megadepth_pnp_tau 5.0 \
#     --resume

# ============================================================================
# TEMPLATE 7: Background Processing (Recommended for Long Jobs) 后台处理
# ============================================================================

# Run in background with nohup:
# nohup python batch_process_datasets.py \
#     --scannet_root "${SCANNET_ROOT}" \
#     --megadepth_root "${MEGADEPTH_ROOT}" \
#     --output_root "${OUTPUT_ROOT}" \
#     --log_dir "${LOG_DIR}" \
#     --resume \
#     > batch_process.log 2>&1 &

# Then monitor progress:
#   tail -f batch_process.log
#   # or
#   tail -f logs/batch_processing/batch_processing_*.log

# ============================================================================
# DEFAULT: If nothing is uncommented, show usage
# ============================================================================

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  VCoMatcher Phase 1: Quick Batch Processing Launcher          ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "⚠️  No template selected!"
echo ""
echo "Please edit this script and:"
echo "  1. Update the dataset paths in the CONFIGURATION section"
echo "  2. Uncomment one of the templates (1-7)"
echo "  3. Run: bash quick_batch_start.sh"
echo ""
echo "Available templates:"
echo "  1. Process Both ScanNet and MegaDepth (Recommended)"
echo "  2. Process Only ScanNet (Indoor)"
echo "  3. Process Only MegaDepth (Outdoor)"
echo "  4. Debug Mode (Small Subset)"
echo "  5. High Quality Mode (Strict Filtering)"
echo "  6. Robust Mode (Relaxed Filtering)"
echo "  7. Background Processing (Long Jobs)"
echo ""
echo "For detailed documentation, see: BATCH_PROCESSING_GUIDE.md"

