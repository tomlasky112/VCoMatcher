# 🚀 VCoMatcher 批处理系统使用指南

**版本**: v1.2.1 (Bug 修复版)  
**日期**: 2025-12-25

---

## ⚡ 30 秒快速开始

```bash
# 1. 验证数据集
python verify_dataset_structure.py \
    --dataset_root /data/scannet \
    --dataset_name scannet

# 2. 启动批处理
python batch_process_datasets.py \
    --scannet_root /data/scannet \
    --megadepth_root /data/megadepth \
    --output_root ./data/vcomatcher_phase1 \
    --resume

# 3. 监控进度（另一个终端）
python monitor_batch_progress.py
```

---

## 📦 核心工具

| 文件 | 功能 | 版本 |
|------|------|------|
| `batch_process_datasets.py` | 批处理引擎 | v1.2 |
| `verify_dataset_structure.py` | 数据集验证器 | v1.0 |
| `monitor_batch_progress.py` | 进度监控器 | v1.0 |
| `quick_batch_start.sh` | 快速启动脚本 | v1.0 |

---

## 🆕 v1.2.1 更新亮点

### ✅ v1.2.1 新增修复 (2025-12-25)

1. **发现与处理一致性**
   - 修复 process_scene 图像目录检查
   - 完全兼容 MegaDepth imgs 目录

2. **参数验证**
   - 早期检测无效参数
   - 清晰的错误消息

3. **权限检查**
   - 提前检测输出目录权限
   - 避免晚期失败

4. **错误处理改进**
   - GPU 错误更好的诊断
   - JSON 加载错误处理

详见: `MD/BUGFIX_v1.2.1.md` ⭐

### ✅ v1.2 核心修复

1. **MegaDepth 支持** - 自动支持 `images` 和 `imgs`
2. **.npz 懒加载** - 修复 mmap 不兼容
3. **路径一致性** - 修复 sys.path 层级
4. **清爽输出** - 日志分离

详见: `MD/BUGFIX_v1.2.md`

---

## 🎯 核心特性

- ✅ **自动化**: 自动发现场景、自动处理 OOM
- ✅ **健壮性**: 断点续传、错误隔离
- ✅ **监控**: 实时进度、估算完成时间
- ✅ **兼容性**: 支持 ScanNet 和 MegaDepth

---

## 📊 性能指标（A100 80GB）

| 场景规模 | 图像数 | 处理时间 |
|---------|-------|---------|
| 小 | 10-30 | 30-60秒 |
| 中 | 30-60 | 1-3分钟 |
| 大 | 60-100 | 3-10分钟 |

**数据集处理时间**:
- ScanNet (1500 场景): ~50 小时
- MegaDepth (200 场景): ~10 小时
- **总计**: ~60 小时（2.5 天）

---

## 🔍 常用命令

### 验证环境

```bash
# 验证 VGGT 导入路径（推荐首次运行）
python check_import_paths.py

# 验证数据集结构
# ScanNet
python verify_dataset_structure.py \
    --dataset_root /data/scannet \
    --dataset_name scannet

# MegaDepth（自动支持 imgs 目录）
python verify_dataset_structure.py \
    --dataset_root /data/megadepth \
    --dataset_name megadepth
```

### 批处理

```bash
# 标准模式
python batch_process_datasets.py \
    --scannet_root /data/scannet \
    --megadepth_root /data/megadepth \
    --output_root ./data/vcomatcher_phase1 \
    --resume

# 后台运行（推荐）
nohup python batch_process_datasets.py \
    --scannet_root /data/scannet \
    --megadepth_root /data/megadepth \
    --output_root ./data/phase1 \
    --resume > batch.log 2>&1 &
```

### 监控

```bash
# 实时监控
python monitor_batch_progress.py

# 查看日志（v1.2 日志更清晰）
tail -f logs/batch_processing/batch_processing_*.log

# 查看 GPU 使用率
watch -n 1 nvidia-smi
```

---

## 📁 输出结构

```
output_root/
├── scannet/
│   ├── scene0000_00.npz
│   └── ...
├── megadepth/
│   ├── 0000.npz
│   └── ...
└── logs/
    └── batch_processing/
        ├── batch_processing_*.log  # 详细日志
        ├── checkpoint.json          # 断点续传
        └── report_*.json            # 最终报告
```

---

## 🐛 故障排除

### Q1: MegaDepth 找不到场景

**v1.2 已修复**: 自动支持 `imgs` 目录

```bash
# 直接使用即可
python batch_process_datasets.py \
    --megadepth_root /data/MegaDepth_v1 \
    --output_root ./data/phase1 \
    --resume
```

### Q2: ValueError: Cannot use mmap_mode

**v1.2 已修复**: 正确使用 .npz 懒加载

### Q3: 控制台输出混乱

**v1.2 已修复**: 日志分离

- **控制台**: 只显示进度条和关键信息
- **日志文件**: 包含所有详细信息

```bash
# 查看详细日志
tail -f logs/batch_processing/batch_processing_*.log
```

### Q4: 如何暂停/恢复

```bash
# 暂停: Ctrl+C
# 恢复: 使用 --resume
python batch_process_datasets.py ... --resume
```

---

## 🎯 参数速查

| 场景类型 | tau_uncertainty | pnp_tau | 说明 |
|---------|----------------|---------|------|
| **标准** | 15.0 | 6.0 | 推荐配置 |
| **困难** | 20.0 | 5.0 | 弱纹理/大视角 |
| **高质量** | 10.0 | 8.0 | 严格过滤 |

**数据集特定**:
- ScanNet: `tau_min=0.1`, `tau_max=10.0`（室内）
- MegaDepth: `tau_min=0.5`, `tau_max=100.0`（室外）

---

## ✅ 典型工作流

```bash
# Step 0: 验证导入路径（首次运行推荐）
python check_import_paths.py

# Step 1: 验证数据集（必须）
python verify_dataset_structure.py --dataset_root <PATH> --dataset_name <NAME>

# Step 2: 启动批处理（在 tmux 中）
tmux new -s vcomatcher
python batch_process_datasets.py \
    --scannet_root /data/scannet \
    --megadepth_root /data/megadepth \
    --output_root ./data/vcomatcher_phase1 \
    --resume

# Step 3: 监控（在另一个终端）
python monitor_batch_progress.py

# Step 4: 完成后查看报告
cat logs/batch_processing/report_*.json | python -m json.tool
```

---

## 💡 最佳实践

1. ✅ 始终使用 `--resume`
2. ✅ 使用 `tmux` 或 `screen`
3. ✅ 监控 GPU 使用情况
4. ✅ 定期检查日志文件
5. ✅ 处理前先验证数据集

---

## 🔄 版本历史

### v1.2.1 (2025-12-25) - Bug 修复版 ⭐

✅ **修复 6 个 Bug**:
- 图像目录检查一致性
- glob 模式优化
- 参数验证
- 权限检查
- GPU 错误处理改进
- verify_dataset_structure 一致性

**推荐所有用户升级**

### v1.2 (2025-12-25) - 集成修复版

✅ **修复**:
- MegaDepth imgs 目录支持
- .npz 懒加载优化
- sys.path 一致性
- 日志与进度条分离

### v1.1 (2025-12-25) - 性能优化版

✅ **修复**:
- 深层目录结构
- I/O 优化
- 智能显存管理

### v1.0 (2025-12-25) - 初始版

---

## 📚 完整文档

| 文档 | 内容 |
|------|------|
| `BATCH_PROCESSING.md` | 本文档（使用指南） |
| `BUGFIX_v1.2.md` | v1.2 修复详解 |
| `CHANGELOG.md` | 完整版本历史 |

---

**准备好开始了吗？** 🚀  
从验证数据集开始，30 分钟内即可启动批处理！

---

**版权所有 © 2025 VCoMatcher Team**
