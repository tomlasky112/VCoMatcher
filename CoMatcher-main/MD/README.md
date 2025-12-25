# VCoMatcher 项目文档

**版本**: v1.8 | **更新日期**: 2025-12-25  
**状态**: Phase 1 & 2 完成 ✅ | 批处理就绪 ✅ | Phase 3 准备中 🔄

---

## ⚡ 快速开始

```bash
# 单场景处理
python vcomatcher_phase1_data_engine.py --scene_dir ... --output_dir ./data/phase1

# 批量处理 (推荐)
python batch_process_datasets.py \
    --scannet_root /data/scannet \
    --megadepth_root /data/megadepth \
    --output_root ./data/vcomatcher_phase1 \
    --resume
```

---

## 📚 文档索引

| 文档 | 用途 |
|------|------|
| **[QUICKSTART.md](QUICKSTART.md)** | 快速入门 ⭐ |
| **[BATCH_PROCESSING.md](BATCH_PROCESSING.md)** | 批处理指南 ⭐ |
| [TESTING.md](TESTING.md) | 测试系统 |
| [BUGFIX_v1.8.md](BUGFIX_v1.8.md) | Bug修复记录 |
| [COMPLETE_WORKFLOW_GUIDE.md](COMPLETE_WORKFLOW_GUIDE.md) | 完整流程 |
| [PARAMETER_GUIDE.md](PARAMETER_GUIDE.md) | 参数调优 |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | 故障排除 |
| [VCOMATCHER_PROJECT_STATUS.md](VCOMATCHER_PROJECT_STATUS.md) | 项目状态 |
| [CHANGELOG.md](CHANGELOG.md) | 更新历史 |

---

## 📊 项目状态

| Phase | 状态 | 版本 |
|-------|------|------|
| Phase 1 数据引擎 | ✅ 生产就绪 | v1.8 |
| 批处理系统 | ✅ 生产就绪 | v1.2.1 |
| Phase 2 数据集 | ✅ 生产就绪 | v1.1 |
| Phase 3 训练 | 🔄 进行中 | - |

---

**文档总数**: 10个 (精简 50%)  
**归档**: `archive/` 目录

