# VCoMatcher 项目文档

**版本**: v1.8 | **更新日期**: 2025-12-23  
**状态**: Phase 1 & 2 完成 ✅ | Phase 3 准备就绪 🔄

---

## ⚡ 快速开始

```bash
# 1. 环境配置
pip install -r requirements_vcomatcher.txt

# 2. Phase 1: 数据生成
python vcomatcher_phase1_data_engine.py --scene_dir ... --output_dir ./data/phase1

# 3. 测试验证
python run_all_tests.py --critical-only

# 4. Phase 2: 数据加载器验证
python run_phase2_validation.py --quick
```

---

## 📚 文档索引

| 文档 | 用途 | 重要性 |
|------|------|--------|
| **[QUICKSTART.md](QUICKSTART.md)** | 快速入门 | ⭐⭐⭐⭐⭐ |
| **[TESTING.md](TESTING.md)** | 测试系统 | ⭐⭐⭐⭐⭐ |
| **[BUGFIX_v1.8.md](BUGFIX_v1.8.md)** | v1.8关键Bug修复 | ⭐⭐⭐⭐ |
| [COMPLETE_WORKFLOW_GUIDE.md](COMPLETE_WORKFLOW_GUIDE.md) | 完整操作手册 | ⭐⭐⭐ |
| [PARAMETER_GUIDE.md](PARAMETER_GUIDE.md) | 参数调优指南 | ⭐⭐⭐ |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | 故障排除 | ⭐⭐⭐ |
| [VCOMATCHER_PROJECT_STATUS.md](VCOMATCHER_PROJECT_STATUS.md) | 项目状态+核心洞察 | ⭐⭐ |
| [CHANGELOG.md](CHANGELOG.md) | 更新历史 | ⭐ |

---

## 📊 项目状态

| Phase | 状态 | 版本 |
|-------|------|------|
| Phase 1 数据引擎 | ✅ 生产就绪 | v1.8 |
| Phase 2 数据集 | ✅ 生产就绪 | v1.1 |
| 测试系统 | ✅ 95%覆盖 | v1.8 |
| Phase 3 训练 | 🔄 进行中 | - |

---

**文档总数**: 8个 (已精简27%)  
**归档**: 3个历史文档移至 `archive/` 目录

