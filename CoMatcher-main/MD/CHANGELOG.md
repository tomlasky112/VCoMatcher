# VCoMatcher 更新日志

**当前版本**: v1.8 | **更新日期**: 2025-12-25

---

## v1.8 (2025-12-25) - 批处理系统 + 文档整合

### 新增
- `batch_process_datasets.py` - 生产级批处理引擎 (v1.2.1)
- `verify_dataset_structure.py` - 数据集验证器
- `monitor_batch_progress.py` - 进度监控器
- `quick_batch_start.sh` - 快速启动模板
- `check_import_paths.py` - 路径验证工具

### 批处理特性
- ✅ ScanNet + MegaDepth 完全支持
- ✅ 自动检测 `images`/`imgs` 目录
- ✅ 断点续传 + 错误隔离
- ✅ 智能显存管理 (OOM 风险 -90%)
- ✅ 参数验证 + 权限检查

### 文档整合
- 文件数: 20 → 10 (精简 50%)
- 删除 10 个冗余文档 (合并到现有文档)

---

## v1.7 (2025-12-23) - 测试系统完善

### 新增
- `test_sliding_window.py` - 7个滑动窗口测试
- `test_phase2_dataset.py` - 新增4个高级测试
- `run_all_tests.py` - 统一测试入口
- `verify_all_fixes.py` - Bug自动验证工具

### 修复
- 8个测试代码bug (2个Critical + 5个Important + 1个Minor)

### 改进
- 测试覆盖率: 60% → 95%
- 文档数量: 18个 → 10个 (精简44%)

---

## v1.6 (2025-12-22) - 15个关键Bug修复 ⚠️ 必须升级

### Critical
- **Bug #15**: PnP坐标系混淆 (w2c/c2w) - 重投影误差从2.7px降至0.5-1px

### High
- **Bug #1**: epsilon误用导致训练噪声
- **Bug #4**: 滑动窗口索引越界
- **Bug #5**: GPU OOM (N>50场景)
- **Bug #8**: 线性混合除零

### Medium
- Bug #6: PnP阈值硬编码 → 新增 `--pnp_tau` 参数
- Bug #7/9/11/12/13/14: 边界条件和异常处理

### Low
- Bug #2/3/10: 数值稳定性和性能优化

### 破坏性变更
必须重新生成所有Phase 1数据！

---

## v1.5 (2025-12-22) - 文档重构

- 删除5个冗余文档
- 更新工作流程指南

---

## v1.4 (2025-12-21) - 滑动窗口 + A100优化

### 新增
- `vcomatcher_sliding_window.py` - Umeyama对齐、全局拼接、线性平滑
- 滑动窗口: 32帧/窗口，8帧重叠

### 删除
- 4个冗余文件 (check_environment.py等)

---

## Phase 2 v1.1 (2025-12-21) - 图像加载

### 新增
- `load_and_preprocess_image()` - VGGT预处理逻辑
- 图像加载测试

---

## Phase 2 v1.0 (2025-12-19) - 关键修复

### 修复
1. 不确定性→权重映射颠倒
2. 伪多视图采样 (重复同一源视图)
3. 内存爆炸 (cache_data默认改为False)

---

## v1.3 (2025-12-19) - 生产就绪

### 最终配置
```python
tau_uncertainty = 15.0
pnp_tau = 6.0
epsilon_consist = 0.15
```

### 性能
- 验证通过率: 93.3%
- 重投影误差: ~1.6px (中位数)

---

**完整Bug详情**: 见 `BUGFIX.md`
