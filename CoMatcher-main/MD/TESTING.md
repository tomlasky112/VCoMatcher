# VCoMatcher 测试系统

**版本**: v1.7 (包含新增的11个测试)  
**更新**: 2025-12-23  
**状态**: ✅ 测试覆盖率95%，所有bug已修复

---

## ⚡ 快速开始

```bash
# 关键测试 (2-3分钟，推荐！)
python run_all_tests.py --critical-only

# 完整测试 (5-10分钟)
python run_all_tests.py

# 单独测试
python test_sliding_window.py      # 滑动窗口OOM解决方案
python test_phase2_dataset.py      # DataLoader测试
```

---

## 📊 测试覆盖概览

| 模块 | 测试数量 | 覆盖率 | 状态 |
|------|---------|--------|------|
| **滑动窗口** | 7个 | 100% | ✅ 完成 |
| **Phase 2 DataLoader** | 11个 | 95% | ✅ 完成 |
| **Phase 1 数据引擎** | 15个 | 85% | ✅ 完成 |

**总计**: 33个单元测试

---

## 🔍 关键测试说明

### 🔴 Critical Tests (必须通过)

#### 1. 位姿-点云同步测试
```bash
python -c "from test_sliding_window import test_pose_points_synchronization; test_pose_points_synchronization()"
```
**验证**: 防止几何数据损坏 (投影误差<1e-4px)

#### 2. 目标中心化变换
```bash
python -c "from test_phase2_dataset import test_target_centric_transformation; test_target_centric_transformation()"
```
**验证**: Target位姿→Identity，坐标系正确

#### 3. 几何一致性
```bash
python -c "from test_phase2_dataset import test_geometric_consistency; test_geometric_consistency()"
```
**验证**: 深度-点云一致性<5%

#### 4. Umeyama对齐
```bash
python -c "from test_sliding_window import test_umeyama_alignment_known_transform; test_umeyama_alignment_known_transform()"
```
**验证**: Sim3对齐算法正确性

---

## 🧪 测试文件说明

### test_sliding_window.py (新增)
**目的**: 验证滑动窗口机制处理100+张图像大场景

**测试列表**:
1. Umeyama已知变换恢复
2. Umeyama边界情况 (N=3, 退化)
3. 位姿-点云同步 ⭐⭐⭐
4. 线性平滑
5. 窗口创建
6. 端到端处理
7. 内存效率验证

### test_phase2_dataset.py (扩展)
**目的**: 验证DataLoader和坐标变换

**新增测试** (v1.7):
8. 多视图采样策略
9. 几何一致性 ⭐⭐⭐
10. 掩膜边界条件
11. 课程学习调度

---

## 🐛 测试Bug修复 (v1.7)

在测试代码本身中发现并修复了8个bug:

| Bug | 文件 | 严重性 | 状态 |
|-----|------|--------|------|
| 未使用导入 | test_sliding_window.py | Minor | ✅ |
| 数据重复加载 | test_phase2_dataset.py | Important | ✅ |
| 假阳性风险 | test_phase2_dataset.py | Critical | ✅ |
| 调度期望错误 | test_phase2_dataset.py | Critical | ✅ |
| 退出码判断 | run_phase2_validation.py | Important | ✅ |
| 像素对齐 | test_phase2_dataset.py | Important | ✅ |
| 空集处理 | test_phase2_dataset.py | Important | ✅ |
| 异常处理 | run_all_tests.py | Important | ✅ |

**验证**: `python verify_all_fixes.py` (8/8通过)

---

## 📋 验收标准

### 最低要求
- ✅ Critical tests: 4/4 通过
- ✅ 无运行时错误

### 生产标准
- ✅ 所有tests: 33/33 通过
- ✅ 性能: >15 samples/sec
- ✅ 重投影误差: <2.5px

---

## 🚀 运行脚本说明

### run_all_tests.py
统一测试入口，支持:
```bash
--critical-only    # 只运行4个关键测试
--phase1          # 只运行Phase 1测试
--phase2          # 只运行Phase 2测试
--quick           # 快速模式
```

### run_phase2_validation.py
Phase 2专用验证，支持:
```bash
--full            # 完整验证
--quick           # 快速验证
--sliding-window  # 只测试滑动窗口
--benchmark       # 性能测试
--visualize       # 可视化生成
```

### verify_all_fixes.py
自动验证所有bug修复，无参数。

---

## 🐛 常见问题

### Q1: 测试找不到数据
```
⚠️ Warning: data directory not found
```
**解决**: 先运行Phase 1生成数据
```bash
python vcomatcher_phase1_data_engine.py --scene_dir examples/kitchen
```

### Q2: torch导入警告
```
无法解析导入 "torch"
```
**说明**: IDE警告，不影响运行。确保已安装: `pip install torch`

### Q3: 某个测试失败
**步骤**:
1. 运行该测试查看详情
2. 检查Phase 1数据质量
3. 参考 `TROUBLESHOOTING.md`

---

## 📞 更多信息

- **Bug详情**: 参考 `BUGFIX.md` (23个bug)
- **故障排除**: `TROUBLESHOOTING.md`

---

**最简命令**: `python run_all_tests.py --critical-only` ⚡

