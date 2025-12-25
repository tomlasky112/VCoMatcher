# 🚀 VCoMatcher 完整工作流程指南

**版本**: v2.5 | **日期**: 2025-12-25  
**状态**: Phase 1 v1.8 (生产级) | Phase 2 v1.1 (就绪) | 测试 v1.8 (完善) | Phase 3 准备中

---

## 🏗️ 1. 环境准备

```bash
cd CoMatcher-main
pip install -r requirements_vcomatcher.txt
```

---

## 🛠️ 2. Phase 1: 数据生成

### 2.1 单场景处理
```bash
python vcomatcher_phase1_data_engine.py \
    --scene_dir ../../vggt-main/examples/kitchen \
    --output_dir ./data/vcomatcher_phase1 \
    --tau_uncertainty 15.0 \
    --pnp_tau 6.0
```

### 2.2 批量处理 (推荐)
```bash
# Step 1: 验证数据集
python verify_dataset_structure.py --dataset_root /data/scannet --dataset_name scannet

# Step 2: 批处理
python batch_process_datasets.py \
    --scannet_root /data/scannet \
    --megadepth_root /data/megadepth \
    --output_root ./data/vcomatcher_phase1 \
    --resume

# Step 3: 监控进度
python monitor_batch_progress.py
```

> 详见 `BATCH_PROCESSING.md`

### 验证生成结果
```bash
python validate_phase1_comprehensive.py --data_file ./data/vcomatcher_phase1/xxx.npz
python run_all_tests.py --critical-only
```

**v1.6 新增参数**:
- `--pnp_tau`: PnP优化阈值（默认6.0，范围5-8）
- 自动批处理: 大场景(N>50)自动启用，防止GPU OOM

### 验证生成结果
```bash
# 数据质量验证
python validate_phase1_comprehensive.py --data_file data/vcomatcher_phase1_test/kitchen.npz

# v1.7新增: 滑动窗口测试
python test_sliding_window.py
```

**验收标准 (v1.6-v1.7更新)**:
- ✅ 验证通过率 > 90%
- ✅ 重投影误差 < **1.5px** (中位数) - v1.6坐标系修复后
- ✅ 重投影误差 < **2.5px** (平均值)
- ✅ mask_loss 覆盖率 60-75% (v1.6优化后)
- ✅ 位姿正交性误差 < 1e-4
- ✅ 滑动窗口测试: 7/7通过 (v1.7新增)

### 🔬 技术细节 (Technical Details)

#### PnP 优化策略
VGGT 输出的位姿存在约 3px 的系统性误差，我们通过 PnP (SOLVEPNP_ITERATIVE) 结合 2D-3D 约束进行微调。
- **解耦阈值**: 使用宽松的 `tau=15.0` 进行 Training Mask 生成，但仅使用严格的 `pnp_tau=6.0` 的点进行位姿解算。
- **中心校正**: 修复了 VGGT 的 0.5px 像素中心偏移。

#### 滑动窗口机制 (Sliding Window)
针对 >32 帧的长序列，采用滑动窗口处理：
- **窗口大小**: 32 帧 (重叠 8 帧)
- **Sim3 对齐**: 在重叠区域计算 Sim3 变换，将所有局部窗口对齐到全局坐标系，并使用线性插值平滑接缝。

#### 双重掩膜系统
- **$M_{geom}$ (Loose)**: 仅过滤无效深度。用于 Phase 2 建图 (Graph Construction)。
- **$\mathbb{I}_{valid}$ (Strict)**: 过滤高不确定性区域。用于 Phase 3 Loss 计算。

---

## 🔄 3. Phase 2: 数据加载 (Dataset & Loader)

v1.1 版本实现了完整的图像加载与几何对齐验证，确保像素数据与 Phase 1 生成的 3D 点完美匹配。

### 核心功能
- **目标中心化变换**: 实时将所有位姿和点云转换到 Target 相机坐标系 (Target Pose = Identity)。
- **混合采样 (Curriculum)**: 动态混合 COLMAP (Easy) 和 VGGT (Hard) 样本。
- **源感知权重**: 根据不确定性动态计算 Loss 权重 `W_src`。

### 验证命令 (v1.7扩展)
```bash
# Phase 2测试 (v1.7新增11个测试)
python test_phase2_dataset.py

# 或使用集成验证
python run_phase2_validation.py --full

# 或运行所有测试 (推荐)
python run_all_tests.py --critical-only
```

### 可视化命令
```bash
python quick_visualize.py --data_file data\vcomatcher_phase1_test\kitchen.npz    
```

**验收标准 (v1.7)**:
- ✅ Phase 2测试: 11/11通过
- ✅ 目标中心化: Target→Identity
- ✅ 几何一致性: 深度-点云<5%误差
- ✅ 多视图采样: 多样性>0.7

## 🧠 4. Phase 3: 模型训练 (Training)

目前处于准备阶段 (50% 完成)。

### 待办事项 (P0)
1. **Loss 集成**: 在 `par_comatcher.py` 中应用 `W_src` 权重。
2. **训练脚本**: 编写 `vcomatcher_train.py`。

### 预计训练流程
```bash
# (待实现)
python vcomatcher_train.py \
    --colmap_data ./data/colmap \
    --vggt_data ./data/phase1 \
    --batch_size 4
```

---

## 🔧 常见问题与维护

- **文件路径**: Phase 2 加载器依赖 Phase 1 记录的绝对路径。
- **参数调整**: 详见 `PARAMETER_GUIDE.md`。
- **问题诊断**: 详见 `TROUBLESHOOTING.md`。
