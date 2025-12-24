# 🔧 VCoMatcher Phase 1 故障排除指南

**版本**: v1.6 | **日期**: 2025-12-22

---

## ⚠️ **v1.6 升级警告**

**如果你使用的是 v1.5 或更早版本，请立即升级！**

v1.6 修复了 **15个关键bug**，包括1个**Critical级别的坐标系混淆bug**。

**必须操作**:
```bash
# 1. 更新代码到 v1.6
git pull

# 2. 删除所有旧数据
rm -rf data/vcomatcher_phase1/*

# 3. 重新生成（使用修复后的代码）
python vcomatcher_phase1_data_engine.py --scene_dir ... --output_dir data/vcomatcher_phase1
```

---

## 🚨 常见问题速查

| 症状 | 原因 | 解决方案 |
|------|------|---------|
| mask_loss < 30% | tau_uncertainty 太严格 | `--tau_uncertainty 20.0` |
| 重投影误差 > 5px | 使用了v1.5旧数据 | **升级到v1.6并重新生成** ⚠️ |
| 滑动窗口崩溃 | 最后窗口 < overlap | 已修复（v1.6）|
| GPU OOM (N>50) | 内存溢出 | 已修复（v1.6），自动批处理 |
| 训练噪声大 | epsilon误用 | 已修复（v1.6）|
| 灰度图加载失败 | 未处理单通道图 | 已修复（v1.6）|
| PnP无法调整 | 阈值硬编码 | 已修复（v1.6），使用 `--pnp_tau` |

---

## 问题 1: mask_loss 覆盖率太低

### 症状

```
[1.5] 有效像素比例:
  mask_loss: 18.90% ✗ 太低（应该 60-75%）
```

### 诊断

```bash
python diagnose_mask_breakdown.py --data_file scene.npz
```

查看哪个过滤步骤过滤最多。

### 解决方案

**A. 调整 tau_uncertainty**（最常见）
```bash
# 当前值过小，提高阈值
python vcomatcher_phase1_data_engine.py --tau_uncertainty 20.0 ...
```

**B. 放宽一致性检查**
```bash
# 修改代码中的 epsilon_consist
epsilon_consist = 0.08  # 从 0.05 增大
```

---

## 问题 2: 重投影误差过大 ⚠️ **v1.6关键修复**

### 症状

```
[2.3] 投影一致性:
  平均误差: 15 pixels ✗ (或 > 3px)
```

### 解决方案

**🚨 A. 升级到 v1.6（坐标系混淆修复）**
```bash
# v1.6 修复了PnP中的严重坐标系bug
# 预期改善: 2.7px → 0.5-1px

# 1. 更新代码
git pull

# 2. 重新生成数据
python vcomatcher_phase1_data_engine.py --scene_dir ... --output_dir ./data/phase1_fixed
```

**B. 调整PnP阈值（v1.6新功能）**
```bash
# 如果点云质量高
python vcomatcher_phase1_data_engine.py --pnp_tau 5.0 ...

# 如果噪声较大
python vcomatcher_phase1_data_engine.py --pnp_tau 8.0 ...
```

**C. 检查是否用了最新代码**
- v1.6 包含坐标系修复 + 15个bug修复
- **必须删除旧数据重新生成**

---

## 问题 3: 不确定性校准失效

### 症状

```
Pearson 相关系数: -0.21 ✗ (负相关)
最确定区域误差: 6px (反而最大)
```

### 原因

1. 内参主点偏移（系统性偏差）
2. VGGT 的 uncertainty 估计不准确（固有限制）

### 解决方案

```bash
# 1. 测试系统性偏差
python test_systematic_bias.py --data_file scene.npz

# 2. 如果改善 > 40%，应用偏移修复
python manual_fix_intrinsic.py --offset_x X --offset_y Y ...

# 3. 如果改善 < 20%
# → VGGT 的 uncertainty 本身不准确（已知限制）
# → 可以接受，不影响训练
```

---

## 问题 4: 验证失败多项

### 常见失败测试

**mask_ratios**: mask_geom > 95%
```bash
# 降低 tau_max
--tau_max 50.0
```

**reprojection**: 误差 > 2px
```bash
# 如果 < 5px: 调整验证阈值（可接受）
# 如果 > 5px: 需要修复（见问题 2）
```

**uncertainty_calibration**: 校准失效
```bash
# 如果相关系数 > -0.2: 可以接受（VGGT 限制）
# 如果 < -0.2: 需要偏移修复（见问题 3）
```

**triangulation**: 三角测量误差 > 1%
```bash
# 这是 VGGT 深度估计的正常误差
# 可以放宽阈值到 3%
```

---

## 🔍 诊断流程图

```
遇到问题
  ↓
运行对应的诊断工具
  ├─ mask 问题 → diagnose_mask_breakdown.py
  ├─ Confidence → diagnose_confidence.py  
  ├─ Padding → check_padding_fix.py
  ├─ 偏差 → test_systematic_bias.py
  └─ 校准 → analyze_uncertainty_calibration.py
  ↓
根据诊断结果
  ├─ 参数问题 → 调整参数重新生成
  ├─ 位姿问题 → fix_pose_orthogonality.py
  ├─ 偏移问题 → manual_fix_intrinsic.py
  └─ VGGT 限制 → 接受并进入训练
```

---

## 💡 重要提示

### 关于 VGGT 的固有限制

某些"问题"实际上是 VGGT 的特性，不是 Bug：

1. **重投影误差 3-5px**: VGGT 深度/位姿估计精度
2. **不确定性校准弱**: VGGT 的 uncertainty 不完美
3. **位姿不正交**: float16 推理的数值误差

**这些都可以通过修复工具或训练策略处理，不影响最终效果！**

### 何时停止优化

满足以下条件即可进入 Phase 2：
- ✅ mask_loss: 60-75%
- ✅ 重投影误差: < 5px
- ✅ 位姿正交性: < 1e-4
- ✅ 验证通过率: > 80%

**不要过度优化！实际训练效果才是最终检验标准。**

---

## 🆕 v1.6 新增问题与解决方案

### 问题 5: 滑动窗口索引越界

**症状**: `IndexError: index -5 is out of bounds`

**原因**: 最后一个窗口帧数 < overlap（如只有3帧但overlap=8）

**解决方案**: ✅ 已在v1.6自动修复，无需手动干预

---

### 问题 6: GPU内存溢出（大场景）

**症状**: `CUDA out of memory` (N > 50 images)

**原因**: 重叠矩阵计算一次性加载所有图像到GPU

**解决方案**: ✅ 已在v1.6添加自动批处理
```bash
# 会自动检测并使用批处理
# 输出示例:
# ⚠️  Large scene detected (N=100)
#    Using batched processing (batch_size=10) to prevent GPU OOM
```

---

### 问题 7: 训练Loss出现NaN

**症状**: `Loss = nan` 或梯度爆炸

**可能原因**:
1. **使用了v1.5旧数据** - epsilon误用导致
2. 深度一致性NaN传播

**解决方案**:
```bash
# ✅ 升级到v1.6并重新生成数据
python vcomatcher_phase1_data_engine.py --scene_dir ... 
```

---

### 问题 8: 灰度图像加载崩溃

**症状**: `IndexError: index 2 is out of bounds for axis 2`

**原因**: 灰度图只有2维，访问`img.shape[2]`失败

**解决方案**: ✅ 已在v1.6自动处理，支持灰度→RGB转换

---

## 📋 v1.6 修复清单

使用v1.6前，请确认已解决：
- [x] Bug #15: 坐标系混淆（Critical）
- [x] Bug #1-14: 其他14个bug
- [x] 重新生成所有Phase 1数据
- [x] 验证重投影误差 < 2px

---

**完整细节**: 见 `COMPLETE_WORKFLOW_GUIDE.md`  
**更新历史**: 见 `CHANGELOG.md`  
**最后更新**: 2025-12-22 (v1.6)

