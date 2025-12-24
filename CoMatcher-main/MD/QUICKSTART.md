# ⚡ VCoMatcher 快速入门

**版本**: v1.7 | **更新日期**: 2025-12-23

---

## 🚀 30秒快速开始

```bash
# 1. 生成数据
python vcomatcher_phase1_data_engine.py \
    --scene_dir ../../vggt-main/examples/kitchen \
    --output_dir ./data/phase1

# 2. 验证质量
python validate_phase1_comprehensive.py --data_file ./data/phase1/kitchen.npz

# 3. 运行测试
python run_all_tests.py --critical-only
```

**期待结果**: 验证通过率 >95%, 重投影误差 <1.5px, 测试 4/4 通过

---

## 🔄 从旧版本升级

如果使用 v1.5 或更早版本，**必须升级**：

```bash
# 1. 删除旧数据
rm -rf data/vcomatcher_phase1/*

# 2. 重新生成
python vcomatcher_phase1_data_engine.py \
    --scene_dir ... --output_dir ./data/vcomatcher_phase1 \
    --tau_uncertainty 15.0 --pnp_tau 6.0

# 3. 验证
python validate_phase1_comprehensive.py --data_file ./data/vcomatcher_phase1/xxx.npz
```

---

## 🎯 推荐配置

### 标准场景
```bash
python vcomatcher_phase1_data_engine.py \
    --scene_dir <YOUR_SCENE> \
    --output_dir ./data/phase1 \
    --tau_uncertainty 15.0 \
    --pnp_tau 6.0
```

### 困难场景（弱纹理/大视角）
```bash
--tau_uncertainty 20.0 --pnp_tau 8.0
```

---

## 🔍 质量检查

```bash
python validate_phase1_comprehensive.py --data_file <YOUR_DATA.npz>
```

**通过标准**:
- ✅ 重投影误差(中位数) < 1.5px
- ✅ mask_loss 覆盖率: 60-75%
- ✅ 验证通过率 > 90%

---

## 📚 文档索引

| 需求 | 文档 |
|------|------|
| 测试系统 | `TESTING.md` |
| Bug清单 | `BUGFIX.md` |
| 完整流程 | `COMPLETE_WORKFLOW_GUIDE.md` |
| 参数调优 | `PARAMETER_GUIDE.md` |
| 故障排除 | `TROUBLESHOOTING.md` |

---

## 🎯 下一步

1. ✅ Phase 1 数据生成
2. ✅ 测试验证: `python run_all_tests.py --critical-only`
3. ⏭️ Phase 2 验证: `python run_phase2_validation.py --full`
4. ⏭️ Phase 3 训练: 见 `VCOMATCHER_PROJECT_STATUS.md`
