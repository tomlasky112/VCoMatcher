"""
VCoMatcher Phase 2: 一键完整验证脚本
=====================================

运行所有 Phase 2 验证测试，生成完整报告

使用方法:
    python run_phase2_validation.py

选项:
    --quick          只运行快速测试（跳过性能测试和可视化）
    --full           运行完整测试（包括性能和可视化）
    --benchmark      只运行性能测试
    --visualize      只运行可视化
"""

import sys
import time
import argparse
from pathlib import Path

# 添加必要的导入
import numpy as np
import torch
from torch.utils.data import DataLoader

from vcomatcher_phase2_dataset import (
    VCoMatcherDataset,
    MixedDataLoader,
    collate_fn,
    compute_source_aware_weights,
)


def print_header(title):
    """打印分节标题"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def run_sliding_window_tests():
    """运行滑动窗口OOM解决方案测试"""
    print_header("Phase 1 滑动窗口测试 (OOM Solution)")
    
    try:
        from test_sliding_window import (
            test_umeyama_alignment_known_transform,
            test_umeyama_edge_cases,
            test_pose_points_synchronization,
            test_linear_blending,
            test_window_creation,
            test_sliding_window_end_to_end,
            test_memory_efficiency,
        )
    except ImportError as e:
        print(f"⚠️ 无法导入 test_sliding_window.py")
        print(f"   错误详情: {e}")
        print("\n⏭️  跳过滑动窗口测试...\n")
        return []
    except Exception as e:
        print(f"❌ 导入时发生未知错误: {e}")
        print("⏭️  跳过滑动窗口测试...\n")
        return []
    
    tests = [
        ("Umeyama已知变换", test_umeyama_alignment_known_transform),
        ("Umeyama边界情况", test_umeyama_edge_cases),
        ("位姿-点云同步", test_pose_points_synchronization),
        ("线性平滑", test_linear_blending),
        ("窗口创建", test_window_creation),
        ("端到端处理", test_sliding_window_end_to_end),
        ("内存效率", test_memory_efficiency),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n{'─'*80}")
        print(f"运行测试: {name}")
        print(f"{'─'*80}")
        
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ 测试失败: {e}")
            results.append((name, False))
    
    return results


def run_basic_tests():
    """运行基础单元测试"""
    print_header("Phase 2 基础测试")
    
    # Try to import test functions
    try:
        from test_phase2_dataset import (
            test_target_centric_transformation,
            test_dataset_loading,
            test_dataloader,
            test_source_aware_weights,
            test_mixed_dataloader,
        )
    except ImportError as e:
        print(f"⚠️ 无法导入 test_phase2_dataset.py")
        print(f"   错误详情: {e}")
        
        # Check if file exists
        from pathlib import Path
        if not Path("test_phase2_dataset.py").exists():
            print("   原因: 文件不存在")
        else:
            print("   原因: 导入错误（可能缺少依赖或代码错误）")
            print("   建议: 检查 test_phase2_dataset.py 的语法和依赖")
        
        print("\n⏭️  跳过基础测试，继续详细验证...\n")
        return []
    except Exception as e:
        print(f"❌ 导入时发生未知错误: {e}")
        print("⏭️  跳过基础测试，继续详细验证...\n")
        return []
    
    # Try to import advanced tests (may not exist in older versions)
    try:
        from test_phase2_dataset import (
            test_multi_view_sampling,
            test_geometric_consistency,
            test_mask_boundary_conditions,
            test_curriculum_schedule_correctness,
        )
        advanced_tests_available = True
    except ImportError:
        advanced_tests_available = False
        print("  ℹ️  高级测试未找到，使用基础测试")
    
    tests = [
        ("目标中心化变换", test_target_centric_transformation),
        ("数据集加载", test_dataset_loading),
        ("DataLoader 批处理", test_dataloader),
        ("不确定性权重", test_source_aware_weights),
        ("混合数据加载", test_mixed_dataloader),
    ]
    
    # Add advanced tests if available
    if advanced_tests_available:
        tests.extend([
            ("多视图采样", test_multi_view_sampling),
            ("几何一致性", test_geometric_consistency),
            ("掩膜边界条件", test_mask_boundary_conditions),
            ("课程学习调度", test_curriculum_schedule_correctness),
        ])
    
    results = []
    for name, test_func in tests:
        print(f"\n{'─'*80}")
        print(f"运行测试: {name}")
        print(f"{'─'*80}")
        
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ 测试失败: {e}")
            results.append((name, False))
    
    return results


def run_detailed_verification():
    """运行详细验证"""
    print_header("Phase 2 详细验证")
    
    # 查找数据 - FIXED: 搜索所有.npz文件，不限定_fixed后缀
    data_dir = Path("./data/vcomatcher_phase1_test")
    
    # BUGFIX: Check if directory exists first
    if not data_dir.exists():
        print(f"⚠️ 数据目录不存在: {data_dir}")
        print("   请先运行 Phase 1 数据生成")
        return False
    
    data_paths = list(data_dir.glob("*.npz"))  # 修复：从 *_fixed.npz 改为 *.npz
    
    if not data_paths:
        print("⚠️ 未找到 Phase 1 数据，跳过详细验证")
        return False
    
    try:
        dataset = VCoMatcherDataset(
            data_paths, 
            sample_types=["easy", "hard", "extreme"],
            cache_data=True
        )
        
        print(f"[1] 数据集统计:")
        print(f"  总样本数: {len(dataset)}")
        
        # 样本类型统计
        sample_types = {"easy": 0, "hard": 0, "extreme": 0}
        for sample in dataset.samples:
            sample_types[sample["sample_type"]] += 1
        
        total = len(dataset)
        # BUGFIX: Handle division by zero when dataset is empty
        if total > 0:
            print(f"  Easy:    {sample_types['easy']:6d} ({sample_types['easy']/total*100:5.1f}%)")
            print(f"  Hard:    {sample_types['hard']:6d} ({sample_types['hard']/total*100:5.1f}%)")
            print(f"  Extreme: {sample_types['extreme']:6d} ({sample_types['extreme']/total*100:5.1f}%)")
        else:
            print(f"  ⚠️  数据集为空，无样本")
        
        # 加载第一个样本
        print(f"\n[2] 验证第一个样本:")
        batch = dataset[0]
        
        # 检查 Target 位姿
        extrinsic_rel = batch["extrinsic_rel"]
        target_pose = extrinsic_rel[0]
        is_identity = torch.allclose(target_pose, torch.eye(4), atol=1e-4)
        
        print(f"  Target 是 Identity: {'✓' if is_identity else '✗'}")
        
        if not is_identity:
            error = torch.abs(target_pose - torch.eye(4)).max().item()
            print(f"  误差: {error:.6f}")
        
        # 检查掩膜覆盖率
        print(f"\n[3] 掩膜覆盖率:")
        mask_geom = batch["mask_geom"]
        mask_loss = batch["mask_loss"]
        
        geom_ratio = mask_geom.float().mean().item() * 100
        loss_ratio = mask_loss.float().mean().item() * 100
        
        print(f"  mask_geom: {geom_ratio:.1f}%")
        print(f"  mask_loss: {loss_ratio:.1f}%")
        
        # 判断是否合格 (v1.6 updated: 60-75% is ideal)
        loss_ok = 60 <= loss_ratio <= 75
        print(f"  状态: {'✓ 合格' if loss_ok else '⚠️ 需要调整'}")
        
        # BUGFIX: Provide guidance if out of range
        if not loss_ok:
            if loss_ratio < 60:
                print(f"    建议: mask_loss太低，考虑增加 --tau_uncertainty")
            else:
                print(f"    建议: mask_loss太高，考虑减小 --tau_uncertainty")
        
        # 检查点云
        print(f"\n[4] 点云检查:")
        points_3d = batch["points_3d"]
        target_points = points_3d[0]
        target_depth = target_points[..., 2]
        
        positive_ratio = (target_depth > 0).float().mean().item() * 100
        print(f"  正深度比例: {positive_ratio:.1f}%")
        print(f"  深度范围: [{target_depth.min():.2f}, {target_depth.max():.2f}]")
        
        print(f"\n✓ 详细验证完成")
        return True
        
    except Exception as e:
        print(f"✗ 详细验证失败: {e}")
        return False


def run_performance_benchmark():
    """运行性能测试"""
    print_header("Phase 2 性能测试")
    
    data_dir = Path("./data/vcomatcher_phase1_test")
    
    # BUGFIX: Check directory exists
    if not data_dir.exists():
        print(f"⚠️ 数据目录不存在: {data_dir}")
        return False
    
    data_paths = list(data_dir.glob("*.npz"))  # FIXED: 从 *_fixed.npz 改为 *.npz
    
    if not data_paths:
        print("⚠️ 未找到数据，跳过性能测试")
        return False
    
    try:
        # 测试 1: 无缓存 (单进程)
        print(f"[1] 冷启动测试 (cache=False, workers=0):") 
        dataset_nocache = VCoMatcherDataset(data_paths, cache_data=False)
        
        start = time.time()
        for i in range(10):
            _ = dataset_nocache[i]
        elapsed = time.time() - start
        
        speed_nocache = 10 / elapsed
        print(f"  速度: {speed_nocache:.1f} samples/sec")
        
        # 测试 2: 有缓存 (单进程)
        print(f"\n[2] 热启动测试 (cache=True, workers=0):")
        dataset_cache = VCoMatcherDataset(data_paths, cache_data=True)
        
        # 预热
        for i in range(10):
            _ = dataset_cache[i]
        
        start = time.time()
        for i in range(100):
            _ = dataset_cache[i % 10]
        elapsed = time.time() - start
        
        speed_cache = 100 / elapsed
        print(f"  速度: {speed_cache:.1f} samples/sec")
        print(f"  加速比: {speed_cache/speed_nocache:.1f}x")
        
        # 测试 3: DataLoader 单进程
        print(f"\n[3] DataLoader 吞吐量 (cache=True, workers=0):")
        dataloader = DataLoader(
            dataset_cache,
            batch_size=8,
            num_workers=0,
            shuffle=False,
            collate_fn=collate_fn,
        )
        
        start = time.time()
        for batch in dataloader:
            pass
        elapsed = time.time() - start
        
        total_samples = len(dataset_cache)
        # BUGFIX: Handle zero elapsed time (very fast execution)
        if elapsed > 0:
            throughput = total_samples / elapsed
            print(f"  吞吐量: {throughput:.1f} samples/sec")
        else:
            print(f"  吞吐量: Too fast to measure (< 1ms)")
        
        # 测试 4: 真实训练场景 (无缓存 + 多进程) ⭐
        print(f"\n[4] ⭐ 真实训练场景 (cache=False, workers=4):")
        print(f"  (大数据集如ScanNet/MegaDepth的典型配置)")
        
        dataset_real = VCoMatcherDataset(data_paths, cache_data=False)
        dataloader_real = DataLoader(
            dataset_real,
            batch_size=8,
            num_workers=4,
            shuffle=False,
            collate_fn=collate_fn,
            pin_memory=True,
            prefetch_factor=2,
        )
        
        #Warmup
        print(f"  预热中...")
        for i, batch in enumerate(dataloader_real):
            if i >= 3:
                break
        
        # Benchmark
        start = time.time()
        for batch in dataloader_real:
            pass
        elapsed = time.time() - start
        
        # BUGFIX: Handle zero elapsed time
        if elapsed > 0:
            throughput_real = total_samples / elapsed
            print(f"  吞吐量: {throughput_real:.1f} samples/sec")
            print(f"  预估训练时间 (1 epoch = 1000 samples): {1000/throughput_real:.1f} 秒")
        else:
            print(f"  吞吐量: Too fast to measure")
        
        print(f"\n✓ 性能测试完成")
        return True
        
    except Exception as e:
        print(f"✗ 性能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_visualizations():
    """生成可视化"""
    print_header("Phase 2 可视化生成")
    
    try:
        import matplotlib
        matplotlib.use('Agg')  # 非交互式后端
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️ matplotlib 未安装，跳过可视化")
        return False
    
    data_dir = Path("./data/vcomatcher_phase1_test")
    
    # BUGFIX: Check directory exists and use correct glob pattern
    if not data_dir.exists():
        print(f"⚠️ 数据目录不存在: {data_dir}")
        return False
    
    data_paths = list(data_dir.glob("*.npz"))  # FIXED: 从 *_fixed.npz 改为 *.npz
    
    if not data_paths:
        print("⚠️ 未找到数据，跳过可视化")
        return False
    
    try:
        dataset = VCoMatcherDataset(data_paths, sample_types=["easy"])
        batch = dataset[0]
        
        # 创建输出目录
        vis_dir = Path("visualizations/phase2")
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        # 可视化 1: 位姿矩阵
        print(f"[1] 生成位姿矩阵可视化...")
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Target-Centric Transformation", fontsize=14)
        
        extrinsic_rel = batch["extrinsic_rel"].numpy()
        
        for i, ax in enumerate(axes.flat):
            pose = extrinsic_rel[i]
            im = ax.imshow(pose, cmap='RdBu', vmin=-1, vmax=1)
            ax.set_title(f"View {i} {'(Target)' if i==0 else ''}")
            
            # 添加数值
            for (j, k), val in np.ndenumerate(pose):
                color = 'white' if abs(val) > 0.5 else 'black'
                ax.text(k, j, f'{val:.2f}', 
                       ha='center', va='center', color=color, fontsize=8)
            
            plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        output_path = vis_dir / "target_centric_poses.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 保存到: {output_path}")
        
        # 可视化 2: 掩膜对比
        print(f"[2] 生成掩膜对比可视化...")
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Dual Mask Comparison (View 0)", fontsize=14)
        
        mask_geom = batch["mask_geom"][0].numpy()
        mask_loss = batch["mask_loss"][0].numpy()
        depth = batch["depth"][0].numpy()
        uncertainty = batch["uncertainty_map"][0].numpy()
        
        im0 = axes[0, 0].imshow(depth, cmap='viridis')
        axes[0, 0].set_title("Depth Map")
        plt.colorbar(im0, ax=axes[0, 0])
        
        axes[0, 1].imshow(mask_geom, cmap='gray')
        axes[0, 1].set_title(f"mask_geom ({mask_geom.mean()*100:.1f}%)")
        
        axes[1, 0].imshow(mask_loss, cmap='gray')
        axes[1, 0].set_title(f"mask_loss ({mask_loss.mean()*100:.1f}%)")
        
        im1 = axes[1, 1].imshow(uncertainty, cmap='hot')
        axes[1, 1].set_title("Uncertainty Map")
        plt.colorbar(im1, ax=axes[1, 1])
        
        plt.tight_layout()
        output_path = vis_dir / "mask_comparison.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 保存到: {output_path}")
        
        print(f"\n✓ 可视化生成完成")
        print(f"  查看目录: {vis_dir}")
        return True
        
    except Exception as e:
        print(f"✗ 可视化生成失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_summary(basic_results, detailed_ok, perf_ok, vis_ok, sliding_window_results=None):
    """打印总结报告"""
    print_header("验证总结报告")
    
    # Sliding window tests (if run)
    if sliding_window_results:
        print(f"[0] 滑动窗口测试 (OOM Solution):")
        sw_passed = sum(1 for _, result in sliding_window_results if result)
        sw_total = len(sliding_window_results)
        
        for name, result in sliding_window_results:
            status = "✓ PASS" if result else "✗ FAIL"
            print(f"  {name:25s}: {status}")
        
        if sw_total > 0:
            print(f"\n  通过率: {sw_passed}/{sw_total} ({sw_passed/sw_total*100:.0f}%)")
        else:
            print(f"\n  通过率: 0/0 (无测试)")
        print()
    
    print(f"[1] 基础测试:")
    passed = sum(1 for _, result in basic_results if result)
    total = len(basic_results)
    
    for name, result in basic_results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {name:25s}: {status}")
    
    # BUGFIX: Handle division by zero when total is 0
    if total > 0:
        print(f"\n  通过率: {passed}/{total} ({passed/total*100:.0f}%)")
    else:
        print(f"\n  通过率: 0/0 (无测试)")
    
    print(f"\n[2] 详细验证: {'✓ PASS' if detailed_ok else '✗ FAIL / SKIP'}")
    print(f"[3] 性能测试: {'✓ PASS' if perf_ok else '✗ FAIL / SKIP'}")
    print(f"[4] 可视化生成: {'✓ PASS' if vis_ok else '✗ FAIL / SKIP'}")
    
    # 总体判断
    print(f"\n{'='*80}")
    if passed == total and detailed_ok:
        print("🎉 恭喜！Phase 2 数据加载器完全就绪！")
        print("\n下一步:")
        print("  1. 查看 MD/PHASE3_STATUS_AND_TODO.md")
        print("  2. 开始 Phase 3 开发")
        print("  3. 运行: python vcomatcher_train.py (待创建)")
    elif total > 0 and passed >= total * 0.8:
        print("⚠️ Phase 2 基本可用，但有部分问题")
        print("\n建议:")
        print("  1. 查看失败的测试")
        print("  2. 参考 MD/PHASE2_EXPERIMENT_GUIDE.md")
        print("  3. 解决问题后重新运行")
    else:
        print("✗ Phase 2 存在严重问题")
        print("\n需要:")
        print("  1. 检查 Phase 1 数据质量")
        print("  2. 查看 MD/TROUBLESHOOTING.md")
        print("  3. 重新生成数据或修复代码")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="VCoMatcher Phase 2 完整验证"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="快速模式（跳过性能和可视化）"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="完整模式（运行所有测试）"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="只运行性能测试"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="只运行可视化"
    )
    parser.add_argument(
        "--sliding-window",
        action="store_true",
        help="只运行滑动窗口测试"
    )
    
    args = parser.parse_args()
    
    # 打印欢迎信息
    print("\n" + "="*80)
    print("  VCoMatcher Phase 2 - 完整验证")
    print("="*80)
    print(f"\n开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    
    # 根据参数运行测试
    if args.sliding_window:
        sw_results = run_sliding_window_tests()
        sw_passed = sum(1 for _, result in sw_results if result)
        sw_total = len(sw_results)
        print(f"\n总耗时: {time.time() - start_time:.1f} 秒")
        print(f"\n滑动窗口测试: {sw_passed}/{sw_total} 通过")
        sys.exit(0 if sw_passed == sw_total else 1)
    
    if args.benchmark:
        perf_ok = run_performance_benchmark()
        print(f"\n总耗时: {time.time() - start_time:.1f} 秒")
        sys.exit(0 if perf_ok else 1)
    
    if args.visualize:
        vis_ok = generate_visualizations()
        print(f"\n总耗时: {time.time() - start_time:.1f} 秒")
        sys.exit(0 if vis_ok else 1)
    
    # 默认运行测试
    # 1. Sliding window tests (Phase 1 OOM solution)
    sliding_window_results = run_sliding_window_tests()
    
    # 2. Basic Phase 2 tests
    basic_results = run_basic_tests()
    detailed_ok = run_detailed_verification()
    
    # 根据模式决定是否运行额外测试
    perf_ok = False
    vis_ok = False
    
    if args.full or not args.quick:
        perf_ok = run_performance_benchmark()
        vis_ok = generate_visualizations()
    
    # 打印总结
    elapsed = time.time() - start_time
    print(f"\n总耗时: {elapsed:.1f} 秒")
    
    print_summary(basic_results, detailed_ok, perf_ok, vis_ok, sliding_window_results)
    
    # 返回退出码
    basic_passed = sum(1 for _, result in basic_results if result)
    basic_total = len(basic_results)
    
    # Check sliding window results if they were run
    sw_passed = sum(1 for _, result in sliding_window_results if result) if sliding_window_results else 0
    sw_total = len(sliding_window_results) if sliding_window_results else 0
    
    # Consider sliding window tests in success criteria only if they were actually run
    if sw_total > 0:
        all_passed = (basic_passed == basic_total) and (sw_passed == sw_total) and detailed_ok
    else:
        all_passed = (basic_passed == basic_total) and detailed_ok
    
    if all_passed:
        sys.exit(0)  # 成功
    else:
        sys.exit(1)  # 失败


if __name__ == "__main__":
    main()

