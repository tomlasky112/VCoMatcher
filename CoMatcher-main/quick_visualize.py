"""
快速可视化检查工具 - 一键生成所有可视化报告
============================================

生成以下可视化：
1. 相机位姿 3D 图
2. 深度图和掩膜对比
3. 重叠矩阵热图
4. 样本分布统计
5. 反投影误差分析

Author: VCoMatcher Team
Date: 2025-12-13
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import json


def visualize_camera_poses(data: dict, output_path: Path):
    """可视化相机位姿（3D 俯视图和侧视图）"""
    extrinsic = data["extrinsic"]  # [N, 4, 4]
    N = extrinsic.shape[0]
    
    # BUGFIX: Handle single camera case
    if N < 2:
        print(f"  ⚠️  只有 {N} 个相机，跳过位姿可视化")
        return
    
    # 提取相机位置（世界坐标系）
    camera_positions = []
    camera_directions = []
    
    for i in range(N):
        # extrinsic 是 w2c，所以相机位置是 -R^T @ t
        R = extrinsic[i, :3, :3]
        t = extrinsic[i, :3, 3]
        cam_pos = -R.T @ t
        camera_positions.append(cam_pos)
        
        # 相机朝向（Z 轴方向）
        cam_dir = R.T @ np.array([0, 0, 1])
        camera_directions.append(cam_dir)
    
    camera_positions = np.array(camera_positions)
    camera_directions = np.array(camera_directions)
    
    # 创建图形
    fig = plt.figure(figsize=(20, 6))
    
    # 1. 俯视图 (X-Z 平面)
    ax1 = fig.add_subplot(131)
    ax1.scatter(camera_positions[:, 0], camera_positions[:, 2], c=range(N), cmap='viridis', s=100)
    ax1.quiver(camera_positions[:, 0], camera_positions[:, 2], 
               camera_directions[:, 0], camera_directions[:, 2],
               scale=5, width=0.005, color='red', alpha=0.6)
    for i in range(N):
        ax1.text(camera_positions[i, 0], camera_positions[i, 2], str(i), fontsize=8)
    ax1.set_xlabel('X (right)')
    ax1.set_ylabel('Z (forward)')
    ax1.set_title('Camera Poses - Top View (X-Z)')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # 2. 侧视图 (Y-Z 平面)
    ax2 = fig.add_subplot(132)
    ax2.scatter(camera_positions[:, 2], camera_positions[:, 1], c=range(N), cmap='viridis', s=100)
    ax2.quiver(camera_positions[:, 2], camera_positions[:, 1],
               camera_directions[:, 2], camera_directions[:, 1],
               scale=5, width=0.005, color='red', alpha=0.6)
    for i in range(N):
        ax2.text(camera_positions[i, 2], camera_positions[i, 1], str(i), fontsize=8)
    ax2.set_xlabel('Z (forward)')
    ax2.set_ylabel('Y (down)')
    ax2.set_title('Camera Poses - Side View (Y-Z)')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # 3. 3D 图
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(camera_positions[:, 0], camera_positions[:, 2], camera_positions[:, 1],
                c=range(N), cmap='viridis', s=100)
    ax3.quiver(camera_positions[:, 0], camera_positions[:, 2], camera_positions[:, 1],
               camera_directions[:, 0], camera_directions[:, 2], camera_directions[:, 1],
               length=0.2, color='red', alpha=0.6)
    ax3.set_xlabel('X (right)')
    ax3.set_ylabel('Z (forward)')
    ax3.set_zlabel('Y (down)')
    ax3.set_title('Camera Poses - 3D View')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ 相机位姿可视化: {output_path}")


def visualize_depth_and_masks(data: dict, output_path: Path):
    """可视化深度图和掩膜"""
    depth = data["depth"]
    mask_geom = data["mask_geom"]
    mask_loss = data["mask_loss"]
    points_conf = data["points_conf"]
    
    N = min(6, depth.shape[0])  # 最多显示6张图
    
    fig, axes = plt.subplots(N, 4, figsize=(16, 4*N))
    if N == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(N):
        # 深度图
        im0 = axes[i, 0].imshow(depth[i], cmap='viridis')
        axes[i, 0].set_title(f'Image {i}: Depth Map')
        axes[i, 0].axis('off')
        plt.colorbar(im0, ax=axes[i, 0], fraction=0.046)
        
        # 不确定性图（VGGT: 值越大越不确定，范围 [1, ∞)）
        im1 = axes[i, 1].imshow(points_conf[i], cmap='hot_r', vmin=1, vmax=10)
        axes[i, 1].set_title(f'Image {i}: Uncertainty (σ_P, lower=better)')
        axes[i, 1].axis('off')
        plt.colorbar(im1, ax=axes[i, 1], fraction=0.046)
        
        # mask_geom
        axes[i, 2].imshow(mask_geom[i], cmap='gray')
        geom_coverage = mask_geom[i].sum() / mask_geom[i].size * 100
        axes[i, 2].set_title(f'mask_geom: {geom_coverage:.1f}%')
        axes[i, 2].axis('off')
        
        # mask_loss
        axes[i, 3].imshow(mask_loss[i], cmap='gray')
        loss_coverage = mask_loss[i].sum() / mask_loss[i].size * 100
        axes[i, 3].set_title(f'mask_loss: {loss_coverage:.1f}%')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ 深度图和掩膜可视化: {output_path}")


def visualize_overlap_matrix(data: dict, output_path: Path):
    """可视化重叠矩阵"""
    overlap_matrix = data["overlap_matrix"]
    N = overlap_matrix.shape[0]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. 完整重叠矩阵
    im1 = axes[0].imshow(overlap_matrix, cmap='hot', vmin=0, vmax=1)
    axes[0].set_title(f'Overlap Matrix ({N}x{N})')
    axes[0].set_xlabel('Target Image Index')
    axes[0].set_ylabel('Source Image Index')
    plt.colorbar(im1, ax=axes[0])
    
    # 添加网格
    for i in range(0, N, 5):
        axes[0].axhline(i-0.5, color='gray', linewidth=0.5, alpha=0.3)
        axes[0].axvline(i-0.5, color='gray', linewidth=0.5, alpha=0.3)
    
    # 2. 重叠分数分布
    overlap_off_diag = overlap_matrix[~np.eye(N, dtype=bool)]
    axes[1].hist(overlap_off_diag, bins=50, edgecolor='black', alpha=0.7)
    axes[1].axvline(0.05, color='red', linestyle='--', label='Extreme threshold (0.05)')
    axes[1].axvline(0.1, color='orange', linestyle='--', label='Hard threshold (0.1)')
    axes[1].axvline(0.4, color='green', linestyle='--', label='Easy threshold (0.4)')
    axes[1].axvline(0.7, color='blue', linestyle='--', label='Upper threshold (0.7)')
    axes[1].set_xlabel('Overlap Score')
    axes[1].set_ylabel('Number of Image Pairs')
    axes[1].set_title('Overlap Score Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ 重叠矩阵可视化: {output_path}")


def visualize_sample_distribution(data: dict, output_path: Path):
    """可视化样本分布"""
    samples = data["samples"]
    
    # 统计
    sample_types = [s["sample_type"] for s in samples]
    overlap_scores = [s["overlap_score"] for s in samples]
    
    n_easy = sample_types.count("easy")
    n_hard = sample_types.count("hard")
    n_extreme = sample_types.count("extreme")
    n_total = len(samples)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. 饼图
    sizes = [n_easy, n_hard, n_extreme]
    
    # BUGFIX: Handle division by zero when n_total is 0
    if n_total > 0:
        labels = [f'Easy\n{n_easy} ({n_easy/n_total*100:.1f}%)',
                  f'Hard\n{n_hard} ({n_hard/n_total*100:.1f}%)',
                  f'Extreme\n{n_extreme} ({n_extreme/n_total*100:.1f}%)']
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        
        # BUGFIX: Filter out zero-size slices (matplotlib warning)
        sizes_filtered = [s for s in sizes if s > 0]
        labels_filtered = [labels[i] for i in range(3) if sizes[i] > 0]
        colors_filtered = [colors[i] for i in range(3) if sizes[i] > 0]
        
        if len(sizes_filtered) > 0:
            axes[0].pie(sizes_filtered, labels=labels_filtered, colors=colors_filtered,
                       autopct='%1.1f%%', startangle=90, textprops={'fontsize': 12})
            axes[0].set_title(f'Sample Distribution (Total: {n_total})')
        else:
            axes[0].text(0.5, 0.5, 'No samples',
                        ha='center', va='center', transform=axes[0].transAxes)
            axes[0].set_title('Sample Distribution (Empty)')
    else:
        axes[0].text(0.5, 0.5, 'No samples',
                    ha='center', va='center', transform=axes[0].transAxes)
        axes[0].set_title('Sample Distribution (Empty)')
    
    # 2. 重叠分数分布（按类型着色）
    easy_scores = [s["overlap_score"] for s in samples if s["sample_type"] == "easy"]
    hard_scores = [s["overlap_score"] for s in samples if s["sample_type"] == "hard"]
    extreme_scores = [s["overlap_score"] for s in samples if s["sample_type"] == "extreme"]
    
    # BUGFIX: Only plot if we have data
    has_data = len(easy_scores) + len(hard_scores) + len(extreme_scores) > 0
    if has_data:
        axes[1].hist([easy_scores, hard_scores, extreme_scores], bins=30,
                     label=['Easy (0.4-0.7)', 'Hard (0.1-0.4)', 'Extreme (0.05-0.1)'],
                     color=colors, alpha=0.7, stacked=True)
    else:
        axes[1].text(0.5, 0.5, 'No samples available',
                    ha='center', va='center', transform=axes[1].transAxes, fontsize=14)
    axes[1].axvline(0.05, color='red', linestyle='--', linewidth=1)
    axes[1].axvline(0.1, color='orange', linestyle='--', linewidth=1)
    axes[1].axvline(0.4, color='green', linestyle='--', linewidth=1)
    axes[1].axvline(0.7, color='blue', linestyle='--', linewidth=1)
    axes[1].set_xlabel('Overlap Score')
    axes[1].set_ylabel('Number of Samples')
    axes[1].set_title('Sample Distribution by Overlap Score')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ 样本分布可视化: {output_path}")


def visualize_statistics(data: dict, output_path: Path):
    """生成统计摘要图"""
    depth = data["depth"]
    points_conf = data["points_conf"]
    mask_geom = data["mask_geom"]
    mask_loss = data["mask_loss"]
    extrinsic = data["extrinsic"]
    
    N = depth.shape[0]
    
    # BUGFIX: Handle case where N is 0
    if N == 0:
        print(f"  ⚠️  没有图像数据，跳过统计可视化")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 深度统计（每张图像）
    depth_means = []
    depth_stds = []
    for i in range(N):
        valid_depth = depth[i][mask_loss[i]]
        if len(valid_depth) > 0:
            depth_means.append(np.mean(valid_depth))
            depth_stds.append(np.std(valid_depth))
        else:
            # BUGFIX: Use NaN instead of 0 for missing data (clearer visualization)
            depth_means.append(np.nan)
            depth_stds.append(np.nan)
    
    # BUGFIX: Filter out NaN values for plotting
    valid_indices = [i for i in range(N) if not np.isnan(depth_means[i])]
    if len(valid_indices) > 0:
        axes[0, 0].bar(valid_indices, [depth_means[i] for i in valid_indices], 
                      yerr=[depth_stds[i] for i in valid_indices], capsize=5, alpha=0.7)
        axes[0, 0].set_xlabel('Image Index')
        axes[0, 0].set_ylabel('Mean Depth ± Std')
        axes[0, 0].set_title('Depth Statistics per Image')
    else:
        axes[0, 0].text(0.5, 0.5, 'No valid depth data',
                       ha='center', va='center', transform=axes[0, 0].transAxes)
        axes[0, 0].set_title('Depth Statistics (Empty)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 置信度分布
    all_conf = points_conf.flatten()
    valid_conf = all_conf[all_conf > 0]
    # BUGFIX: Handle empty confidence array
    if len(valid_conf) > 0:
        axes[0, 1].hist(valid_conf, bins=50, edgecolor='black', alpha=0.7)
    else:
        axes[0, 1].text(0.5, 0.5, 'No valid confidence data',
                       ha='center', va='center', transform=axes[0, 1].transAxes)
    
    # 从数据中读取实际的 tau_uncertainty
    tau_unc = float(data.get('tau_uncertainty', 15.0))
    axes[0, 1].axvline(tau_unc, color='red', linestyle='--', 
                      label=f'tau_uncertainty ({tau_unc:.1f})')
    axes[0, 1].set_xlabel('Confidence (σ_P) [Uncertainty, lower=better]')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Point Confidence Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 掩膜覆盖率（每张图像）
    geom_coverage = [mask_geom[i].sum() / mask_geom[i].size * 100 for i in range(N)]
    loss_coverage = [mask_loss[i].sum() / mask_loss[i].size * 100 for i in range(N)]
    
    x = np.arange(N)
    width = 0.35
    axes[1, 0].bar(x - width/2, geom_coverage, width, label='mask_geom', alpha=0.7)
    axes[1, 0].bar(x + width/2, loss_coverage, width, label='mask_loss', alpha=0.7)
    axes[1, 0].set_xlabel('Image Index')
    axes[1, 0].set_ylabel('Coverage (%)')
    axes[1, 0].set_title('Mask Coverage per Image')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 相机基线分布
    translations = extrinsic[:, :3, 3]
    baselines = []
    
    # BUGFIX: Handle case where N < 2
    if N >= 2:
        for i in range(N):
            for j in range(i+1, N):
                baseline = np.linalg.norm(translations[i] - translations[j])
                baselines.append(baseline)
    
    if len(baselines) > 0:
        axes[1, 1].hist(baselines, bins=50, edgecolor='black', alpha=0.7)
        axes[1, 1].set_xlabel('Baseline Distance')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Camera Baseline Distribution')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 添加统计信息
        stats_text = f"Mean: {np.mean(baselines):.3f}\n"
        stats_text += f"Std: {np.std(baselines):.3f}\n"
        stats_text += f"Min: {np.min(baselines):.3f}\n"
        stats_text += f"Max: {np.max(baselines):.3f}"
        axes[1, 1].text(0.95, 0.95, stats_text, transform=axes[1, 1].transAxes,
                        fontsize=10, verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        axes[1, 1].text(0.5, 0.5, 'N < 2\nNo baselines', 
                       transform=axes[1, 1].transAxes,
                       ha='center', va='center', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ 统计摘要可视化: {output_path}")


def generate_summary_report(data: dict, inspection_results: dict, output_path: Path):
    """生成文本摘要报告"""
    N = data["depth"].shape[0]
    samples = data["samples"]
    
    # BUGFIX: Handle case where samples might be empty
    n_total = len(samples)
    if n_total == 0:
        n_easy = n_hard = n_extreme = 0
    else:
        n_easy = sum(1 for s in samples if s["sample_type"] == "easy")
        n_hard = sum(1 for s in samples if s["sample_type"] == "hard")
        n_extreme = sum(1 for s in samples if s["sample_type"] == "extreme")
    
    report = []
    report.append("=" * 80)
    report.append("VCoMatcher Phase 1 数据质量报告")
    report.append("=" * 80)
    report.append("")
    
    # 基本信息
    report.append("【基本信息】")
    report.append(f"  文件: {inspection_results['file_path']}")
    report.append(f"  大小: {inspection_results['file_size_mb']:.2f} MB")
    report.append(f"  图像数量: {N}")
    report.append(f"  分辨率: {data['resolution']} x {data['resolution']}")
    report.append("")
    
    # 样本统计
    report.append("【样本统计】")
    report.append(f"  Total: {n_total}")
    if n_total > 0:
        report.append(f"  Easy (0.4-0.7):     {n_easy:6d} ({n_easy/n_total*100:5.1f}%)")
        report.append(f"  Hard (0.1-0.4):     {n_hard:6d} ({n_hard/n_total*100:5.1f}%)")
        report.append(f"  Extreme (0.05-0.1): {n_extreme:6d} ({n_extreme/n_total*100:5.1f}%)")
    else:
        report.append(f"  ⚠️  没有有效样本")
    report.append("")
    
    # 掩膜覆盖率
    report.append("【掩膜覆盖率】")
    report.append(f"  mask_geom: {inspection_results['mask_geom_coverage']:.2f}%")
    report.append(f"  mask_loss: {inspection_results['mask_loss_coverage']:.2f}%")
    report.append("")
    
    # 质量评估
    report.append("【质量评估】")
    if len(inspection_results['issues']) == 0:
        report.append("  ✅ 所有检查通过，数据质量良好")
    else:
        report.append(f"  ⚠️  发现 {len(inspection_results['issues'])} 个问题:")
        for issue in inspection_results['issues']:
            report.append(f"    - {issue}")
    report.append("")
    
    # 参数设置
    report.append("【参数设置】")
    report.append(f"  tau_min: {data['tau_min']}")
    report.append(f"  tau_max: {data['tau_max']}")
    report.append(f"  tau_uncertainty: {data['tau_uncertainty']}")
    report.append("")
    
    report.append("=" * 80)
    
    # 保存报告
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    # 也打印到控制台
    print('\n'.join(report))


def main():
    parser = argparse.ArgumentParser(
        description="快速可视化检查 Phase 1 数据"
    )
    parser.add_argument(
        "--data_file",
        type=str,
        required=True,
        help=".npz 文件路径",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出目录（默认与数据文件同目录）",
    )
    
    args = parser.parse_args()
    
    # 加载数据
    data_file = Path(args.data_file)
    if not data_file.exists():
        print(f"❌ 文件不存在: {data_file}")
        return
    
    print(f"\n{'='*80}")
    print(f"快速可视化检查: {data_file.name}")
    print(f"{'='*80}\n")
    
    print("加载数据...")
    # BUGFIX: Close NpzFile after loading to prevent file handle leak
    npz_file = np.load(data_file, allow_pickle=True)
    data_dict = {key: npz_file[key] for key in npz_file.files}
    npz_file.close()
    
    # 设置输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = data_file.parent / "visualizations"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载检查结果
    inspection_file = data_file.parent / f"{data_file.stem}_inspection.json"
    if inspection_file.exists():
        with open(inspection_file, 'r') as f:
            inspection_results = json.load(f)
    else:
        print(f"⚠️  未找到检查结果文件: {inspection_file}")
        inspection_results = {
            "file_path": str(data_file),
            "file_size_mb": data_file.stat().st_size / (1024*1024),
            "issues": [],
            "n_images": data_dict["depth"].shape[0],
            "resolution": list(data_dict["depth"].shape[1:]),
            "n_samples": len(data_dict["samples"]),
            "mask_geom_coverage": data_dict["mask_geom"].sum() / data_dict["mask_geom"].size * 100,
            "mask_loss_coverage": data_dict["mask_loss"].sum() / data_dict["mask_loss"].size * 100,
        }
    
    # 生成可视化
    print("\n生成可视化...")
    
    visualize_camera_poses(data_dict, output_dir / f"{data_file.stem}_camera_poses.png")
    visualize_depth_and_masks(data_dict, output_dir / f"{data_file.stem}_depth_masks.png")
    visualize_overlap_matrix(data_dict, output_dir / f"{data_file.stem}_overlap_matrix.png")
    visualize_sample_distribution(data_dict, output_dir / f"{data_file.stem}_sample_distribution.png")
    visualize_statistics(data_dict, output_dir / f"{data_file.stem}_statistics.png")
    
    # 生成文本报告
    print("\n生成摘要报告...")
    generate_summary_report(data_dict, inspection_results, output_dir / f"{data_file.stem}_report.txt")
    
    print(f"\n{'='*80}")
    print(f"✅ 可视化完成！")
    print(f"{'='*80}")
    print(f"\n所有文件保存在: {output_dir}")
    print(f"\n生成的文件:")
    print(f"  1. {data_file.stem}_camera_poses.png       - 相机位姿 3D 可视化")
    print(f"  2. {data_file.stem}_depth_masks.png        - 深度图和掩膜对比")
    print(f"  3. {data_file.stem}_overlap_matrix.png     - 重叠矩阵热图")
    print(f"  4. {data_file.stem}_sample_distribution.png - 样本分布统计")
    print(f"  5. {data_file.stem}_statistics.png         - 统计摘要")
    print(f"  6. {data_file.stem}_report.txt             - 文本摘要报告")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()

