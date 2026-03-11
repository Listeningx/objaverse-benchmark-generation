#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenShape 聚类结果可视化脚本

可视化内容：
1. 聚类大小分布（直方图）
2. 聚类质心的2D降维可视化（t-SNE / PCA）
3. 聚类统计信息
4. 各聚类的图像展示

Author: Auto-generated
Date: 2024-02-10
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm
from PIL import Image

try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ sklearn 未安装，降维可视化功能不可用")

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# ==================== 数据结构 ====================

@dataclass
class ClusterInfo:
    """聚类信息"""
    cluster_id: int
    centroid: np.ndarray
    object_ids: List[str] = field(default_factory=list)
    neighbor_cluster_ids: List[int] = field(default_factory=list)
    
    @property
    def size(self) -> int:
        return len(self.object_ids)


# ==================== 数据加载 ====================

def load_clusters(clusters_file: str):
    """
    加载聚类结果
    
    Args:
        clusters_file: clusters.pkl 文件路径
        
    Returns:
        (clusters, object_cluster_map)
    """
    with open(clusters_file, 'rb') as f:
        data = pickle.load(f)
    
    clusters = data['clusters']
    object_cluster_map = data['object_cluster_map']
    
    print(f"✅ 加载了 {len(clusters)} 个聚类")
    print(f"✅ 总物体数: {len(object_cluster_map)}")
    
    return clusters, object_cluster_map


def load_embeddings(embeddings_file: str):
    """
    加载 embeddings
    
    Args:
        embeddings_file: openshape_embeddings.npz 文件路径
        
    Returns:
        (object_ids, embeddings)
    """
    data = np.load(embeddings_file, allow_pickle=True)
    object_ids = data['object_ids'].tolist()
    embeddings = data['embeddings']
    
    print(f"✅ 加载了 {len(object_ids)} 个 embeddings")
    print(f"   Embedding 维度: {embeddings.shape[1]}")
    
    return object_ids, embeddings


def load_input_json(json_file: str):
    """
    加载输入 JSON 获取物体信息
    
    Args:
        json_file: 输入 JSON 文件路径
        
    Returns:
        objects_dict
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    objects_dict = {}
    for group in data.get('groups', []):
        for obj_data in group.get('objects', []):
            obj_id = obj_data.get('object_id', '')
            objects_dict[obj_id] = obj_data
    
    print(f"✅ 加载了 {len(objects_dict)} 个物体信息")
    return objects_dict


# ==================== 可视化函数 ====================

def plot_cluster_size_distribution(
    clusters: Dict[int, ClusterInfo],
    save_path: str = None,
    figsize: tuple = (14, 5)
):
    """
    绘制聚类大小分布
    
    Args:
        clusters: 聚类字典
        save_path: 保存路径
        figsize: 图像大小
    """
    sizes = [c.size for c in clusters.values()]
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # 1. 直方图
    ax1 = axes[0]
    ax1.hist(sizes, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
    ax1.axvline(np.mean(sizes), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(sizes):.1f}')
    ax1.axvline(np.median(sizes), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(sizes):.1f}')
    ax1.set_xlabel('Cluster Size', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Cluster Size Distribution', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. 箱线图
    ax2 = axes[1]
    bp = ax2.boxplot(sizes, patch_artist=True)
    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][0].set_alpha(0.6)
    ax2.set_ylabel('Cluster Size', fontsize=12)
    ax2.set_title('Cluster Size Boxplot', fontsize=14)
    ax2.grid(axis='y', alpha=0.3)
    
    # 添加统计信息
    stats_text = f"Min: {min(sizes)}\nMax: {max(sizes)}\nMean: {np.mean(sizes):.1f}\nMedian: {np.median(sizes):.1f}\nStd: {np.std(sizes):.1f}"
    ax2.text(1.3, np.mean(sizes), stats_text, fontsize=10, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 3. 排序后的聚类大小
    ax3 = axes[2]
    sorted_sizes = sorted(sizes, reverse=True)
    colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_sizes)))
    ax3.bar(range(len(sorted_sizes)), sorted_sizes, color=colors, edgecolor='none')
    ax3.set_xlabel('Cluster Rank', fontsize=12)
    ax3.set_ylabel('Cluster Size', fontsize=12)
    ax3.set_title('Clusters Sorted by Size', fontsize=14)
    ax3.axhline(30, color='red', linestyle='--', linewidth=1.5, label='Min Size (30)')
    ax3.legend(fontsize=10)
    ax3.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ 聚类大小分布图已保存: {save_path}")
    
    plt.show()
    plt.close()


def plot_cluster_2d_visualization(
    clusters: Dict[int, ClusterInfo],
    object_ids: List[str] = None,
    embeddings: np.ndarray = None,
    method: str = 'tsne',
    save_path: str = None,
    figsize: tuple = (24, 12),
    max_points: int = 8000,
    dpi: int = 300,
    enhanced_separation: bool = True
):
    """
    绘制聚类的2D降维可视化（高分辨率版本）
    
    Args:
        clusters: 聚类字典
        object_ids: 物体ID列表（用于绘制所有点）
        embeddings: embedding数组（用于绘制所有点）
        method: 降维方法 'tsne' 或 'pca'
        save_path: 保存路径
        figsize: 图像大小
        max_points: 最大绘制点数（避免过慢）
        dpi: 图像分辨率
        enhanced_separation: 是否增强聚类分离度
    """
    if not SKLEARN_AVAILABLE:
        print("❌ sklearn 未安装，无法进行降维可视化")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # ===== 左图：聚类质心可视化 =====
    ax1 = axes[0]
    
    # 收集质心
    cluster_ids = list(clusters.keys())
    centroids = np.array([clusters[cid].centroid for cid in cluster_ids])
    sizes = np.array([clusters[cid].size for cid in cluster_ids])
    
    print(f"\n正在对 {len(centroids)} 个聚类质心进行 {method.upper()} 降维...")
    
    # 降维（增强参数以获得更好的分离效果）
    if method.lower() == 'tsne':
        perplexity = min(50, max(5, len(centroids) // 3))  # 增加 perplexity
        reducer = TSNE(
            n_components=2, 
            random_state=42, 
            perplexity=perplexity, 
            max_iter=2000,  # 增加迭代次数
            learning_rate='auto',
            init='pca'  # 使用 PCA 初始化
        )
    else:
        reducer = PCA(n_components=2, random_state=42)
    
    centroids_2d = reducer.fit_transform(centroids)
    
    # 绘制质心（更大的点，更明显的边框）
    scatter1 = ax1.scatter(
        centroids_2d[:, 0], centroids_2d[:, 1],
        c=sizes, cmap='plasma', s=sizes * 3,  # 增大点的大小
        alpha=0.85, edgecolors='black', linewidths=1.5
    )
    
    # 添加颜色条
    cbar1 = plt.colorbar(scatter1, ax=ax1, shrink=0.8)
    cbar1.set_label('Cluster Size', fontsize=14)
    
    # 标注大聚类
    large_threshold = np.percentile(sizes, 85)
    for i, (x, y, size, cid) in enumerate(zip(centroids_2d[:, 0], centroids_2d[:, 1], sizes, cluster_ids)):
        if size >= large_threshold:
            ax1.annotate(
                f'{cid}\n({size})',
                (x, y),
                fontsize=10,
                fontweight='bold',
                ha='center',
                va='bottom',
                alpha=0.9,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='gray')
            )
    
    ax1.set_xlabel(f'{method.upper()} Dimension 1', fontsize=14)
    ax1.set_ylabel(f'{method.upper()} Dimension 2', fontsize=14)
    ax1.set_title(f'Cluster Centroids ({method.upper()})\nPoint size ∝ Cluster size', fontsize=16, fontweight='bold')
    ax1.grid(alpha=0.3, linestyle='--')
    ax1.set_facecolor('#fafafa')  # 浅灰背景
    
    # ===== 右图：所有物体可视化（按聚类着色）=====
    ax2 = axes[1]
    
    if embeddings is not None and object_ids is not None:
        # 创建物体到聚类的映射
        obj_to_cluster = {}
        for cid, cluster in clusters.items():
            for obj_id in cluster.object_ids:
                obj_to_cluster[obj_id] = cid
        
        # 筛选有聚类标签的物体
        valid_indices = []
        valid_cluster_ids = []
        for i, obj_id in enumerate(object_ids):
            if obj_id in obj_to_cluster:
                valid_indices.append(i)
                valid_cluster_ids.append(obj_to_cluster[obj_id])
        
        # 采样（如果点太多）
        if len(valid_indices) > max_points:
            print(f"  采样 {max_points} 个点（共 {len(valid_indices)} 个）")
            sample_idx = np.random.choice(len(valid_indices), max_points, replace=False)
            valid_indices = [valid_indices[i] for i in sample_idx]
            valid_cluster_ids = [valid_cluster_ids[i] for i in sample_idx]
        
        valid_embeddings = embeddings[valid_indices]
        
        print(f"正在对 {len(valid_embeddings)} 个物体进行 {method.upper()} 降维...")
        print(f"  使用增强参数以提高聚类分离度...")
        
        # 降维（优化参数以获得更好的聚类分离）
        if method.lower() == 'tsne':
            # 根据数据量调整 perplexity
            perplexity = min(100, max(30, len(valid_embeddings) // 50))
            reducer2 = TSNE(
                n_components=2, 
                random_state=42, 
                perplexity=perplexity,
                max_iter=3000,  # 更多迭代次数
                learning_rate='auto',
                init='pca',
                early_exaggeration=24.0 if enhanced_separation else 12.0,  # 增强早期放大
                metric='cosine'  # 使用余弦距离
            )
        else:
            reducer2 = PCA(n_components=2, random_state=42)
        
        points_2d = reducer2.fit_transform(valid_embeddings)
        
        # 为每个聚类分配鲜艳、高对比度的颜色
        unique_clusters = list(set(valid_cluster_ids))
        n_clusters = len(unique_clusters)
        
        # 使用多种颜色方案组合，确保高对比度
        def generate_distinct_colors(n):
            """生成 n 种高对比度颜色"""
            colors = []
            # 使用 HSV 颜色空间生成均匀分布的颜色
            import colorsys
            for i in range(n):
                hue = i / n
                # 高饱和度、高亮度
                saturation = 0.85 + 0.1 * (i % 2)  # 交替饱和度
                value = 0.9 + 0.1 * ((i // 2) % 2)  # 交替亮度
                rgb = colorsys.hsv_to_rgb(hue, saturation, value)
                colors.append(rgb + (1.0,))  # 添加 alpha
            return np.array(colors)
        
        colors = generate_distinct_colors(n_clusters)
        
        cluster_to_color = {cid: colors[i % len(colors)] for i, cid in enumerate(unique_clusters)}
        point_colors = np.array([cluster_to_color[cid] for cid in valid_cluster_ids])
        
        # 计算每个聚类的中心点位置（用于后续标注）
        cluster_to_points = {cid: [] for cid in unique_clusters}
        for i, cid in enumerate(valid_cluster_ids):
            cluster_to_points[cid].append(points_2d[i])
        
        # 绘制所有点（不绘制边界线，只用颜色区分）
        ax2.scatter(
            points_2d[:, 0], points_2d[:, 1],
            c=point_colors[:, :3], s=30, alpha=0.8,  # 更大的点，更高的不透明度
            edgecolors='white', linewidths=0.2
        )
        
        # 绘制聚类中心（用大的标记，便于识别）
        print("  标注聚类中心...")
        for cid in unique_clusters:
            pts = np.array(cluster_to_points[cid])
            center = np.mean(pts, axis=0)
            # 绘制中心点（大圆点+黑色边框）
            ax2.scatter(
                center[0], center[1],
                c=[cluster_to_color[cid][:3]], s=180,
                marker='o', edgecolors='black', linewidths=1.5,
                zorder=100, alpha=0.95
            )
            # 只标注较大的聚类（避免标签过多）
            if clusters[cid].size >= np.percentile(sizes, 75):
                ax2.annotate(
                    f'{cid}',
                    (center[0], center[1]),
                    fontsize=8,
                    fontweight='bold',
                    ha='center',
                    va='center',
                    color='white',
                    zorder=101
                )
        
        ax2.set_xlabel(f'{method.upper()} Dimension 1', fontsize=14)
        ax2.set_ylabel(f'{method.upper()} Dimension 2', fontsize=14)
        ax2.set_title(f'All Objects ({method.upper()}) - {len(valid_embeddings)} points\nColored by Cluster', fontsize=16, fontweight='bold')
        ax2.grid(alpha=0.2, linestyle='--')
        ax2.set_facecolor('#fafafa')
        
        # 添加图例（只显示最大的几个聚类）
        top_clusters = sorted(unique_clusters, key=lambda c: clusters[c].size, reverse=True)[:15]
        legend_handles = [
            mpatches.Patch(color=cluster_to_color[cid][:3], label=f'Cluster {cid} ({clusters[cid].size})')
            for cid in top_clusters
        ]
        ax2.legend(handles=legend_handles, loc='upper right', fontsize=9, 
                   title='Top 15 Clusters', title_fontsize=10,
                   framealpha=0.9, edgecolor='gray')
    else:
        ax2.text(0.5, 0.5, 'No embedding data available', 
                 ha='center', va='center', transform=ax2.transAxes, fontsize=14)
        ax2.set_title('All Objects Visualization', fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"✅ 2D可视化图已保存: {save_path} (分辨率: {dpi} dpi)")
    
    plt.show()
    plt.close()


def plot_cluster_statistics(
    clusters: Dict[int, ClusterInfo],
    save_path: str = None,
    figsize: tuple = (12, 8)
):
    """
    绘制聚类统计信息
    
    Args:
        clusters: 聚类字典
        save_path: 保存路径
        figsize: 图像大小
    """
    sizes = [c.size for c in clusters.values()]
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. 累积分布
    ax1 = axes[0, 0]
    sorted_sizes = np.sort(sizes)
    cumulative = np.arange(1, len(sorted_sizes) + 1) / len(sorted_sizes)
    ax1.plot(sorted_sizes, cumulative, 'b-', linewidth=2)
    ax1.axvline(30, color='red', linestyle='--', label='Min Size (30)')
    ax1.set_xlabel('Cluster Size', fontsize=12)
    ax1.set_ylabel('Cumulative Proportion', fontsize=12)
    ax1.set_title('Cumulative Distribution of Cluster Sizes', fontsize=14)
    ax1.grid(alpha=0.3)
    ax1.legend()
    
    # 2. 大小区间分布
    ax2 = axes[0, 1]
    bins = [0, 30, 50, 100, 200, 500, max(sizes) + 1]
    bin_labels = ['<30', '30-50', '50-100', '100-200', '200-500', '>500']
    counts = []
    for i in range(len(bins) - 1):
        count = sum(1 for s in sizes if bins[i] <= s < bins[i + 1])
        counts.append(count)
    
    colors = ['red' if l == '<30' else 'steelblue' for l in bin_labels]
    bars = ax2.bar(bin_labels, counts, color=colors, edgecolor='white')
    ax2.set_xlabel('Size Range', fontsize=12)
    ax2.set_ylabel('Number of Clusters', fontsize=12)
    ax2.set_title('Cluster Size Ranges', fontsize=14)
    ax2.grid(axis='y', alpha=0.3)
    
    # 在柱子上添加数字
    for bar, count in zip(bars, counts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 str(count), ha='center', va='bottom', fontsize=10)
    
    # 3. 前N大聚类
    ax3 = axes[1, 0]
    top_n = 20
    sorted_clusters = sorted(clusters.items(), key=lambda x: x[1].size, reverse=True)[:top_n]
    cluster_ids = [str(c[0]) for c in sorted_clusters]
    cluster_sizes = [c[1].size for c in sorted_clusters]
    
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, top_n))
    ax3.barh(range(top_n), cluster_sizes, color=colors)
    ax3.set_yticks(range(top_n))
    ax3.set_yticklabels(cluster_ids)
    ax3.invert_yaxis()
    ax3.set_xlabel('Cluster Size', fontsize=12)
    ax3.set_ylabel('Cluster ID', fontsize=12)
    ax3.set_title(f'Top {top_n} Largest Clusters', fontsize=14)
    ax3.grid(axis='x', alpha=0.3)
    
    # 在柱子右侧添加数字
    for i, size in enumerate(cluster_sizes):
        ax3.text(size + 1, i, str(size), va='center', fontsize=9)
    
    # 4. 统计摘要表格
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # 计算统计信息
    total_objects = sum(sizes)
    valid_clusters = sum(1 for s in sizes if s >= 30)
    invalid_clusters = len(sizes) - valid_clusters
    
    stats_data = [
        ['Total Clusters', str(len(clusters))],
        ['Total Objects', str(total_objects)],
        ['Valid Clusters (≥30)', str(valid_clusters)],
        ['Invalid Clusters (<30)', str(invalid_clusters)],
        ['', ''],
        ['Min Size', str(min(sizes))],
        ['Max Size', str(max(sizes))],
        ['Mean Size', f'{np.mean(sizes):.1f}'],
        ['Median Size', f'{np.median(sizes):.1f}'],
        ['Std Dev', f'{np.std(sizes):.1f}'],
        ['', ''],
        ['25th Percentile', f'{np.percentile(sizes, 25):.1f}'],
        ['75th Percentile', f'{np.percentile(sizes, 75):.1f}'],
        ['90th Percentile', f'{np.percentile(sizes, 90):.1f}'],
    ]
    
    table = ax4.table(
        cellText=stats_data,
        colLabels=['Metric', 'Value'],
        loc='center',
        cellLoc='left',
        colWidths=[0.5, 0.3]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)
    
    # 设置表头样式
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#4472C4')
            cell.set_text_props(color='white', weight='bold')
        elif row % 2 == 0:
            cell.set_facecolor('#D6DCE5')
    
    ax4.set_title('Cluster Statistics Summary', fontsize=14, pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ 统计信息图已保存: {save_path}")
    
    plt.show()
    plt.close()


def plot_cluster_neighbor_heatmap(
    clusters: Dict[int, ClusterInfo],
    save_path: str = None,
    figsize: tuple = (12, 10),
    max_clusters: int = 30
):
    """
    绘制聚类邻居关系热力图
    
    Args:
        clusters: 聚类字典
        save_path: 保存路径
        figsize: 图像大小
        max_clusters: 最大显示聚类数
    """
    # 只选择最大的几个聚类
    sorted_clusters = sorted(clusters.items(), key=lambda x: x[1].size, reverse=True)
    selected_clusters = sorted_clusters[:max_clusters]
    selected_ids = [c[0] for c in selected_clusters]
    
    n = len(selected_ids)
    id_to_idx = {cid: i for i, cid in enumerate(selected_ids)}
    
    # 计算相似度矩阵
    centroids = np.array([clusters[cid].centroid for cid in selected_ids])
    similarity_matrix = np.dot(centroids, centroids.T)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(similarity_matrix, cmap='RdYlBu_r', aspect='auto')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Cosine Similarity', fontsize=12)
    
    # 设置刻度
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f'{cid}\n({clusters[cid].size})' for cid in selected_ids], fontsize=8, rotation=45, ha='right')
    ax.set_yticklabels([f'{cid} ({clusters[cid].size})' for cid in selected_ids], fontsize=8)
    
    ax.set_xlabel('Cluster ID (Size)', fontsize=12)
    ax.set_ylabel('Cluster ID (Size)', fontsize=12)
    ax.set_title(f'Cluster Similarity Heatmap (Top {n} Clusters)', fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ 邻居关系热力图已保存: {save_path}")
    
    plt.show()
    plt.close()


def visualize_cluster_images(
    clusters: Dict[int, ClusterInfo],
    objects_dict: Dict[str, dict],
    cluster_id: int,
    save_path: str = None,
    max_images: int = 20,
    images_per_row: int = 5,
    image_size: tuple = (100, 100)
):
    """
    可视化某个聚类中的图像
    
    Args:
        clusters: 聚类字典
        objects_dict: 物体信息字典
        cluster_id: 要可视化的聚类ID
        save_path: 保存路径
        max_images: 最大显示图像数
        images_per_row: 每行图像数
        image_size: 图像缩放大小
    """
    if cluster_id not in clusters:
        print(f"❌ 聚类 {cluster_id} 不存在")
        return
    
    cluster = clusters[cluster_id]
    object_ids = cluster.object_ids[:max_images]
    
    n_images = len(object_ids)
    n_rows = (n_images + images_per_row - 1) // images_per_row
    
    fig, axes = plt.subplots(n_rows, images_per_row, figsize=(images_per_row * 2.5, n_rows * 2.5))
    
    if n_rows == 1:
        axes = [axes]
    
    for i, obj_id in enumerate(object_ids):
        row = i // images_per_row
        col = i % images_per_row
        ax = axes[row][col] if images_per_row > 1 else axes[row]
        
        obj_info = objects_dict.get(obj_id, {})
        image_path = obj_info.get('image_path', '')
        
        if image_path and os.path.exists(image_path):
            try:
                img = Image.open(image_path).convert('RGB')
                img = img.resize(image_size)
                ax.imshow(img)
            except Exception as e:
                ax.text(0.5, 0.5, 'Error', ha='center', va='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, 'No Image', ha='center', va='center', transform=ax.transAxes)
        
        ax.set_title(f'{obj_id[:15]}...', fontsize=8)
        ax.axis('off')
    
    # 隐藏空白子图
    for i in range(n_images, n_rows * images_per_row):
        row = i // images_per_row
        col = i % images_per_row
        ax = axes[row][col] if images_per_row > 1 else axes[row]
        ax.axis('off')
    
    plt.suptitle(f'Cluster {cluster_id} (Size: {cluster.size})', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ 聚类 {cluster_id} 图像已保存: {save_path}")
    
    plt.show()
    plt.close()


def print_cluster_summary(clusters: Dict[int, ClusterInfo]):
    """
    打印聚类摘要
    
    Args:
        clusters: 聚类字典
    """
    sizes = [c.size for c in clusters.values()]
    total_objects = sum(sizes)
    valid_clusters = sum(1 for s in sizes if s >= 30)
    
    print("\n" + "=" * 60)
    print("📊 聚类统计摘要")
    print("=" * 60)
    print(f"总聚类数: {len(clusters)}")
    print(f"总物体数: {total_objects}")
    print(f"有效聚类数 (≥30): {valid_clusters}")
    print(f"无效聚类数 (<30): {len(clusters) - valid_clusters}")
    print("-" * 60)
    print(f"最小聚类大小: {min(sizes)}")
    print(f"最大聚类大小: {max(sizes)}")
    print(f"平均聚类大小: {np.mean(sizes):.1f}")
    print(f"中位聚类大小: {np.median(sizes):.1f}")
    print(f"标准差: {np.std(sizes):.1f}")
    print("-" * 60)
    print(f"25% 分位数: {np.percentile(sizes, 25):.1f}")
    print(f"50% 分位数: {np.percentile(sizes, 50):.1f}")
    print(f"75% 分位数: {np.percentile(sizes, 75):.1f}")
    print(f"90% 分位数: {np.percentile(sizes, 90):.1f}")
    print("=" * 60)
    
    # 打印前10大聚类
    print("\n🏆 Top 10 最大聚类:")
    sorted_clusters = sorted(clusters.items(), key=lambda x: x[1].size, reverse=True)[:10]
    for i, (cid, cluster) in enumerate(sorted_clusters, 1):
        print(f"  {i:2d}. Cluster {cid:3d}: {cluster.size} 个物体")
    
    print()


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(description='OpenShape 聚类结果可视化')
    
    parser.add_argument('--clusters', type=str, required=True,
                        help='clusters.pkl 文件路径')
    parser.add_argument('--embeddings', type=str, default=None,
                        help='openshape_embeddings.npz 文件路径（可选，用于2D可视化）')
    parser.add_argument('--input_json', type=str, default=None,
                        help='输入 JSON 文件（可选，用于获取图像路径）')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录（默认与 clusters 同目录）')
    parser.add_argument('--method', type=str, default='pca',
                        choices=['tsne', 'pca'],
                        help='降维方法')
    parser.add_argument('--show_cluster', type=int, default=None,
                        help='显示指定聚类的图像')
    
    args = parser.parse_args()
    
    # 设置输出目录
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.clusters)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("🎨 OpenShape 聚类结果可视化")
    print("=" * 70)
    print(f"聚类文件: {args.clusters}")
    print(f"输出目录: {args.output_dir}")
    print("=" * 70)
    
    # 加载数据
    clusters, object_cluster_map = load_clusters(args.clusters)
    
    # 打印摘要
    print_cluster_summary(clusters)
    
    # 加载 embeddings（如果提供）
    object_ids = None
    embeddings = None
    if args.embeddings and os.path.exists(args.embeddings):
        object_ids, embeddings = load_embeddings(args.embeddings)
    
    # 加载物体信息（如果提供）
    objects_dict = None
    if args.input_json and os.path.exists(args.input_json):
        objects_dict = load_input_json(args.input_json)
    
    # 1. 绘制聚类大小分布
    print("\n📈 绘制聚类大小分布...")
    plot_cluster_size_distribution(
        clusters,
        save_path=os.path.join(args.output_dir, 'cluster_size_distribution.png')
    )
    
    # 2. 绘制统计信息
    print("\n📊 绘制统计信息...")
    plot_cluster_statistics(
        clusters,
        save_path=os.path.join(args.output_dir, 'cluster_statistics.png')
    )
    
    # 3. 绘制2D可视化
    print(f"\n🗺️ 绘制 {args.method.upper()} 2D可视化...")
    plot_cluster_2d_visualization(
        clusters,
        object_ids=object_ids,
        embeddings=embeddings,
        method=args.method,
        save_path=os.path.join(args.output_dir, f'cluster_2d_{args.method}.png')
    )
    
    # 4. 绘制邻居关系热力图
    print("\n🔥 绘制邻居关系热力图...")
    plot_cluster_neighbor_heatmap(
        clusters,
        save_path=os.path.join(args.output_dir, 'cluster_neighbor_heatmap.png')
    )
    
    # 5. 可视化指定聚类的图像（如果指定）
    if args.show_cluster is not None and objects_dict is not None:
        print(f"\n🖼️ 可视化聚类 {args.show_cluster} 的图像...")
        visualize_cluster_images(
            clusters,
            objects_dict,
            args.show_cluster,
            save_path=os.path.join(args.output_dir, f'cluster_{args.show_cluster}_images.png')
        )
    
    print("\n" + "=" * 70)
    print("✅ 可视化完成！")
    print(f"📁 输出目录: {args.output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
