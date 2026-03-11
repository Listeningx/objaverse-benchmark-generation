"""
排序结果可视化脚本

用于可视化 openshape_clustering_output 目录下的排序结果文件，
展示每个 case 的查询物体和排序结果，并标注排名得分。

使用方法:
    python visualize_ranking_results.py --input <result_json_file> [--output_dir <output_directory>]
    python visualize_ranking_results.py --input_dir <result_directory> [--output_dir <output_directory>]
"""

import os
import sys
import json
import glob
import argparse
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import matplotlib.font_manager as fm

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def load_ranking_result(json_path: str) -> dict:
    """
    加载排序结果 JSON 文件
    
    Args:
        json_path: JSON 文件路径
        
    Returns:
        dict: 排序结果数据
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_image_safe(image_path: str) -> Optional[np.ndarray]:
    """
    安全加载图像，如果失败返回 None
    
    Args:
        image_path: 图像路径
        
    Returns:
        numpy.ndarray 或 None
    """
    if not os.path.exists(image_path):
        return None
    try:
        img = Image.open(image_path).convert('RGB')
        return np.array(img)
    except Exception as e:
        print(f"⚠️ 加载图像失败: {image_path}, 错误: {e}")
        return None


def get_object_score(
    obj_id: str, 
    case_data: dict, 
    rank: int = None, 
    total: int = None
) -> Tuple[float, str]:
    """
    获取物体的得分（优先使用真实的加权分数，否则根据排名计算）
    
    分数来源优先级：
    1. weighted_scores - Agent 精排的加权分数（仅对精排的20个物体有效）
    2. distant_similarity_scores - 较远cluster物体的余弦相似度（不参与精排）
    3. neighbor_similarity_scores - 兼容旧版本的邻居物体分数
    4. other_category_info - 其他类别物体（不参与精排）
    5. 根据排名计算的分数（作为兜底）
    
    Args:
        obj_id: 物体ID
        case_data: case数据
        rank: 排名（从1开始）
        total: 总物体数
        
    Returns:
        (score, score_type): 分数值和分数类型 ("weighted", "similarity", "other", "rank")
    """
    # 尝试从 weighted_scores 获取（精排物体，最高优先级）
    weighted_scores = case_data.get('weighted_scores', {})
    if obj_id in weighted_scores:
        score_data = weighted_scores[obj_id]
        if isinstance(score_data, dict):
            return score_data.get('total_score', 0.0), "weighted"
        return float(score_data), "weighted"
    
    # 尝试从 distant_similarity_scores 获取（较远cluster物体）
    distant_scores = case_data.get('distant_similarity_scores', {})
    if obj_id in distant_scores:
        # 余弦相似度范围是 [-1, 1]，转换为 0-5 的分数
        similarity = distant_scores[obj_id]
        score = (similarity + 1) * 2.5
        return score, "similarity"
    
    # 兼容旧版本：尝试从 neighbor_similarity_scores 获取
    neighbor_scores = case_data.get('neighbor_similarity_scores', {})
    if obj_id in neighbor_scores:
        similarity = neighbor_scores[obj_id]
        score = (similarity + 1) * 2.5
        return score, "similarity"
    
    # 尝试从 other_category_info 获取（其他类别物体）
    other_category_info = case_data.get('other_category_info', {})
    if obj_id in other_category_info:
        # 其他类别物体没有分数，用固定值表示
        return 0.0, "other"
    
    # 如果都没有，根据排名计算
    if rank is not None and total is not None:
        return calculate_rank_score(rank, total), "rank"
    
    return 0.0, "unknown"


def calculate_rank_score(rank: int, total: int) -> float:
    """
    计算排名得分（排名越高得分越高）
    
    Args:
        rank: 排名（从1开始）
        total: 总物体数
        
    Returns:
        float: 归一化得分 (0-100)
    """
    # 使用倒数排名得分，排名第1得100分
    return 100 * (total - rank + 1) / total


def visualize_single_case(
    case_data: dict,
    output_path: str,
    max_display: int = 20,
    show_distant: bool = True,
    show_other_category: bool = True,
    dpi: int = 150
):
    """
    可视化单个排序 case (50个物体版本)
    
    新布局说明：
    - 第一行：Query + 精排后的20个物体（10个同cluster相似度最高 + 10个相邻cluster随机）
    - 第二行：较远cluster的20个物体（不参与精排，按余弦相似度排序）
    - 第三行：其他类别（非 Character）的9个物体（不参与精排）
    
    Args:
        case_data: 单个 case 的数据
        output_path: 输出图像路径
        max_display: 最大显示物体数量
        show_distant: 是否显示较远cluster的物体
        show_other_category: 是否显示其他类别的物体
        dpi: 输出图像 DPI
    """
    case_id = case_data.get('case_id', 'unknown')
    query_id = case_data.get('query_object_id', '')
    cluster_id = case_data.get('cluster_id', -1)
    category = case_data.get('category', 'Unknown')
    final_ranking = case_data.get('final_ranking', [])
    objects = case_data.get('objects', {})
    ranking_details = case_data.get('ranking_details', {})
    weighted_scores = case_data.get('weighted_scores', {})
    
    reranked_objects = ranking_details.get('reranked_objects', [])
    # 兼容新旧版本
    distant_ranking = ranking_details.get('distant_ranking', ranking_details.get('neighbor_ranking', []))
    other_category_objects = ranking_details.get('other_category_objects', [])
    
    # 计算显示的物体数量
    n_reranked = min(len(reranked_objects), max_display)
    n_distant = min(len(distant_ranking), max_display) if show_distant else 0
    n_other = min(len(other_category_objects), 10) if show_other_category else 0
    
    # 计算图像布局
    n_rows = 1
    if show_distant and n_distant > 0:
        n_rows = 2
    if show_other_category and n_other > 0:
        n_rows = 3
    
    n_cols = max(n_reranked, n_distant, n_other) + 1  # +1 for query/label
    
    # 创建图像
    fig_width = min(n_cols * 2.5, 45)
    fig_height = n_rows * 3.5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    
    # 确保 axes 是二维数组
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # 清除所有子图
    for ax_row in axes:
        for ax in ax_row:
            ax.axis('off')
    
    # 标题 - 显示是否有真实的加权分数
    has_weighted = bool(weighted_scores)
    score_info = "(Real Agent Scores)" if has_weighted else "(Rank-based Scores)"
    total_count = len(final_ranking)
    fig.suptitle(
        f"Case: {case_id} | Total Objects: {total_count}\nQuery: {query_id} | Cluster: {cluster_id} | Category: {category}\n{score_info}",
        fontsize=11, fontweight='bold', y=0.98
    )
    
    # 获取查询物体图像
    query_obj = objects.get(query_id, {})
    query_img_path = query_obj.get('image_path', '')
    query_img = load_image_safe(query_img_path)
    
    # 显示查询物体
    ax_query = axes[0, 0]
    if query_img is not None:
        ax_query.imshow(query_img)
    else:
        ax_query.text(0.5, 0.5, 'No Image', ha='center', va='center', fontsize=10)
        ax_query.set_xlim(0, 1)
        ax_query.set_ylim(0, 1)
    
    ax_query.set_title(f"Query\n{query_id[:20]}...", fontsize=8, color='blue', fontweight='bold')
    # 添加蓝色边框
    rect = Rectangle((0, 0), 1, 1, transform=ax_query.transAxes, 
                      fill=False, edgecolor='blue', linewidth=3)
    ax_query.add_patch(rect)
    
    # 显示 Reranked 物体（第一行）- 精排后的物体
    total_objects = len(final_ranking)
    for i, obj_id in enumerate(reranked_objects[:n_reranked]):
        ax = axes[0, i + 1]
        obj_info = objects.get(obj_id, {})
        img_path = obj_info.get('image_path', '')
        img = load_image_safe(img_path)
        
        # 计算排名和得分
        try:
            rank = final_ranking.index(obj_id) + 1
        except ValueError:
            rank = i + 2  # 如果不在 final_ranking 中，使用位置
        
        # 获取真实分数（如果有）
        score, score_type = get_object_score(obj_id, case_data, rank, total_objects)
        
        if img is not None:
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, 'No Image', ha='center', va='center', fontsize=8)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        
        # 标题显示排名和得分
        if score_type == "weighted":
            title = f"#{rank} | W:{score:.2f}\n{obj_id[:16]}..."
            title_color = 'darkgreen'
        else:
            title = f"#{rank} | R:{score:.1f}\n{obj_id[:16]}..."
            title_color = 'green'
        
        ax.set_title(title, fontsize=7, color=title_color)
        
        # 添加绿色边框（reranked）
        rect = Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                          fill=False, edgecolor='green', linewidth=2)
        ax.add_patch(rect)
    
    # 显示较远cluster物体（第二行，如果需要）
    if show_distant and n_distant > 0:
        # 第二行第一列为空或显示标签
        ax_label = axes[1, 0]
        ax_label.text(0.5, 0.5, 'Distant\nCluster', ha='center', va='center', 
                      fontsize=10, fontweight='bold', color='gray')
        ax_label.set_xlim(0, 1)
        ax_label.set_ylim(0, 1)
        
        for i, obj_id in enumerate(distant_ranking[:n_distant]):
            if i + 1 >= n_cols:
                break
            ax = axes[1, i + 1]
            obj_info = objects.get(obj_id, {})
            img_path = obj_info.get('image_path', '')
            img = load_image_safe(img_path)
            
            # 计算排名和得分
            try:
                rank = final_ranking.index(obj_id) + 1
            except ValueError:
                rank = len(reranked_objects) + i + 2
            
            # 获取真实分数（如果有）
            score, score_type = get_object_score(obj_id, case_data, rank, total_objects)
            
            if img is not None:
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, 'No Image', ha='center', va='center', fontsize=8)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
            
            # 标题显示排名和得分
            if score_type == "similarity":
                title = f"#{rank} | S:{score:.2f}\n{obj_id[:16]}..."
            else:
                title = f"#{rank} | R:{score:.1f}\n{obj_id[:16]}..."
            
            ax.set_title(title, fontsize=7, color='gray')
            
            # 添加灰色边框（distant，不参与精排）
            rect = Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                              fill=False, edgecolor='gray', linewidth=2)
            ax.add_patch(rect)
    
    # 显示其他类别物体（第三行，如果需要）
    if show_other_category and n_other > 0:
        row_idx = 2 if (show_distant and n_distant > 0) else 1
        
        # 第三行第一列显示标签
        ax_label = axes[row_idx, 0]
        ax_label.text(0.5, 0.5, 'Other\nCategories', ha='center', va='center', 
                      fontsize=10, fontweight='bold', color='purple')
        ax_label.set_xlim(0, 1)
        ax_label.set_ylim(0, 1)
        
        for i, obj_id in enumerate(other_category_objects[:n_other]):
            if i + 1 >= n_cols:
                break
            ax = axes[row_idx, i + 1]
            obj_info = objects.get(obj_id, {})
            img_path = obj_info.get('image_path', '')
            img = load_image_safe(img_path)
            
            # 计算排名
            try:
                rank = final_ranking.index(obj_id) + 1
            except ValueError:
                rank = len(reranked_objects) + len(distant_ranking) + i + 2
            
            # 获取类别信息
            other_info = case_data.get('other_category_info', {}).get(obj_id, {})
            obj_category = other_info.get('category', obj_info.get('category', 'Unknown'))
            
            if img is not None:
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, 'No Image', ha='center', va='center', fontsize=8)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
            
            # 标题显示排名和类别
            title = f"#{rank} | {obj_category[:8]}\n{obj_id[:16]}..."
            ax.set_title(title, fontsize=7, color='purple')
            
            # 添加紫色边框（其他类别，不参与精排）
            rect = Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                              fill=False, edgecolor='purple', linewidth=2)
            ax.add_patch(rect)
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"✅ 保存: {output_path}")


def visualize_ranking_grid(
    case_data: dict,
    output_path: str,
    grid_size: Tuple[int, int] = (5, 10),
    dpi: int = 200
):
    """
    以网格形式可视化排序结果（更紧凑的布局）
    显示真实的加权分数（如果有）或根据排名计算的分数
    
    Args:
        case_data: 单个 case 的数据
        output_path: 输出图像路径
        grid_size: 网格大小 (rows, cols)
        dpi: 输出图像 DPI
    """
    case_id = case_data.get('case_id', 'unknown')
    query_id = case_data.get('query_object_id', '')
    cluster_id = case_data.get('cluster_id', -1)
    category = case_data.get('category', 'Unknown')
    final_ranking = case_data.get('final_ranking', [])
    objects = case_data.get('objects', {})
    weighted_scores = case_data.get('weighted_scores', {})
    distant_similarity_scores = case_data.get('distant_similarity_scores', {})
    # 兼容旧版本
    neighbor_similarity_scores = case_data.get('neighbor_similarity_scores', {})
    
    # 判断是否有真实分数
    has_real_scores = bool(weighted_scores) or bool(distant_similarity_scores) or bool(neighbor_similarity_scores)
    
    n_rows, n_cols = grid_size
    max_objects = n_rows * n_cols - 1  # 减1给查询物体
    
    # 创建图像
    fig_width = n_cols * 2.5
    fig_height = n_rows * 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    
    # 标题
    score_info = "(Real Weighted Scores)" if has_real_scores else "(Rank-based Scores)"
    fig.suptitle(
        f"Ranking Results: {case_id}\nQuery: {query_id} | Cluster: {cluster_id} | Category: {category}\n{score_info}",
        fontsize=11, fontweight='bold', y=0.99
    )
    
    # 展平 axes
    axes_flat = axes.flatten()
    
    # 清除所有子图
    for ax in axes_flat:
        ax.axis('off')
    
    # 显示查询物体（第一个位置）
    query_obj = objects.get(query_id, {})
    query_img_path = query_obj.get('image_path', '')
    query_img = load_image_safe(query_img_path)
    
    ax_query = axes_flat[0]
    if query_img is not None:
        ax_query.imshow(query_img)
    else:
        ax_query.text(0.5, 0.5, 'No Image', ha='center', va='center', fontsize=10)
        ax_query.set_xlim(0, 1)
        ax_query.set_ylim(0, 1)
    
    ax_query.set_title(f"QUERY\n{query_id[:16]}...", fontsize=8, color='blue', fontweight='bold')
    rect = Rectangle((0, 0), 1, 1, transform=ax_query.transAxes,
                      fill=False, edgecolor='blue', linewidth=4)
    ax_query.add_patch(rect)
    
    # 显示排序结果
    total_objects = len(final_ranking)
    display_objects = [obj_id for obj_id in final_ranking if obj_id != query_id][:max_objects]
    
    for idx, obj_id in enumerate(display_objects):
        ax = axes_flat[idx + 1]
        obj_info = objects.get(obj_id, {})
        img_path = obj_info.get('image_path', '')
        img = load_image_safe(img_path)
        
        rank = idx + 1  # 排名从1开始（不含 query）
        
        # 获取真实分数（如果有）
        score, score_type = get_object_score(obj_id, case_data, rank, total_objects - 1)
        
        if img is not None:
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, 'No Image', ha='center', va='center', fontsize=8)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        
        # 根据分数类型和排名设置颜色
        if rank <= 3:
            color = 'red'
            lw = 3
        elif rank <= 10:
            color = 'orange'
            lw = 2
        else:
            color = 'gray'
            lw = 1
        
        # 标题显示排名和分数
        if score_type == "weighted":
            title = f"#{rank} | W:{score:.2f}"
        elif score_type == "similarity":
            title = f"#{rank} | S:{score:.2f}"
        else:
            title = f"#{rank} | R:{score:.1f}"
        
        ax.set_title(title, fontsize=8, color=color, fontweight='bold')
        rect = Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                          fill=False, edgecolor=color, linewidth=lw)
        ax.add_patch(rect)
    
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"✅ 保存网格视图: {output_path}")


def visualize_all_cases_summary(
    result_data: dict,
    output_path: str,
    dpi: int = 150
):
    """
    生成所有 case 的摘要统计图（包括失败 case 的统计）
    
    Args:
        result_data: 完整结果数据
        output_path: 输出路径
        dpi: DPI
    """
    metadata = result_data.get('metadata', {})
    cluster_stats = result_data.get('cluster_statistics', {})
    cases = result_data.get('cases', [])
    failed_cases = result_data.get('failed_cases', [])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 聚类大小分布
    ax1 = axes[0, 0]
    cluster_sizes = list(cluster_stats.get('cluster_sizes', {}).values())
    if cluster_sizes:
        ax1.hist(cluster_sizes, bins=20, color='steelblue', edgecolor='white', alpha=0.8)
        ax1.axvline(np.mean(cluster_sizes), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(cluster_sizes):.1f}')
        ax1.set_xlabel('Cluster Size')
        ax1.set_ylabel('Count')
        ax1.set_title('Cluster Size Distribution')
        ax1.legend()
    
    # 2. 每个 case 的物体数量
    ax2 = axes[0, 1]
    case_ids = [c.get('case_id', '')[:20] for c in cases]
    case_sizes = [len(c.get('final_ranking', [])) for c in cases]
    if case_sizes:
        colors = plt.cm.viridis(np.linspace(0, 1, len(case_sizes)))
        bars = ax2.bar(range(len(case_sizes)), case_sizes, color=colors)
        ax2.set_xlabel('Case Index')
        ax2.set_ylabel('Number of Objects')
        ax2.set_title('Objects per Case')
        ax2.set_xticks(range(len(case_sizes)))
        ax2.set_xticklabels([f'Case {i}' for i in range(len(case_sizes))], 
                            rotation=45, ha='right', fontsize=8)
    
    # 3. 类别分布
    ax3 = axes[1, 0]
    categories = [c.get('category', 'Unknown') for c in cases]
    category_counts = {}
    for cat in categories:
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    if category_counts:
        cats = list(category_counts.keys())
        counts = list(category_counts.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(cats)))
        ax3.pie(counts, labels=cats, colors=colors, autopct='%1.1f%%', startangle=90)
        ax3.set_title('Category Distribution')
    
    # 4. 元数据信息
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # 失败 case 统计信息
    failed_info = ""
    if failed_cases:
        failed_info = f"\n    \u274c Failed Cases: {len(failed_cases)}"
        for fc in failed_cases[:3]:  # 最多显示3个
            failed_info += f"\n      - Case {fc.get('case_index', 'N/A')}: {fc.get('error_message', 'Unknown')[:50]}..."
        if len(failed_cases) > 3:
            failed_info += f"\n      ... and {len(failed_cases) - 3} more"
    
    info_text = f"""
    📊 Ranking Results Summary
    {'='*40}
    
    Source File: {os.path.basename(metadata.get('source_file', 'Unknown'))}
    Generated Time: {metadata.get('generated_time', 'Unknown')}
    
    ✅ Successful Cases: {metadata.get('total_cases', len(cases))}
    ❌ Failed Cases: {metadata.get('failed_cases_count', len(failed_cases))}
    Total Clusters: {cluster_stats.get('total_clusters', 'N/A')}
    
    Ranking Structure:
      - Rerank Size: {metadata.get('ranking_structure', {}).get('rerank_size', 'N/A')}
      - Top Similar: {metadata.get('ranking_structure', {}).get('top_similar_count', 'N/A')}
      - Neighbor Random: {metadata.get('ranking_structure', {}).get('neighbor_random_count', 'N/A')}
      - Distant Size: {metadata.get('ranking_structure', {}).get('distant_size', metadata.get('ranking_structure', {}).get('neighbor_size', 'N/A'))}
      - Other Category Size: {metadata.get('ranking_structure', {}).get('other_category_size', 'N/A')}
      - Total per Case: {metadata.get('ranking_structure', {}).get('total_per_case', 'N/A')}
    
    Use Agent Ranking: {'✅ Yes' if metadata.get('use_agent_ranking', False) else '❌ No'}{failed_info}
    """
    ax4.text(0.1, 0.9, info_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"✅ 保存摘要图: {output_path}")


def visualize_rerank_comparison(
    case_data: dict,
    output_path: str,
    dpi: int = 200
):
    """
    可视化精排前后的对比图（仅针对参与精排的20个物体）
    
    布局说明（4x5排版）：
    - 第0行：Query + 精排前第1-4个物体（按余弦相似度排序）
    - 第1行：精排前第5-9个物体
    - 第2行：精排前第10-14个物体
    - 第3行：精排前第15-19个物体
    - 第4行：精排前第20个物体 + 分隔
    - 第5行：Query + 精排后第1-4个物体
    - 第6行：精排后第5-9个物体
    - 第7行：精排后第10-14个物体
    - 第8行：精排后第15-19个物体
    - 第9行：精排后第20个物体
    
    简化为：上半部分4行5列显示精排前，下半部分4行5列显示精排后
    每部分第一个位置显示Query
    
    Args:
        case_data: 单个 case 的数据
        output_path: 输出图像路径
        dpi: 输出图像 DPI
    """
    case_id = case_data.get('case_id', 'unknown')
    query_id = case_data.get('query_object_id', '')
    cluster_id = case_data.get('cluster_id', -1)
    category = case_data.get('category', 'Unknown')
    objects = case_data.get('objects', {})
    ranking_details = case_data.get('ranking_details', {})
    weighted_scores = case_data.get('weighted_scores', {})
    pre_rerank_cosine_scores = case_data.get('pre_rerank_cosine_scores', {})
    
    # 获取精排前后的排序
    pre_rerank_cosine_order = ranking_details.get('pre_rerank_cosine_order', [])
    reranked_objects = ranking_details.get('reranked_objects', [])
    
    # 如果没有精排前的数据，跳过
    if not pre_rerank_cosine_order or not reranked_objects:
        print(f"⚠️ Case {case_id} 没有精排对比数据，跳过")
        return
    
    n_objects = len(reranked_objects)
    if n_objects == 0:
        return
    
    # 4x5布局：每部分4行5列，上半部分精排前，下半部分精排后
    # 总共8行5列（加1行分隔 = 9行）
    n_cols = 5
    n_rows_per_section = 4  # 每部分4行
    n_rows = n_rows_per_section * 2 + 1  # 上半部分4行 + 分隔1行 + 下半部分4行 = 9行
    
    # 创建图像 - 紧凑布局适合论文
    fig_width = n_cols * 1.8  # 减小每列宽度
    fig_height = n_rows * 1.6  # 减小每行高度
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    
    # 调整子图间距 - 更紧凑
    plt.subplots_adjust(
        left=0.02, right=0.98,   # 左右边距
        top=0.93, bottom=0.02,   # 上下边距
        wspace=0.08,             # 列间距
        hspace=0.35              # 行间距（保留一些空间给标题）
    )
    
    # 标题
    fig.suptitle(
        f"Rerank Comparison: {case_id}\n"
        f"Query: {query_id} | Cluster: {cluster_id} | Category: {category}",
        fontsize=9, fontweight='bold', y=0.98
    )
    
    # 清除所有子图
    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            axes[row_idx, col_idx].axis('off')
    
    # 获取查询物体图像
    query_obj = objects.get(query_id, {})
    query_img_path = query_obj.get('image_path', '')
    query_img = load_image_safe(query_img_path)
    
    # ========== 上半部分：精排前（按余弦相似度排序）- 4行5列 ==========
    # 在第一个位置显示Query
    ax_query_pre = axes[0, 0]
    if query_img is not None:
        ax_query_pre.imshow(query_img)
    else:
        ax_query_pre.text(0.5, 0.5, 'No Image', ha='center', va='center', fontsize=10)
        ax_query_pre.set_xlim(0, 1)
        ax_query_pre.set_ylim(0, 1)
    ax_query_pre.set_title(f"QUERY\n(Before)", fontsize=7, color='blue', fontweight='bold')
    rect = Rectangle((0, 0), 1, 1, transform=ax_query_pre.transAxes,
                      fill=False, edgecolor='blue', linewidth=2)
    ax_query_pre.add_patch(rect)
    
    # 添加上半部分标签
    axes[0, 2].text(0.5, 1.3, "BEFORE RERANK (Cosine Similarity)", 
                    fontsize=8, fontweight='bold', color='darkorange',
                    ha='center', va='bottom', transform=axes[0, 2].transAxes)
    
    # 显示精排前的物体（按余弦相似度排序）- 从位置1开始，共19个位置（4行x5列-1个Query位置=19）
    for i, obj_id in enumerate(pre_rerank_cosine_order[:19]):
        # 计算位置：跳过第一个位置（Query）
        pos = i + 1  # 从1开始
        row = pos // n_cols
        col = pos % n_cols
        
        if row >= n_rows_per_section:
            break
        
        ax = axes[row, col]
        obj_info = objects.get(obj_id, {})
        img_path = obj_info.get('image_path', '')
        img = load_image_safe(img_path)
        
        # 获取余弦相似度分数
        cosine_score = pre_rerank_cosine_scores.get(obj_id, 0.0)
        
        if img is not None:
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, 'No Image', ha='center', va='center', fontsize=8)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        
        # 精排前的排名
        pre_rank = i + 1
        # 精排后的排名（用于显示变化）
        try:
            post_rank = reranked_objects.index(obj_id) + 1
            rank_change = pre_rank - post_rank  # 正值表示排名上升
            if rank_change > 0:
                change_str = f"↑{rank_change}"
                border_color = 'limegreen'
            elif rank_change < 0:
                change_str = f"↓{abs(rank_change)}"
                border_color = 'tomato'
            else:
                change_str = "="
                border_color = 'gray'
        except ValueError:
            change_str = "?"
            border_color = 'gray'
        
        title = f"#{pre_rank} Cos:{cosine_score:.2f}\n{change_str}→#{post_rank if change_str != '?' else '?'}"
        ax.set_title(title, fontsize=6, color='darkorange')
        
        # 根据排名变化设置边框颜色
        rect = Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                          fill=False, edgecolor=border_color, linewidth=1.5)
        ax.add_patch(rect)
    
    # 如果有第20个物体，显示在第4行最后
    if len(pre_rerank_cosine_order) >= 20:
        obj_id = pre_rerank_cosine_order[19]
        ax = axes[3, 4]  # 第4行第5列
        obj_info = objects.get(obj_id, {})
        img_path = obj_info.get('image_path', '')
        img = load_image_safe(img_path)
        cosine_score = pre_rerank_cosine_scores.get(obj_id, 0.0)
        
        if img is not None:
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, 'No Image', ha='center', va='center', fontsize=8)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        
        pre_rank = 20
        try:
            post_rank = reranked_objects.index(obj_id) + 1
            rank_change = pre_rank - post_rank
            if rank_change > 0:
                change_str = f"↑{rank_change}"
                border_color = 'limegreen'
            elif rank_change < 0:
                change_str = f"↓{abs(rank_change)}"
                border_color = 'tomato'
            else:
                change_str = "="
                border_color = 'gray'
        except ValueError:
            change_str = "?"
            border_color = 'gray'
            post_rank = "?"
        
        title = f"#{pre_rank} Cos:{cosine_score:.2f}\n{change_str}→#{post_rank}"
        ax.set_title(title, fontsize=6, color='darkorange')
        rect = Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                          fill=False, edgecolor=border_color, linewidth=1.5)
        ax.add_patch(rect)
    
    # ========== 分隔行（第5行，索引4）==========
    sep_row = n_rows_per_section  # 第5行（索引4）
    for col in range(n_cols):
        axes[sep_row, col].axhline(y=0.5, color='black', linewidth=1)
        axes[sep_row, col].set_xlim(0, 1)
        axes[sep_row, col].set_ylim(0, 1)
    axes[sep_row, 2].text(0.5, 0.5, "─── ↓ AFTER RERANK ↓ ───", 
                          fontsize=7, ha='center', va='center', color='gray')
    
    # ========== 下半部分：精排后（按 Agent 分数排序）- 4行5列 ==========
    post_start_row = n_rows_per_section + 1  # 第6行（索引5）
    
    # 在第一个位置显示Query
    ax_query_post = axes[post_start_row, 0]
    if query_img is not None:
        ax_query_post.imshow(query_img)
    else:
        ax_query_post.text(0.5, 0.5, 'No Image', ha='center', va='center', fontsize=10)
        ax_query_post.set_xlim(0, 1)
        ax_query_post.set_ylim(0, 1)
    ax_query_post.set_title(f"QUERY\n(After)", fontsize=7, color='blue', fontweight='bold')
    rect = Rectangle((0, 0), 1, 1, transform=ax_query_post.transAxes,
                      fill=False, edgecolor='blue', linewidth=2)
    ax_query_post.add_patch(rect)
    
    # 添加下半部分标签
    axes[post_start_row, 2].text(0.5, 1.3, "AFTER RERANK (Agent Scores)", 
                                  fontsize=8, fontweight='bold', color='darkgreen',
                                  ha='center', va='bottom', transform=axes[post_start_row, 2].transAxes)
    
    # 显示精排后的物体 - 从位置1开始
    for i, obj_id in enumerate(reranked_objects[:19]):
        # 计算位置：跳过第一个位置（Query）
        pos = i + 1
        row = post_start_row + pos // n_cols
        col = pos % n_cols
        
        if row >= n_rows:
            break
        
        ax = axes[row, col]
        obj_info = objects.get(obj_id, {})
        img_path = obj_info.get('image_path', '')
        img = load_image_safe(img_path)
        
        # 获取加权分数
        score_data = weighted_scores.get(obj_id, {})
        if isinstance(score_data, dict):
            weighted_score = score_data.get('total_score', 0.0)
        else:
            weighted_score = float(score_data) if score_data else 0.0
        
        # 获取余弦相似度分数（用于对比）
        cosine_score = pre_rerank_cosine_scores.get(obj_id, 0.0)
        
        if img is not None:
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, 'No Image', ha='center', va='center', fontsize=8)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        
        # 精排后的排名
        post_rank = i + 1
        # 精排前的排名
        try:
            pre_rank = pre_rerank_cosine_order.index(obj_id) + 1
            rank_change = pre_rank - post_rank  # 正值表示排名上升
            if rank_change > 0:
                change_str = f"↑{rank_change}"
                border_color = 'limegreen'
            elif rank_change < 0:
                change_str = f"↓{abs(rank_change)}"
                border_color = 'tomato'
            else:
                change_str = "="
                border_color = 'gray'
        except ValueError:
            change_str = "?"
            border_color = 'gray'
            pre_rank = "?"
        
        title = f"#{post_rank} W:{weighted_score:.1f}\n{change_str}(#{pre_rank})"
        ax.set_title(title, fontsize=6, color='darkgreen')
        
        # 根据排名变化设置边框颜色
        rect = Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                          fill=False, edgecolor=border_color, linewidth=1.5)
        ax.add_patch(rect)
    
    # 如果有第20个物体，显示在最后一行最后一列
    if len(reranked_objects) >= 20:
        obj_id = reranked_objects[19]
        ax = axes[n_rows - 1, 4]  # 最后一行第5列
        obj_info = objects.get(obj_id, {})
        img_path = obj_info.get('image_path', '')
        img = load_image_safe(img_path)
        
        score_data = weighted_scores.get(obj_id, {})
        if isinstance(score_data, dict):
            weighted_score = score_data.get('total_score', 0.0)
        else:
            weighted_score = float(score_data) if score_data else 0.0
        
        if img is not None:
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, 'No Image', ha='center', va='center', fontsize=8)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        
        post_rank = 20
        try:
            pre_rank = pre_rerank_cosine_order.index(obj_id) + 1
            rank_change = pre_rank - post_rank
            if rank_change > 0:
                change_str = f"↑{rank_change}"
                border_color = 'limegreen'
            elif rank_change < 0:
                change_str = f"↓{abs(rank_change)}"
                border_color = 'tomato'
            else:
                change_str = "="
                border_color = 'gray'
        except ValueError:
            change_str = "?"
            border_color = 'gray'
            pre_rank = "?"
        
        title = f"#{post_rank} W:{weighted_score:.1f}\n{change_str}(#{pre_rank})"
        ax.set_title(title, fontsize=6, color='darkgreen')
        rect = Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                          fill=False, edgecolor=border_color, linewidth=1.5)
        ax.add_patch(rect)
    
    # 保存图像（不使用 tight_layout，因为已经手动设置了 subplots_adjust）
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white', pad_inches=0.05)
    plt.close(fig)
    
    print(f"✅ 保存精排对比图: {output_path}")


def visualize_result_file(
    json_path: str,
    output_dir: str = None,
    max_cases: int = None,
    grid_mode: bool = True,
    dpi: int = 200
):
    """
    可视化单个结果文件（只渲染成功的 case）
    
    Args:
        json_path: JSON 文件路径
        output_dir: 输出目录（默认与 JSON 文件同目录）
        max_cases: 最大可视化 case 数量
        grid_mode: 是否使用网格模式
        dpi: 输出 DPI
    """
    print(f"\n{'='*60}")
    print(f"📂 加载结果文件: {json_path}")
    print(f"{'='*60}")
    
    # 加载数据
    result_data = load_ranking_result(json_path)
    
    # 设置输出目录
    if output_dir is None:
        output_dir = os.path.dirname(json_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取文件名前缀
    json_basename = os.path.splitext(os.path.basename(json_path))[0]
    
    # 打印基本信息
    metadata = result_data.get('metadata', {})
    cases = result_data.get('cases', [])
    failed_cases = result_data.get('failed_cases', [])
    
    print(f"📊 元数据:")
    print(f"   - 源文件: {metadata.get('source_file', 'Unknown')}")
    print(f"   - 生成时间: {metadata.get('generated_time', 'Unknown')}")
    print(f"   - 成功 Case 数量: {len(cases)}")
    if failed_cases:
        print(f"   - 失败 Case 数量: {len(failed_cases)}")
    
    # 生成摘要图
    summary_path = os.path.join(output_dir, f"{json_basename}_summary.png")
    visualize_all_cases_summary(result_data, summary_path, dpi=dpi)
    
    # 可视化每个 case（只渲染成功的 case）
    if max_cases is not None:
        cases = cases[:max_cases]
    
    # 过滤掉无效的 case（没有 final_ranking 或 final_ranking 为空）
    valid_cases = []
    for case_data in cases:
        final_ranking = case_data.get('final_ranking', [])
        if final_ranking and len(final_ranking) > 0:
            valid_cases.append(case_data)
        else:
            case_id = case_data.get('case_id', 'unknown')
            print(f"⚠️ 跳过无效 case: {case_id}（final_ranking 为空）")
    
    print(f"\n🎨 开始可视化 {len(valid_cases)} 个有效 case...")
    if len(valid_cases) < len(cases):
        print(f"   （跳过了 {len(cases) - len(valid_cases)} 个无效 case）")
    
    for idx, case_data in enumerate(valid_cases):
        case_id = case_data.get('case_id', f'case_{idx}')
        # 简化 case_id 用于文件名
        case_id_short = case_id.replace('case_', '')[:20]
        
        if grid_mode:
            # 网格模式
            output_path = os.path.join(output_dir, f"{json_basename}_{case_id_short}_grid.png")
            visualize_ranking_grid(case_data, output_path, grid_size=(5, 10), dpi=dpi)
        else:
            # 详细模式
            output_path = os.path.join(output_dir, f"{json_basename}_{case_id_short}_detail.png")
            visualize_single_case(case_data, output_path, max_display=20, dpi=dpi)
        
        # 精排对比图（单独渲染）
        rerank_comparison_path = os.path.join(output_dir, f"{json_basename}_{case_id_short}_rerank_comparison.png")
        visualize_rerank_comparison(case_data, rerank_comparison_path, dpi=dpi)
    
    print(f"\n✅ 可视化完成！输出目录: {output_dir}")
    return output_dir


def visualize_directory(
    input_dir: str,
    output_dir: str = None,
    max_cases_per_file: int = None,
    grid_mode: bool = True,
    dpi: int = 200
):
    """
    可视化目录下的所有结果文件
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        max_cases_per_file: 每个文件最大可视化 case 数量
        grid_mode: 是否使用网格模式
        dpi: 输出 DPI
    """
    print(f"\n{'='*70}")
    print(f"📂 批量可视化目录: {input_dir}")
    print(f"{'='*70}")
    
    # 查找所有 ranking_cases*.json 文件
    json_files = glob.glob(os.path.join(input_dir, "ranking_cases*.json"))
    
    if not json_files:
        print(f"❌ 未找到 ranking_cases*.json 文件")
        return
    
    print(f"✅ 找到 {len(json_files)} 个结果文件")
    
    # 设置输出目录
    if output_dir is None:
        output_dir = os.path.join(input_dir, "visualizations")
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理每个文件
    for i, json_path in enumerate(json_files, 1):
        print(f"\n📄 [{i}/{len(json_files)}] {os.path.basename(json_path)}")
        try:
            visualize_result_file(
                json_path=json_path,
                output_dir=output_dir,
                max_cases=max_cases_per_file,
                grid_mode=grid_mode,
                dpi=dpi
            )
        except Exception as e:
            print(f"❌ 处理失败: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*70}")
    print(f"✅ 批量可视化完成！")
    print(f"   输出目录: {output_dir}")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description='可视化排序结果')
    parser.add_argument('--input', '-i', type=str, help='输入 JSON 文件路径')
    parser.add_argument('--input_dir', '-d', type=str, help='输入目录（批量处理）')
    parser.add_argument('--output_dir', '-o', type=str, help='输出目录')
    parser.add_argument('--max_cases', '-n', type=int, default=None, help='每个文件最大可视化 case 数量')
    parser.add_argument('--detail_mode', action='store_true', help='使用详细模式（默认网格模式）')
    parser.add_argument('--dpi', type=int, default=400, help='输出图像 DPI')
    
    args = parser.parse_args()
    
    if args.input:
        # 单文件模式
        visualize_result_file(
            json_path=args.input,
            output_dir=args.output_dir,
            max_cases=args.max_cases,
            grid_mode=not args.detail_mode,
            dpi=args.dpi
        )
    elif args.input_dir:
        # 目录模式
        visualize_directory(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            max_cases_per_file=args.max_cases,
            grid_mode=not args.detail_mode,
            dpi=args.dpi
        )
    else:
        # 默认使用预设路径
        default_input_dir = "/apdcephfs/share_303565425/DCC3/serenasnliu/GAR/Uni3D/data/test_datasets/objaverse/openshape_clustering_output"
        
        print("未指定输入文件，使用默认目录...")
        if os.path.exists(default_input_dir):
            visualize_directory(
                input_dir=default_input_dir,
                output_dir=args.output_dir,
                max_cases_per_file=args.max_cases,
                grid_mode=not args.detail_mode,
                dpi=args.dpi
            )
        else:
            parser.print_help()


if __name__ == "__main__":
    main()
