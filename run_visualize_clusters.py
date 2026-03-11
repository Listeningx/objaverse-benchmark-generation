#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
便捷运行脚本：可视化 OpenShape 聚类结果

使用方法:
    python run_visualize_clusters.py
"""

import os
import sys

# 添加当前目录到 Python 路径
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from visualize_clusters import (
    load_clusters,
    load_embeddings,
    load_input_json,
    print_cluster_summary,
    plot_cluster_size_distribution,
    plot_cluster_statistics,
    plot_cluster_2d_visualization,
    plot_cluster_neighbor_heatmap,
    visualize_cluster_images
)


# ==================== 配置 ====================

# 聚类文件路径
CLUSTERS_FILE = "/apdcephfs/share_303565425/DCC3/serenasnliu/GAR/Uni3D/data/test_datasets/objaverse/openshape_cache/clusters.pkl"

# Embeddings 文件路径
EMBEDDINGS_FILE = "/apdcephfs/share_303565425/DCC3/serenasnliu/GAR/Uni3D/data/test_datasets/objaverse/openshape_cache/openshape_embeddings.npz"

# 输入 JSON 文件（用于获取图像路径）
INPUT_JSON = "/apdcephfs/share_303565425/DCC3/serenasnliu/GAR/Uni3D/data/test_datasets/objaverse/objaverse_golden_character_groups.json"

# 输出目录
OUTPUT_DIR = "/apdcephfs/share_303565425/DCC3/serenasnliu/GAR/Uni3D/data/test_datasets/objaverse/openshape_cache/visualizations"


# ==================== 主函数 ====================

def main():
    print("\n" + "=" * 70)
    print("🎨 OpenShape 聚类结果可视化")
    print("=" * 70)
    print(f"聚类文件: {CLUSTERS_FILE}")
    print(f"Embeddings 文件: {EMBEDDINGS_FILE}")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 70)
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. 加载聚类数据
    if not os.path.exists(CLUSTERS_FILE):
        print(f"❌ 聚类文件不存在: {CLUSTERS_FILE}")
        return
    
    clusters, object_cluster_map = load_clusters(CLUSTERS_FILE)
    
    # 打印摘要
    print_cluster_summary(clusters)
    
    # 2. 加载 embeddings（可选）
    object_ids = None
    embeddings = None
    if os.path.exists(EMBEDDINGS_FILE):
        object_ids, embeddings = load_embeddings(EMBEDDINGS_FILE)
    else:
        print(f"⚠️ Embeddings 文件不存在，跳过物体级别的2D可视化")
    
    # 3. 加载物体信息（可选）
    objects_dict = None
    if os.path.exists(INPUT_JSON):
        objects_dict = load_input_json(INPUT_JSON)
    else:
        print(f"⚠️ 输入 JSON 文件不存在，跳过图像可视化")
    
    # ==================== 可视化 ====================
    
    # 4. 绘制聚类大小分布
    print("\n📈 绘制聚类大小分布...")
    plot_cluster_size_distribution(
        clusters,
        save_path=os.path.join(OUTPUT_DIR, 'cluster_size_distribution.png')
    )
    
    # 5. 绘制统计信息
    print("\n📊 绘制统计信息...")
    plot_cluster_statistics(
        clusters,
        save_path=os.path.join(OUTPUT_DIR, 'cluster_statistics.png')
    )
    
    # 6. 绘制 PCA 2D可视化
    print("\n🗺️ 绘制 PCA 2D可视化...")
    plot_cluster_2d_visualization(
        clusters,
        object_ids=object_ids,
        embeddings=embeddings,
        method='pca',
        save_path=os.path.join(OUTPUT_DIR, 'cluster_2d_pca.png'),
        dpi=300,
        enhanced_separation=True
    )
    
    # 7. 绘制 t-SNE 2D可视化（高分辨率版本，可能较慢）
    print("\n🗺️ 绘制 t-SNE 2D可视化（高分辨率，增强聚类分离）...")
    plot_cluster_2d_visualization(
        clusters,
        object_ids=object_ids,
        embeddings=embeddings,
        method='tsne',
        save_path=os.path.join(OUTPUT_DIR, 'cluster_2d_tsne.png'),
        figsize=(28, 14),  # 更大的图像
        dpi=300,  # 高分辨率
        max_points=10000,  # 更多的点
        enhanced_separation=True  # 增强聚类分离
    )
    
    # 8. 绘制邻居关系热力图
    print("\n🔥 绘制邻居关系热力图...")
    plot_cluster_neighbor_heatmap(
        clusters,
        save_path=os.path.join(OUTPUT_DIR, 'cluster_neighbor_heatmap.png')
    )
    
    # 9. 可视化几个最大聚类的图像
    if objects_dict is not None:
        print("\n🖼️ 可视化前3个最大聚类的图像...")
        sorted_clusters = sorted(clusters.items(), key=lambda x: x[1].size, reverse=True)
        for i, (cid, cluster) in enumerate(sorted_clusters[:3]):
            visualize_cluster_images(
                clusters,
                objects_dict,
                cid,
                save_path=os.path.join(OUTPUT_DIR, f'cluster_{cid}_images.png'),
                max_images=20,
                images_per_row=5
            )
    
    # ==================== 完成 ====================
    
    print("\n" + "=" * 70)
    print("✅ 可视化完成！")
    print(f"📁 输出目录: {OUTPUT_DIR}")
    print("\n生成的文件:")
    for f in os.listdir(OUTPUT_DIR):
        if f.endswith('.png'):
            print(f"  📊 {f}")
    print("=" * 70)


if __name__ == '__main__':
    main()
