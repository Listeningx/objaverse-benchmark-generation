#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运行 OpenShape 聚类精排 Pipeline 的便捷脚本

使用方法：
1. 仅生成 embeddings（不聚类不精排）:
   python run_openshape_clustering.py --step embeddings

2. 生成 embeddings + 聚类（不精排）:
   python run_openshape_clustering.py --step cluster

3. 完整流程（embeddings + 聚类 + 精排 case 生成）:
   python run_openshape_clustering.py --step full --num_cases 100

4. 使用缓存的 embeddings 重新聚类:
   python run_openshape_clustering.py --step cluster --force_clusters

5. 不使用 agent 精排（仅余弦相似度）:
   python run_openshape_clustering.py --step full --no_agent --num_cases 100
"""

import os
import sys
import argparse
from datetime import datetime

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ==================== 配置 ====================

# 默认输入文件
DEFAULT_INPUT = "/apdcephfs/share_303565425/DCC3/serenasnliu/GAR/Uni3D/data/test_datasets/objaverse/objaverse_golden_character_groups.json"

# 默认输出目录
DEFAULT_OUTPUT_DIR = "/apdcephfs/share_303565425/DCC3/serenasnliu/GAR/Uni3D/data/test_datasets/objaverse/openshape_clustering_output"

# 默认缓存目录（embedding 和聚类结果保存位置）
DEFAULT_CACHE_DIR = "/apdcephfs/share_303565425/DCC3/serenasnliu/GAR/Uni3D/data/test_datasets/objaverse/openshape_cache"


def run_embeddings_only(args):
    """
    仅生成 embeddings
    """
    from openshape_clustering_pipeline import OpenShapeClusteringPipeline
    
    print("\n" + "=" * 70)
    print("🔧 步骤 1: 生成 OpenShape Embeddings")
    print("=" * 70)
    
    pipeline = OpenShapeClusteringPipeline(
        input_json=args.input,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        use_agent_ranking=False
    )
    
    # 加载数据
    pipeline.load_input_data()
    
    # 提取和编码
    pipeline.extract_and_encode(force_recompute=args.force_embeddings)
    
    print("\n✅ Embeddings 生成完成！")
    print(f"   保存位置: {args.cache_dir}/openshape_embeddings.npz")


def run_clustering_only(args):
    """
    生成 embeddings + 聚类
    """
    from openshape_clustering_pipeline import OpenShapeClusteringPipeline
    
    print("\n" + "=" * 70)
    print("🔧 步骤 1-2: 生成 Embeddings + 聚类")
    print("=" * 70)
    
    pipeline = OpenShapeClusteringPipeline(
        input_json=args.input,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        use_agent_ranking=False
    )
    
    # 加载数据
    pipeline.load_input_data()
    
    # 提取和编码
    pipeline.extract_and_encode(force_recompute=args.force_embeddings)
    
    # 聚类
    pipeline.perform_clustering(
        force_recompute=args.force_clusters,
        target_num_clusters=args.num_clusters
    )
    
    print("\n✅ Embeddings 和聚类完成！")
    print(f"   Embeddings: {args.cache_dir}/openshape_embeddings.npz")
    print(f"   聚类结果: {args.cache_dir}/clusters.pkl")


def run_full_pipeline(args):
    """
    运行完整 pipeline
    """
    from openshape_clustering_pipeline import OpenShapeClusteringPipeline
    
    print("\n" + "=" * 70)
    print("🚀 运行完整 Pipeline")
    print("=" * 70)
    
    pipeline = OpenShapeClusteringPipeline(
        input_json=args.input,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        use_agent_ranking=not args.no_agent,
        llm_mode=args.llm_mode
    )
    
    output_path = pipeline.run(
        num_cases=args.num_cases,
        force_recompute_embeddings=args.force_embeddings,
        force_recompute_clusters=args.force_clusters,
        target_num_clusters=args.num_clusters,
        random_seed=args.seed
    )
    
    print(f"\n✅ 完整 Pipeline 完成！")
    print(f"   输出文件: {output_path}")


def show_cache_status(args):
    """
    显示缓存状态
    """
    import json
    import numpy as np
    
    print("\n" + "=" * 70)
    print("📊 缓存状态")
    print("=" * 70)
    
    cache_dir = args.cache_dir
    
    # 检查 embeddings
    emb_file = os.path.join(cache_dir, "openshape_embeddings.npz")
    emb_meta_file = os.path.join(cache_dir, "openshape_metadata.json")
    
    if os.path.exists(emb_file):
        data = np.load(emb_file)
        num_emb = len(data['object_ids'])
        emb_dim = data['embeddings'].shape[1] if len(data['embeddings'].shape) > 1 else 0
        
        print(f"\n✅ Embeddings 缓存:")
        print(f"   文件: {emb_file}")
        print(f"   物体数: {num_emb}")
        print(f"   Embedding 维度: {emb_dim}")
        
        if os.path.exists(emb_meta_file):
            with open(emb_meta_file, 'r') as f:
                meta = json.load(f)
            print(f"   保存时间: {meta.get('saved_time', 'N/A')}")
    else:
        print(f"\n❌ 未找到 Embeddings 缓存")
    
    # 检查聚类结果
    cluster_file = os.path.join(cache_dir, "clusters.pkl")
    cluster_info_file = os.path.join(cache_dir, "cluster_info.json")
    
    if os.path.exists(cluster_info_file):
        with open(cluster_info_file, 'r') as f:
            info = json.load(f)
        
        print(f"\n✅ 聚类缓存:")
        print(f"   文件: {cluster_file}")
        print(f"   聚类数: {info.get('total_clusters', 'N/A')}")
        print(f"   物体数: {info.get('total_objects', 'N/A')}")
        print(f"   保存时间: {info.get('saved_time', 'N/A')}")
        
        # 显示聚类大小分布
        if 'cluster_sizes' in info:
            sizes = list(info['cluster_sizes'].values())
            print(f"   聚类大小范围: {min(sizes)} - {max(sizes)}")
            print(f"   平均大小: {sum(sizes) / len(sizes):.1f}")
    else:
        print(f"\n❌ 未找到聚类缓存")


def main():
    parser = argparse.ArgumentParser(
        description='OpenShape 聚类精排 Pipeline 运行脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # 基本参数
    parser.add_argument('--step', type=str, default='full',
                        choices=['embeddings', 'cluster', 'full', 'status'],
                        help='运行步骤: embeddings(仅embedding), cluster(embedding+聚类), full(完整), status(查看缓存)')
    
    parser.add_argument('--input', type=str, default=DEFAULT_INPUT,
                        help=f'输入 JSON 文件（默认: {DEFAULT_INPUT}）')
    
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f'输出目录（默认: {DEFAULT_OUTPUT_DIR}）')
    
    parser.add_argument('--cache_dir', type=str, default=DEFAULT_CACHE_DIR,
                        help=f'缓存目录（默认: {DEFAULT_CACHE_DIR}）')
    
    # 生成参数
    parser.add_argument('--num_cases', type=int, default=10,
                        help='生成的 case 数量（默认: 10）')
    
    parser.add_argument('--num_clusters', type=int, default=None,
                        help='目标聚类数（默认: 自动计算）')
    
    # 精排参数
    parser.add_argument('--no_agent', action='store_true',
                        help='不使用 agent 精排，仅使用余弦相似度')
    
    parser.add_argument('--llm_mode', type=str, default='api',
                        choices=['api', 'qwen', 'mock'],
                        help='LLM 模式（默认: api）')
    
    # 缓存控制
    parser.add_argument('--force_embeddings', action='store_true',
                        help='强制重新计算 embeddings（忽略缓存）')
    
    parser.add_argument('--force_clusters', action='store_true',
                        help='强制重新聚类（忽略缓存）')
    
    # 其他
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子（默认: 42）')
    
    args = parser.parse_args()
    
    # 创建目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    
    # 打印配置
    print("\n" + "=" * 70)
    print("⚙️  配置信息")
    print("=" * 70)
    print(f"运行步骤: {args.step}")
    print(f"输入文件: {args.input}")
    print(f"输出目录: {args.output_dir}")
    print(f"缓存目录: {args.cache_dir}")
    
    if args.step in ['full']:
        print(f"生成 case 数: {args.num_cases}")
        print(f"使用 Agent 精排: {not args.no_agent}")
        print(f"LLM 模式: {args.llm_mode}")
    
    print("=" * 70)
    
    # 运行对应步骤
    if args.step == 'status':
        show_cache_status(args)
    elif args.step == 'embeddings':
        run_embeddings_only(args)
    elif args.step == 'cluster':
        run_clustering_only(args)
    elif args.step == 'full':
        run_full_pipeline(args)
    else:
        print(f"❌ 未知步骤: {args.step}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
