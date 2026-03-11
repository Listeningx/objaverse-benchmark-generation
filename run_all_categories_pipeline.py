#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量运行多个类别的 OpenShape 聚类精排 Pipeline

用法:
    python run_all_categories_pipeline.py [--num_cases 10] [--llm_mode qwen] [--no_agent] [--seed 20260225]

支持的类别: Object, Building, Weapon, Vehicle, Animal (以及已有的 Character)

流程:
1. 从 categorized_objaverse_golden.json 中为每个类别生成对应的分组文件
2. 对每个类别分别运行 OpenShape 聚类精排 Pipeline
3. 每个类别使用独立的缓存目录和输出目录
"""

import os
import sys
import json
import argparse
from datetime import datetime

# 基础路径配置
BASE_DATA_DIR = "/apdcephfs/share_303565425/DCC3/serenasnliu/GAR/Uni3D/data/test_datasets/objaverse"
ALL_GROUPS_JSON = os.path.join(BASE_DATA_DIR, "objaverse_golden_all_groups.json")
CATEGORIZED_JSON = os.path.join(BASE_DATA_DIR, "categorized_objaverse_golden.json")

# Pipeline 脚本路径
PIPELINE_DIR = "/apdcephfs/share_303565425/DCC3/serenasnliu/GAR/Uni3D/ranking_agent/unified-embeddings/agent_skills_for_bcmk"

# 要处理的类别列表（物体数量 >= 30 的类别才有意义，因为聚类要求最小30个物体）
TARGET_CATEGORIES = ["Object", "Building", "Weapon", "Vehicle", "Animal"]

# Per-category default case counts (proportional to Character=150)
DEFAULT_CATEGORY_NUM_CASES = {
    # "Object": 196,
    "Object": 50,
    "Building": 41,
    "Weapon": 34,
    "Vehicle": 27,
    "Animal": 13,
}

# 最终输出基础目录
FINAL_OUTPUT_BASE = os.path.join(BASE_DATA_DIR, "openshape_ranking_results_qwen3.5max")
INTERMEDIATE_CACHE_BASE = os.path.join(BASE_DATA_DIR, "openshape_intermediate_cache")


def generate_category_groups(category: str) -> str:
    """
    从 all_groups.json 中筛选特定类别的分组文件
    
    Args:
        category: 类别名称
        
    Returns:
        生成的分组文件路径
    """
    category_lower = category.lower()
    output_path = os.path.join(BASE_DATA_DIR, f"objaverse_golden_{category_lower}_groups.json")
    
    # 如果文件已存在，跳过生成
    if os.path.exists(output_path):
        print(f"  📁 分组文件已存在: {output_path}")
        return output_path
    
    print(f"  📝 生成 {category} 类别分组文件...")
    
    with open(ALL_GROUPS_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    original_groups = data.get('groups', [])
    filtered_groups = [g for g in original_groups if g.get('category') == category]
    
    if len(filtered_groups) == 0:
        print(f"  ⚠️ 类别 {category} 没有找到任何组，跳过")
        return None
    
    new_data = {
        "metadata": {
            "group_size": 10,
            "total_groups": len(filtered_groups),
            "total_categories": 1,
            "categories_with_groups": 1,
            "source_file": ALL_GROUPS_JSON,
            "filtered_category": category
        },
        "category_statistics": {
            category: data.get('category_statistics', {}).get(category, {})
        },
        "groups": filtered_groups
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)
    
    print(f"  ✅ 已生成: {output_path} ({len(filtered_groups)} 组)")
    return output_path


def run_pipeline_for_category(
    category: str,
    num_cases: int = 10,
    llm_mode: str = "qwen",
    model_name: str = None,
    use_agent_ranking: bool = True,
    force_embeddings: bool = False,
    force_clusters: bool = False,
    num_clusters: int = None,
    seed: int = 20260228
):
    """
    对单个类别运行完整的 Pipeline
    
    Args:
        category: 类别名称
        num_cases: 每个类别生成的 case 数量
        llm_mode: LLM 模式
        model_name: 模型名称
        use_agent_ranking: 是否使用 agent 精排
        force_embeddings: 强制重新计算 embeddings
        force_clusters: 强制重新聚类
        num_clusters: 目标聚类数
        seed: 随机种子
    """
    category_lower = category.lower()
    
    print(f"\n{'='*70}")
    print(f"🚀 开始处理类别: {category}")
    print(f"{'='*70}")
    
    # 1. 生成分组文件
    groups_file = generate_category_groups(category)
    if groups_file is None:
        print(f"❌ 跳过类别 {category}")
        return
    
    # 2. 设置各类别独立的目录
    cache_dir = os.path.join(BASE_DATA_DIR, f"openshape_cache_{category_lower}")
    output_dir = os.path.join(FINAL_OUTPUT_BASE, category_lower)
    intermediate_cache_dir = os.path.join(INTERMEDIATE_CACHE_BASE, category_lower)
    
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(intermediate_cache_dir, exist_ok=True)
    
    # 3. 导入并运行 Pipeline
    sys.path.insert(0, PIPELINE_DIR)
    from openshape_clustering_pipeline import OpenShapeClusteringPipeline
    
    pipeline = OpenShapeClusteringPipeline(
        input_json=groups_file,
        output_dir=output_dir,
        cache_dir=cache_dir,
        intermediate_cache_dir=intermediate_cache_dir,
        use_agent_ranking=use_agent_ranking,
        llm_mode=llm_mode,
        model_name=model_name,
        categorized_json_path=CATEGORIZED_JSON,
        target_category=category
    )
    
    output_path = pipeline.run(
        num_cases=num_cases,
        force_recompute_embeddings=force_embeddings,
        force_recompute_clusters=force_clusters,
        target_num_clusters=num_clusters,
        random_seed=seed
    )
    
    print(f"\n✅ 类别 {category} 处理完成，结果: {output_path}")
    return output_path


def parse_category_num_cases(pairs: list) -> dict:
    """
    Parse category=num_cases pairs from command line arguments.
    
    Args:
        pairs: list of strings like ["Object=50", "Animal=20"]
        
    Returns:
        dict mapping category name to num_cases
    """
    result = {}
    if pairs is None:
        return result
    for pair in pairs:
        if '=' not in pair:
            print(f"⚠️ Invalid format '{pair}', expected 'Category=num_cases', skipping")
            continue
        key, value = pair.split('=', 1)
        try:
            result[key] = int(value)
        except ValueError:
            print(f"⚠️ Invalid num_cases value '{value}' for category '{key}', skipping")
    return result


def main():
    parser = argparse.ArgumentParser(description='批量运行多个类别的 OpenShape 聚类精排 Pipeline')
    
    parser.add_argument('--categories', type=str, nargs='+', default=None,
                        help=f'要处理的类别列表（默认: {TARGET_CATEGORIES}）')
    parser.add_argument('--num_cases', type=int, default=None,
                        help='统一指定每个类别生成的 case 数量（覆盖默认值，优先级低于 --category_num_cases）')
    parser.add_argument('--category_num_cases', type=str, nargs='+', default=None,
                        help='为每个类别单独指定 case 数量，格式: Category=num，'
                             '例如: --category_num_cases Object=50 Animal=20（优先级最高）')
    parser.add_argument('--llm_mode', type=str, default='qwen',
                        choices=['api', 'qwen', 'mock'],
                        help='LLM 模式（默认: qwen）')
    parser.add_argument('--no_agent', action='store_true',
                        help='不使用 agent 精排，仅使用余弦相似度')
    parser.add_argument('--force_embeddings', action='store_true',
                        help='强制重新计算 embeddings')
    parser.add_argument('--force_clusters', action='store_true',
                        help='强制重新聚类')
    parser.add_argument('--num_clusters', type=int, default=None,
                        help='目标聚类数（默认自动计算）')
    parser.add_argument('--seed', type=int, default=20260301,
                        help='随机种子（默认: 20260225）')
    
    args = parser.parse_args()
    
    categories = args.categories or TARGET_CATEGORIES
    
    # Build per-category num_cases mapping
    # Priority: --category_num_cases > --num_cases > DEFAULT_CATEGORY_NUM_CASES
    category_num_cases_override = parse_category_num_cases(args.category_num_cases)
    
    category_num_cases = {}
    for cat in categories:
        if cat in category_num_cases_override:
            category_num_cases[cat] = category_num_cases_override[cat]
        elif args.num_cases is not None:
            category_num_cases[cat] = args.num_cases
        else:
            category_num_cases[cat] = DEFAULT_CATEGORY_NUM_CASES.get(cat, 10)
    
    # 验证类别有效性
    with open(CATEGORIZED_JSON, 'r', encoding='utf-8') as f:
        all_categories = list(json.load(f).keys())
    
    for cat in categories:
        if cat not in all_categories:
            print(f"⚠️ 未知类别: {cat}，可选类别: {all_categories}")
            categories = [c for c in categories if c in all_categories]
    
    print(f"📋 将处理以下类别: {categories}")
    print(f"📊 各类别生成 case 数量:")
    for cat in categories:
        print(f"   - {cat}: {category_num_cases.get(cat, 10)} cases")
    print(f"🤖 Agent 精排: {'开启' if not args.no_agent else '关闭'}")
    print(f"🔧 LLM 模式: {args.llm_mode}")
    
    # 记录结果
    results = {}
    start_time = datetime.now()
    
    for category in categories:
        cat_start = datetime.now()
        cat_num_cases = category_num_cases.get(category, 10)
        try:
            output_path = run_pipeline_for_category(
                category=category,
                num_cases=cat_num_cases,
                llm_mode=args.llm_mode,
                use_agent_ranking=not args.no_agent,
                force_embeddings=args.force_embeddings,
                force_clusters=args.force_clusters,
                num_clusters=args.num_clusters,
                seed=args.seed
            )
            cat_duration = (datetime.now() - cat_start).total_seconds()
            results[category] = {"status": "success", "output": output_path, "duration_seconds": cat_duration}
        except Exception as e:
            import traceback
            cat_duration = (datetime.now() - cat_start).total_seconds()
            results[category] = {"status": "failed", "error": str(e), "duration_seconds": cat_duration}
            print(f"❌ 类别 {category} 处理失败: {e}")
            traceback.print_exc()
    
    # 汇总
    total_duration = (datetime.now() - start_time).total_seconds()
    
    print(f"\n{'='*70}")
    print(f"📊 批量处理汇总")
    print(f"{'='*70}")
    print(f"总耗时: {total_duration:.1f} 秒")
    
    for category, result in results.items():
        status = "✅" if result["status"] == "success" else "❌"
        duration = result["duration_seconds"]
        print(f"  {status} {category}: {result['status']} ({duration:.1f}s)")
        if result["status"] == "success":
            print(f"     输出: {result['output']}")
        else:
            print(f"     错误: {result['error']}")
    
    # 保存汇总信息
    summary_path = os.path.join(FINAL_OUTPUT_BASE, "batch_run_summary.json")
    os.makedirs(FINAL_OUTPUT_BASE, exist_ok=True)
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            "run_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_duration_seconds": total_duration,
            "categories": categories,
            "category_num_cases": category_num_cases,
            "use_agent_ranking": not args.no_agent,
            "llm_mode": args.llm_mode,
            "results": results
        }, f, ensure_ascii=False, indent=2)
    print(f"\n📝 汇总信息已保存到: {summary_path}")


if __name__ == '__main__':
    main()
