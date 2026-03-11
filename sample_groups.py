#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从相似度组文件中随机抽取指定数量的组。

输入: 包含所有可选组的JSON文件
输出: 随机抽取的指定数量组的JSON文件
"""

import json
import os
import random
import argparse
from typing import Dict, List, Any, Optional


def sample_groups(
    groups_data: Dict[str, Any],
    num_samples: int,
    seed: Optional[int] = None,
    category_filter: Optional[List[str]] = None,
    balanced: bool = False
) -> Dict[str, Any]:
    """
    从组数据中随机抽取指定数量的组
    
    Args:
        groups_data: 包含所有组的字典
        num_samples: 要抽取的组数
        seed: 随机种子（可选）
        category_filter: 只从指定类别中抽取（可选）
        balanced: 是否在各类别间均衡抽取（可选）
    
    Returns:
        抽取结果的字典
    """
    if seed is not None:
        random.seed(seed)
    
    all_groups = groups_data['groups']
    
    # 如果指定了类别过滤
    if category_filter:
        all_groups = [g for g in all_groups if g['category'] in category_filter]
    
    if len(all_groups) == 0:
        raise ValueError("过滤后没有可用的组")
    
    if num_samples > len(all_groups):
        print(f"警告: 请求的组数 ({num_samples}) 大于可用组数 ({len(all_groups)})，将返回所有可用组")
        num_samples = len(all_groups)
    
    if balanced:
        # 均衡抽取：尽量从每个类别中均匀抽取
        category_groups = {}
        for g in all_groups:
            cat = g['category']
            if cat not in category_groups:
                category_groups[cat] = []
            category_groups[cat].append(g)
        
        # 计算每个类别应该抽取的数量
        num_categories = len(category_groups)
        base_per_category = num_samples // num_categories
        remainder = num_samples % num_categories
        
        sampled_groups = []
        categories_list = list(category_groups.keys())
        random.shuffle(categories_list)
        
        for i, cat in enumerate(categories_list):
            cat_groups = category_groups[cat]
            # 前remainder个类别多抽一个
            n_to_sample = base_per_category + (1 if i < remainder else 0)
            n_to_sample = min(n_to_sample, len(cat_groups))
            sampled = random.sample(cat_groups, n_to_sample)
            sampled_groups.extend(sampled)
        
        # 如果还没抽够，从剩余中补充
        if len(sampled_groups) < num_samples:
            sampled_ids = {g['group_id'] for g in sampled_groups}
            remaining = [g for g in all_groups if g['group_id'] not in sampled_ids]
            extra_needed = num_samples - len(sampled_groups)
            if remaining:
                sampled_groups.extend(random.sample(remaining, min(extra_needed, len(remaining))))
    else:
        # 简单随机抽取
        sampled_groups = random.sample(all_groups, num_samples)
    
    # 统计抽取结果中各类别的数量
    category_counts = {}
    for g in sampled_groups:
        cat = g['category']
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    result = {
        "metadata": {
            "num_sampled_groups": len(sampled_groups),
            "group_size": groups_data['metadata']['group_size'],
            "total_available_groups": len(all_groups),
            "seed": seed,
            "balanced_sampling": balanced,
            "category_filter": category_filter
        },
        "sampled_category_counts": category_counts,
        "groups": sampled_groups
    }
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="从相似度组文件中随机抽取指定数量的组"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="输入的组JSON文件路径（由generate_similarity_groups.py生成）"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="输出的抽样结果JSON文件路径（默认在输入文件同目录下生成）"
    )
    parser.add_argument(
        "--num_samples", "-n",
        type=int,
        required=True,
        help="要抽取的组数"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=None,
        help="随机种子，用于复现结果（可选）"
    )
    parser.add_argument(
        "--categories", "-c",
        type=str,
        nargs='+',
        default=None,
        help="只从指定的类别中抽取（可选，空格分隔多个类别）"
    )
    parser.add_argument(
        "--balanced", "-b",
        action='store_true',
        help="是否在各类别间均衡抽取"
    )
    
    args = parser.parse_args()
    
    # 读取输入文件
    print(f"正在读取输入文件: {args.input}")
    with open(args.input, 'r', encoding='utf-8') as f:
        groups_data = json.load(f)
    
    # 抽取组
    print(f"正在随机抽取 {args.num_samples} 个组...")
    if args.seed is not None:
        print(f"  使用随机种子: {args.seed}")
    if args.categories:
        print(f"  类别过滤: {args.categories}")
    if args.balanced:
        print(f"  均衡抽取模式")
    
    result = sample_groups(
        groups_data,
        args.num_samples,
        seed=args.seed,
        category_filter=args.categories,
        balanced=args.balanced
    )
    
    # 确定输出文件路径
    if args.output is None:
        input_dir = os.path.dirname(args.input)
        input_name = os.path.basename(args.input)
        base_name = os.path.splitext(input_name)[0]
        output_path = os.path.join(input_dir, f"{base_name}_sampled_{args.num_samples}.json")
    else:
        output_path = args.output
    
    # 保存结果
    print(f"正在保存结果到: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    # 打印统计信息
    print("\n===== 抽样统计 =====")
    print(f"可用组数: {result['metadata']['total_available_groups']}")
    print(f"抽取组数: {result['metadata']['num_sampled_groups']}")
    print(f"每组物体数: {result['metadata']['group_size']}")
    
    print("\n===== 各类别抽取数量 =====")
    for category, count in sorted(result['sampled_category_counts'].items(), 
                                   key=lambda x: x[1], 
                                   reverse=True):
        print(f"  {category}: {count}组")
    
    print("\n完成!")


if __name__ == "__main__":
    main()
