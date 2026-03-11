#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从已分类的3D资产JSON文件中生成同类物体相似度排序的ground truth组。
每个组包含10个同类物体，用于相似度排序评测。

输入: 按类别分类的JSON文件
输出: 包含所有可选组的JSON文件，每个组包含10个同类物体
"""

import json
import os
import argparse
from typing import Dict, List, Any


def extract_object_id(mesh_path: str) -> str:
    """
    从mesh_path中提取物体ID
    例如: "objaverse/hf-objaverse-v1/000-117/eddeed80a8fd4fbfbe103a4e66d33793"
    返回: "eddeed80a8fd4fbfbe103a4e66d33793"
    """
    return mesh_path.rstrip('/').split('/')[-1]


def generate_groups(categorized_data: Dict[str, List[Dict]], group_size: int = 10) -> Dict[str, Any]:
    """
    从分类数据中生成物体组
    
    Args:
        categorized_data: 按类别分类的物体字典
        group_size: 每组物体数量，默认为10
    
    Returns:
        包含所有组的字典，包括统计信息
    """
    all_groups = []
    category_stats = {}
    
    for category, objects in categorized_data.items():
        num_objects = len(objects)
        num_groups = num_objects // group_size
        remainder = num_objects % group_size
        
        category_stats[category] = {
            "total_objects": num_objects,
            "num_groups": num_groups,
            "discarded_objects": remainder
        }
        
        # 只处理有足够物体形成至少一组的类别
        if num_groups == 0:
            continue
        
        # 按每group_size个物体分组
        for group_idx in range(num_groups):
            start_idx = group_idx * group_size
            end_idx = start_idx + group_size
            group_objects = objects[start_idx:end_idx]
            
            # 为每个物体添加ID和类别信息
            group_items = []
            for obj in group_objects:
                item = obj.copy()  # 保留所有原始字段
                item['object_id'] = extract_object_id(obj['mesh_path'])
                item['category'] = category
                group_items.append(item)
            
            group_info = {
                "group_id": f"{category}_{group_idx}",
                "category": category,
                "group_index": group_idx,
                "objects": group_items
            }
            all_groups.append(group_info)
    
    result = {
        "metadata": {
            "group_size": group_size,
            "total_groups": len(all_groups),
            "total_categories": len(categorized_data),
            "categories_with_groups": len([c for c, s in category_stats.items() if s["num_groups"] > 0])
        },
        "category_statistics": category_stats,
        "groups": all_groups
    }
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="从分类JSON文件生成相似度排序ground truth组"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="输入的分类JSON文件路径"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="输出的组JSON文件路径（默认在输入文件同目录下生成）"
    )
    parser.add_argument(
        "--group_size", "-g",
        type=int,
        default=10,
        help="每组物体数量，默认为10"
    )
    
    args = parser.parse_args()
    
    # 读取输入文件
    print(f"正在读取输入文件: {args.input}")
    with open(args.input, 'r', encoding='utf-8') as f:
        categorized_data = json.load(f)
    
    # 生成组
    print(f"正在生成物体组（每组 {args.group_size} 个物体）...")
    result = generate_groups(categorized_data, args.group_size)
    
    # 确定输出文件路径
    if args.output is None:
        input_dir = os.path.dirname(args.input)
        input_name = os.path.basename(args.input)
        base_name = os.path.splitext(input_name)[0]
        output_path = os.path.join(input_dir, f"{base_name}_similarity_groups.json")
    else:
        output_path = args.output
    
    # 保存结果
    print(f"正在保存结果到: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    # 打印统计信息
    print("\n===== 统计信息 =====")
    print(f"总类别数: {result['metadata']['total_categories']}")
    print(f"有效类别数（至少1组）: {result['metadata']['categories_with_groups']}")
    print(f"总组数: {result['metadata']['total_groups']}")
    print(f"每组物体数: {result['metadata']['group_size']}")
    
    print("\n===== 各类别详情 =====")
    for category, stats in sorted(result['category_statistics'].items(), 
                                   key=lambda x: x[1]['num_groups'], 
                                   reverse=True):
        if stats['num_groups'] > 0:
            print(f"  {category}: {stats['total_objects']}个物体 -> {stats['num_groups']}组 (舍弃{stats['discarded_objects']}个)")
    
    print("\n完成!")


if __name__ == "__main__":
    main()
