#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从 groups JSON 文件中筛选指定 category 的组并另存为新文件
支持批量生成多个类别的分组文件
"""

import json
import sys

# 输入路径
input_path = "/apdcephfs/share_303565425/DCC3/serenasnliu/GAR/Uni3D/data/test_datasets/objaverse/objaverse_golden_all_groups.json"
# 输出目录（与输入文件同目录）
output_dir = "/apdcephfs/share_303565425/DCC3/serenasnliu/GAR/Uni3D/data/test_datasets/objaverse"

# 要筛选的类别列表
TARGET_CATEGORIES = ["Character", "Object", "Building", "Weapon", "Vehicle", "Animal"]

# 读取原文件
print(f"正在读取: {input_path}")
with open(input_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

original_groups = data.get('groups', [])
print(f"原始总组数: {len(original_groups)}")

# 如果命令行指定了类别，则只处理指定的类别
if len(sys.argv) > 1:
    categories_to_process = sys.argv[1:]
    # 验证类别是否有效
    for cat in categories_to_process:
        if cat not in TARGET_CATEGORIES:
            print(f"⚠️ 未知类别: {cat}，支持的类别: {TARGET_CATEGORIES}")
    categories_to_process = [c for c in categories_to_process if c in TARGET_CATEGORIES]
else:
    categories_to_process = TARGET_CATEGORIES

print(f"将处理以下类别: {categories_to_process}\n")

for target_category in categories_to_process:
    # 生成输出文件名：类别名转小写
    category_lower = target_category.lower()
    output_path = f"{output_dir}/objaverse_golden_{category_lower}_groups.json"
    
    # 筛选指定category的groups
    filtered_groups = [g for g in original_groups if g.get('category') == target_category]
    
    print(f"[{target_category}] 筛选后组数: {len(filtered_groups)}")
    
    if len(filtered_groups) == 0:
        print(f"  ⚠️ 类别 {target_category} 没有找到任何组，跳过")
        continue
    
    # 构建新的数据结构
    new_data = {
        "metadata": {
            "group_size": 10,
            "total_groups": len(filtered_groups),
            "total_categories": 1,
            "categories_with_groups": 1,
            "source_file": input_path,
            "filtered_category": target_category
        },
        "category_statistics": {
            target_category: data.get('category_statistics', {}).get(target_category, {})
        },
        "groups": filtered_groups
    }
    
    # 保存新文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)
    
    print(f"  ✅ 已保存到: {output_path}")

print("\n完成!")
