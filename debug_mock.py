"""
调试脚本：验证 mock 响应是否正确
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from skills import JudgeSkill, DimensionPlannerSkill
from dimension_bank import get_dimension_bank
import json

# 1. 测试 DimensionPlannerSkill
print("=" * 50)
print("测试 DimensionPlannerSkill")
print("=" * 50)

planner = DimensionPlannerSkill()
planner_result = planner.run({
    "query": "我需要一个适合办公室使用的水杯",
    "dimension_bank": get_dimension_bank()
})

print(f"推断场景: {planner_result.get('inferred_scenario')}")
dimensions = planner_result.get("dimensions", [])
print(f"维度数量: {len(dimensions)}")
for dim in dimensions:
    print(f"  - {dim['name']}: 权重 {dim.get('weight', 0):.2f}")

# 2. 测试 JudgeSkill
print("\n" + "=" * 50)
print("测试 JudgeSkill")
print("=" * 50)

judge = JudgeSkill()
candidate_descriptions = {
    "overall_shape": "测试形状描述",
    "primary_function": "测试功能描述"
}

judge_result = judge.run({
    "query": "我需要一个适合办公室使用的水杯",
    "candidate_id": "test_001",
    "candidate_descriptions": candidate_descriptions,
    "dimensions": dimensions
})

print(f"候选物品 ID: {judge_result.get('candidate_id')}")
print(f"返回的 scores 类型: {type(judge_result.get('scores'))}")
print(f"返回的 scores 内容:")
scores = judge_result.get("scores", {})
print(json.dumps(scores, ensure_ascii=False, indent=2))

# 3. 验证维度名是否匹配
print("\n" + "=" * 50)
print("验证维度名匹配")
print("=" * 50)

dimension_names = [d["name"] for d in dimensions]
score_keys = list(scores.keys())

print(f"规划的维度: {dimension_names}")
print(f"评分的维度: {score_keys}")

matching = set(dimension_names) & set(score_keys)
print(f"匹配的维度: {matching}")
print(f"匹配数量: {len(matching)}/{len(dimension_names)}")

if len(matching) == 0:
    print("\n警告：没有匹配的维度！这说明 mock 响应中的维度名与规划的维度名不一致。")
