"""
主程序入口

演示如何使用 Agent-Skills 排序系统进行高精度排序 ground truth 生成。
"""

import json
import sys
import os

# 确保模块路径正确
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dimension_bank import get_dimension_bank, DIMENSION_BANK
from llm_interface import get_llm_interface, call_llm, LLMInterface
from skills import (
    DimensionPlannerSkill,
    DescriptorSkill,
    JudgeSkill,
    ValidateSkill
)
from pipeline import RankingPipeline, PipelineConfig, create_pipeline


def demo_basic_usage():
    """
    基础使用示例
    
    演示最简单的使用方式：直接使用 RankingPipeline 进行排序。
    """
    print("\n" + "=" * 70)
    print("示例 1: 基础使用")
    print("=" * 70)
    
    # 定义查询和候选物品
    query = "我需要一个适合办公室使用的水杯，要求保温性能好，外观简约大方，容量适中（约350-500ml），便于携带。"
    
    candidate_ids = [
        "candidate_001",  # 假设是一个不锈钢保温杯
        "candidate_002",  # 假设是一个玻璃水杯
        "candidate_003",  # 假设是一个塑料运动水壶
        "candidate_004",  # 假设是一个陶瓷马克杯
        "candidate_005"   # 假设是一个保温瓶
    ]
    
    # 可选：提供候选物品的附加信息
    candidate_info = {
        "candidate_001": "不锈钢双层真空保温杯，500ml容量，磨砂黑色外观，带提手",
        "candidate_002": "透明玻璃水杯，450ml容量，带硅胶套，简约设计",
        "candidate_003": "运动塑料水壶，750ml大容量，带吸管，多色可选",
        "candidate_004": "陶瓷马克杯，350ml容量，白色简约风格，带盖子",
        "candidate_005": "大容量保温瓶，1L容量，户外风格，带背带"
    }
    
    # 创建并执行流水线
    pipeline = RankingPipeline(verbose=True)
    result = pipeline.run(
        query=query,
        candidate_ids=candidate_ids,
        candidate_info=candidate_info
    )
    
    # 打印最终排序结果
    print("\n" + "-" * 50)
    print("最终排序结果（从最相似到最不相似）:")
    print("-" * 50)
    for rank, candidate_id in enumerate(result["final_ranking"], 1):
        print(f"  {rank}. {candidate_id}")
    
    # 生成并打印解释报告
    print("\n" + "-" * 50)
    print("生成解释报告:")
    print("-" * 50)
    report = pipeline.generate_explanation_report(result)
    print(report)
    
    return result


def demo_custom_dimension_bank():
    """
    自定义维度银行示例
    
    演示如何使用自定义的维度银行。
    """
    print("\n" + "=" * 70)
    print("示例 2: 自定义维度银行")
    print("=" * 70)
    
    # 定义自定义维度银行（针对服装领域）
    custom_dimension_bank = {
        "visual": [
            {
                "name": "color_style",
                "description": "服装的颜色和色彩搭配风格",
                "applicable_scenarios": ["fashion", "matching", "style"]
            },
            {
                "name": "silhouette",
                "description": "服装的整体轮廓和版型",
                "applicable_scenarios": ["fit", "body_type", "style"]
            },
            {
                "name": "pattern_design",
                "description": "服装的图案和印花设计",
                "applicable_scenarios": ["aesthetic", "occasion", "style"]
            }
        ],
        "functional": [
            {
                "name": "comfort",
                "description": "穿着舒适度，包括面料触感和活动自由度",
                "applicable_scenarios": ["daily_wear", "sports", "casual"]
            },
            {
                "name": "occasion_suitability",
                "description": "适合的场合和穿着情境",
                "applicable_scenarios": ["formal", "casual", "special_event"]
            }
        ],
        "quality": [
            {
                "name": "material_quality",
                "description": "面料品质和做工精细度",
                "applicable_scenarios": ["premium", "durability", "value"]
            }
        ]
    }
    
    # 配置流水线
    config = PipelineConfig(
        use_custom_dimension_bank=True,
        custom_dimension_bank=custom_dimension_bank,
        verbose=True
    )
    
    # 创建流水线
    pipeline = create_pipeline(config)
    
    # 定义查询
    query = "我需要一件适合商务场合穿着的男士西装外套，要求修身剪裁，深色系，面料质感好。"
    
    candidate_ids = [
        "suit_001",
        "suit_002",
        "suit_003"
    ]
    
    # 执行流水线
    result = pipeline.run(
        query=query,
        candidate_ids=candidate_ids
    )
    
    print("\n最终排序:", result["final_ranking"])
    
    return result


def demo_individual_skills():
    """
    单独使用各个 Skill 的示例
    
    演示如何独立调用每个 Skill。
    """
    print("\n" + "=" * 70)
    print("示例 3: 单独使用各个 Skill")
    print("=" * 70)
    
    # 1. 使用 DimensionPlannerSkill
    print("\n--- 测试 DimensionPlannerSkill ---")
    planner = DimensionPlannerSkill()
    
    planner_result = planner.run({
        "query": "找一个适合拍摄人像的相机镜头",
        "dimension_bank": get_dimension_bank()
    })
    
    print(f"推断场景: {planner_result.get('inferred_scenario', 'N/A')}")
    print(f"规划维度数量: {len(planner_result.get('dimensions', []))}")
    
    dimensions = planner_result.get("dimensions", [])
    for dim in dimensions:
        print(f"  - {dim['name']}: 权重 {dim.get('weight', 0):.2f}")
    
    # 2. 使用 DescriptorSkill
    print("\n--- 测试 DescriptorSkill ---")
    descriptor = DescriptorSkill()
    
    descriptor_result = descriptor.run({
        "candidate_id": "lens_001",
        "candidate_info": "85mm f/1.4 定焦镜头",
        "dimensions": dimensions
    })
    
    print(f"候选物品: {descriptor_result.get('candidate_id', 'N/A')}")
    print("描述维度:")
    for dim_name, desc in descriptor_result.get("descriptions", {}).items():
        print(f"  - {dim_name}: {desc[:50]}...")
    
    # 3. 使用 JudgeSkill
    print("\n--- 测试 JudgeSkill ---")
    judge = JudgeSkill()
    
    judge_result = judge.run({
        "query": "找一个适合拍摄人像的相机镜头",
        "candidate_id": "lens_001",
        "candidate_descriptions": descriptor_result.get("descriptions", {}),
        "dimensions": dimensions
    })
    
    print(f"候选物品: {judge_result.get('candidate_id', 'N/A')}")
    print("评分结果:")
    for dim_name, score_data in judge_result.get("scores", {}).items():
        # 支持简化格式（分数直接是数字）和标准格式（分数是对象）
        if isinstance(score_data, (int, float)):
            print(f"  - {dim_name}: {float(score_data):.2f}")
        elif isinstance(score_data, dict):
            print(f"  - {dim_name}: {score_data.get('score', 0):.2f}")
    
    # 4. 使用 ValidateSkill
    print("\n--- 测试 ValidateSkill ---")
    validator = ValidateSkill()
    
    # 模拟多个候选物品的评分
    all_scores = {
        "lens_001": {"dimension_scores": judge_result.get("scores", {})},
        "lens_002": {"dimension_scores": {
            dim["name"]: {"score": 0.75, "reason": "中等相似度"}
            for dim in dimensions
        }},
        "lens_003": {"dimension_scores": {
            dim["name"]: {"score": 0.6, "reason": "较低相似度"}
            for dim in dimensions
        }}
    }
    
    validate_result = validator.run({
        "all_candidate_scores": all_scores,
        "dimensions": dimensions
    })
    
    print(f"最终排序: {validate_result.get('final_ranking', [])}")
    print(f"验证说明: {validate_result.get('validation_notes', 'N/A')[:100]}...")
    
    return {
        "planner": planner_result,
        "descriptor": descriptor_result,
        "judge": judge_result,
        "validate": validate_result
    }


def demo_export_results():
    """
    导出结果示例
    
    演示如何导出排序结果到文件。
    """
    print("\n" + "=" * 70)
    print("示例 4: 导出结果")
    print("=" * 70)
    
    # 创建流水线
    pipeline = RankingPipeline(verbose=False)  # 关闭详细日志
    
    # 执行排序
    result = pipeline.run(
        query="寻找一款适合户外露营的便携式蓝牙音箱，需要防水防尘，续航时间长。",
        candidate_ids=["speaker_001", "speaker_002", "speaker_003", "speaker_004"]
    )
    
    # 导出 JSON 结果
    output_path = os.path.join(os.path.dirname(__file__), "ranking_result.json")
    pipeline.export_result(result, output_path)
    print(f"JSON 结果已导出到: {output_path}")
    
    # 生成并导出文本报告
    report = pipeline.generate_explanation_report(result)
    report_path = os.path.join(os.path.dirname(__file__), "ranking_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"文本报告已导出到: {report_path}")
    
    return result


def demo_llm_statistics():
    """
    LLM 调用统计示例
    
    演示如何查看 LLM 调用统计信息。
    """
    print("\n" + "=" * 70)
    print("示例 5: LLM 调用统计")
    print("=" * 70)
    
    # 重置 LLM 接口统计
    llm = get_llm_interface()
    llm.reset_statistics()
    
    # 执行一个简单的流水线
    pipeline = RankingPipeline(verbose=False)
    pipeline.run(
        query="找一个适合书房使用的台灯",
        candidate_ids=["lamp_001", "lamp_002"]
    )
    
    # 查看统计信息
    stats = llm.get_call_statistics()
    print(f"LLM 调用统计:")
    print(f"  - 总调用次数: {stats['total_calls']}")
    print(f"  - 运行模式: {stats['mode']}")
    print(f"  - 历史记录数: {stats['history_count']}")
    
    return stats


def main():
    """
    主函数
    
    运行所有演示示例。
    """
    print("\n" + "=" * 70)
    print("Agent-Skills 排序系统演示")
    print("=" * 70)
    print("\n本系统用于生成高精度排序的 ground truth。")
    print("系统特点:")
    print("  1. Agent-Skills 架构，顺序执行")
    print("  2. 可解释性强，每个决策都有清晰的解释路径")
    print("  3. 支持维度银行，同时允许动态新增维度")
    print("  4. Mock LLM 实现，可直接运行测试")
    
    # 运行各个示例
    try:
        # 示例 1: 基础使用
        result1 = demo_basic_usage()
        
        # 示例 2: 自定义维度银行
        result2 = demo_custom_dimension_bank()
        
        # 示例 3: 单独使用各个 Skill
        result3 = demo_individual_skills()
        
        # 示例 4: 导出结果
        result4 = demo_export_results()
        
        # 示例 5: LLM 调用统计
        result5 = demo_llm_statistics()
        
        print("\n" + "=" * 70)
        print("所有演示示例执行完成！")
        print("=" * 70)
        
        return {
            "basic_usage": result1,
            "custom_dimension_bank": result2,
            "individual_skills": result3,
            "export_results": result4,
            "llm_statistics": result5
        }
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
