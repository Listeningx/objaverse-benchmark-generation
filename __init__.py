"""
Agent-Skills 排序系统

用于高精度排序 ground truth 生成的 Agent-Skills 风格排序系统。

模块结构:
- dimension_bank.py: 维度银行定义
- llm_interface.py: LLM 调用接口（含 mock 实现）
- skills.py: 所有 Skill 类定义
- pipeline.py: 排序流水线
- main.py: 主程序入口和演示示例
"""

from dimension_bank import (
    DIMENSION_BANK,
    get_dimension_bank,
    get_all_dimensions_flat,
    search_dimensions_by_scenario
)

from llm_interface import (
    LLMInterface,
    get_llm_interface,
    call_llm
)

from skills import (
    BaseSkill,
    DimensionPlannerSkill,
    DescriptorSkill,
    JudgeSkill,
    ValidateSkill
)

from pipeline import (
    RankingPipeline,
    PipelineConfig,
    create_pipeline
)

__version__ = "1.0.0"
__author__ = "Agent-Skills Ranking System"

__all__ = [
    # 维度银行
    "DIMENSION_BANK",
    "get_dimension_bank",
    "get_all_dimensions_flat",
    "search_dimensions_by_scenario",
    
    # LLM 接口
    "LLMInterface",
    "get_llm_interface",
    "call_llm",
    
    # Skills
    "BaseSkill",
    "DimensionPlannerSkill",
    "DescriptorSkill",
    "JudgeSkill",
    "ValidateSkill",
    
    # Pipeline
    "RankingPipeline",
    "PipelineConfig",
    "create_pipeline"
]
