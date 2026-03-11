"""
维度银行（Dimension Bank）定义模块

维度银行是知识资源，用于辅助维度规划，不是固定模板，不是强约束。
LLM 可以在规划阶段根据需要新增维度。
"""

# 维度银行定义
DIMENSION_BANK = {
    "visual": [
        {
            "name": "overall_shape",
            "description": "物体的整体几何外形特征，包括轮廓、比例、对称性等",
            "applicable_scenarios": ["identification", "replacement", "similarity", "visual_matching"]
        },
        {
            "name": "color_pattern",
            "description": "物体的颜色分布、色调、饱和度及图案特征",
            "applicable_scenarios": ["aesthetic", "decoration", "visual_matching", "style"]
        },
        {
            "name": "texture_surface",
            "description": "物体表面的质感、纹理、光泽度等特征",
            "applicable_scenarios": ["material_identification", "quality", "tactile", "visual_matching"]
        },
        {
            "name": "size_scale",
            "description": "物体的尺寸大小、相对比例关系",
            "applicable_scenarios": ["spatial_planning", "replacement", "fit_check"]
        },
        {
            "name": "structural_complexity",
            "description": "物体的结构复杂度，包括部件数量、组装方式等",
            "applicable_scenarios": ["assembly", "manufacturing", "complexity_analysis"]
        },
        {
            "name": "visual_style",
            "description": "物体的视觉风格，如现代、复古、简约、华丽等",
            "applicable_scenarios": ["aesthetic", "design", "style_matching"]
        }
    ],
    "functional": [
        {
            "name": "affordance",
            "description": "物体支持的使用方式与功能，包括如何操作、握持、交互等",
            "applicable_scenarios": ["usage", "tool", "interaction", "ergonomics"]
        },
        {
            "name": "primary_function",
            "description": "物体的主要功能与用途",
            "applicable_scenarios": ["replacement", "alternative", "functional_matching"]
        },
        {
            "name": "secondary_functions",
            "description": "物体的次要功能或附加用途",
            "applicable_scenarios": ["versatility", "multi_purpose", "value_assessment"]
        },
        {
            "name": "operational_mechanism",
            "description": "物体的工作原理和操作机制",
            "applicable_scenarios": ["technical", "repair", "understanding"]
        },
        {
            "name": "performance_capability",
            "description": "物体的性能表现和能力范围",
            "applicable_scenarios": ["comparison", "selection", "quality"]
        }
    ],
    "contextual": [
        {
            "name": "usage_scene_match",
            "description": "物体是否匹配目标使用场景，包括环境适应性",
            "applicable_scenarios": ["scenario_matching", "environment_fit", "context"]
        },
        {
            "name": "target_user_fit",
            "description": "物体是否适合目标用户群体的需求和特征",
            "applicable_scenarios": ["user_matching", "personalization", "accessibility"]
        },
        {
            "name": "cultural_relevance",
            "description": "物体在特定文化背景下的相关性和适宜性",
            "applicable_scenarios": ["cultural", "regional", "symbolic"]
        },
        {
            "name": "temporal_relevance",
            "description": "物体在时间维度上的相关性，如季节性、时代性",
            "applicable_scenarios": ["seasonal", "trend", "timeless"]
        }
    ],
    "material": [
        {
            "name": "material_composition",
            "description": "物体的材质组成，如金属、木材、塑料、玻璃等",
            "applicable_scenarios": ["material_matching", "quality", "durability"]
        },
        {
            "name": "durability_quality",
            "description": "物体的耐用性和质量等级",
            "applicable_scenarios": ["longevity", "investment", "quality"]
        },
        {
            "name": "weight_density",
            "description": "物体的重量和密度特征",
            "applicable_scenarios": ["portability", "handling", "shipping"]
        }
    ],
    "semantic": [
        {
            "name": "category_membership",
            "description": "物体所属的语义类别和分类层级",
            "applicable_scenarios": ["classification", "taxonomy", "identification"]
        },
        {
            "name": "conceptual_similarity",
            "description": "物体在概念层面上的相似性",
            "applicable_scenarios": ["association", "analogy", "conceptual_matching"]
        },
        {
            "name": "brand_identity",
            "description": "物体的品牌特征和身份标识",
            "applicable_scenarios": ["brand_matching", "authenticity", "premium"]
        }
    ],
    "relational": [
        {
            "name": "complementary_objects",
            "description": "物体与其他物品的搭配和互补关系",
            "applicable_scenarios": ["pairing", "set_completion", "coordination"]
        },
        {
            "name": "substitutability",
            "description": "物体作为替代品的可行性和程度",
            "applicable_scenarios": ["replacement", "alternative", "backup"]
        },
        {
            "name": "compatibility",
            "description": "物体与其他系统或物品的兼容性",
            "applicable_scenarios": ["integration", "system_fit", "interoperability"]
        }
    ]
}


def get_dimension_bank() -> dict:
    """
    获取维度银行
    
    Returns:
        dict: 维度银行字典
    """
    return DIMENSION_BANK


def get_all_dimensions_flat() -> list:
    """
    获取所有维度的扁平化列表
    
    Returns:
        list: 包含所有维度的列表
    """
    all_dimensions = []
    for category, dimensions in DIMENSION_BANK.items():
        for dim in dimensions:
            dim_copy = dim.copy()
            dim_copy["category"] = category
            all_dimensions.append(dim_copy)
    return all_dimensions


def search_dimensions_by_scenario(scenario: str) -> list:
    """
    根据场景搜索相关维度
    
    Args:
        scenario: 使用场景
        
    Returns:
        list: 匹配的维度列表
    """
    matched_dimensions = []
    scenario_lower = scenario.lower()
    
    for category, dimensions in DIMENSION_BANK.items():
        for dim in dimensions:
            for applicable_scenario in dim["applicable_scenarios"]:
                if scenario_lower in applicable_scenario.lower() or applicable_scenario.lower() in scenario_lower:
                    dim_copy = dim.copy()
                    dim_copy["category"] = category
                    matched_dimensions.append(dim_copy)
                    break
    
    return matched_dimensions
