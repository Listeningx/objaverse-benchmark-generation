"""
Skills 模块

包含所有 Agent-Skills 的类定义：
- DimensionPlannerSkill: 维度规划
- DescriptorSkill: 候选描述
- JudgeSkill: 相似度判断
- ValidateSkill: 排序与一致性校验
"""

import json
import os
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod

from llm_interface import call_llm, call_llm_with_images, call_llm_with_all_images
from dimension_bank import get_dimension_bank, get_all_dimensions_flat


class BaseSkill(ABC):
    # ... existing code ...
    """
    Skill 基类
    
    所有 skill 必须继承此基类并实现 run() 方法。
    """
    
    def __init__(self, name: str, description: str):
        """
        初始化 Skill
        
        Args:
            name: Skill 名称
            description: Skill 描述
        """
        self.name = name
        self.description = description
        self.execution_log = []
    
    @abstractmethod
    def get_prompt_template(self) -> str:
        """获取 prompt 模板"""
        pass
    
    @abstractmethod
    def get_input_schema(self) -> dict:
        """获取输入 schema"""
        pass
    
    @abstractmethod
    def get_output_schema(self) -> dict:
        """获取输出 schema"""
        pass
    
    @abstractmethod
    def run(self, input_data: dict) -> dict:
        """
        执行 skill
        
        Args:
            input_data: 输入数据
            
        Returns:
            dict: 输出结果
        """
        pass
    
    def _log_execution(self, stage: str, data: Any):
        """记录执行日志"""
        self.execution_log.append({
            "stage": stage,
            "data": data
        })
    
    def _parse_json_response(self, response: str) -> dict:
        """
        解析 LLM 返回的 JSON 响应
        
        Args:
            response: LLM 响应字符串
            
        Returns:
            dict: 解析后的字典
        """
        import re
        
        def _fix_json_string(json_str: str) -> str:
            """修复常见的 JSON 格式问题"""
            # 移除可能的 BOM 标记
            json_str = json_str.lstrip('\ufeff')
            
            # 移除尾随逗号 (在 ] 或 } 之前的逗号)
            json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
            
            # 将单引号替换为双引号（仅在键名位置）
            # 注意：这是一个简化的处理，可能不适用于所有情况
            json_str = re.sub(r"'([^']+)'(\s*:)", r'"\1"\2', json_str)
            
            # 移除注释 (// 和 /* */)
            json_str = re.sub(r'//.*?$', '', json_str, flags=re.MULTILINE)
            json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
            
            # 处理未加引号的键名
            json_str = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)(\s*:)', r'\1"\2"\3', json_str)
            
            return json_str
        
        def _try_parse(json_str: str) -> dict:
            """尝试解析 JSON，如果失败则尝试修复后再解析"""
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                # 尝试修复后再解析
                fixed_str = _fix_json_string(json_str)
                return json.loads(fixed_str)
        
        try:
            # 尝试直接解析
            return _try_parse(response)
        except json.JSONDecodeError:
            pass
        
        # 尝试提取 JSON 块
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response)
        if json_match:
            try:
                return _try_parse(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # 尝试提取大括号内容
        brace_match = re.search(r'\{[\s\S]*\}', response)
        if brace_match:
            try:
                return _try_parse(brace_match.group(0))
            except json.JSONDecodeError as e:
                # 记录详细错误信息便于调试
                error_pos = e.pos if hasattr(e, 'pos') else 0
                error_context = brace_match.group(0)[max(0, error_pos-50):error_pos+50]
                print(f"JSON 解析错误位置附近内容: ...{error_context}...")
        
        # 尝试提取方括号内容（数组）
        bracket_match = re.search(r'\[[\s\S]*\]', response)
        if bracket_match:
            try:
                return _try_parse(bracket_match.group(0))
            except json.JSONDecodeError:
                pass
        
        raise ValueError(f"无法解析 LLM 响应为 JSON: {response[:500]}...")


class DimensionPlannerSkill(BaseSkill):
    # ... existing code ...
    """
    维度规划 Skill
    
    功能：针对当前 case，生成用于排序的最终维度集合。
    
    执行逻辑：
    1. 从 query 推断使用场景（scenario）
    2. 从维度银行中召回所有可能相关的维度
    3. 对召回维度进行筛选、合并、重写
    4. 自检是否存在语义空缺，若存在则新增维度
    5. 为最终维度分配权重（权重和为 1）
    6. 给出每个维度的明确评分标准
    """
    
    # 内嵌 prompt 模板
    PROMPT_TEMPLATE = """你是一个维度规划专家，负责为物品相似度排序任务规划评估维度。

## 输入信息

### 用户查询
{query}

### 维度银行
{dimension_bank_formatted}

## 任务要求

1. 分析查询，推断使用场景
2. 从维度银行召回相关维度，筛选/合并/重写
3. 检查是否有遗漏的重要维度，必要时新增
4. 为每个维度分配权重（和为1）
5. 为每个维度制定简洁的评分标准（5分制，每档一句话）

## 输出格式（精简模式）

```json
{{
  "inferred_scenario": "使用场景（一句话）",
  "dimensions": [
    {{
      "name": "维度名称",
      "description": "一句话描述",
      "weight": 0.XX,
      "scoring_criteria": "5分:...; 4分:...; 3分:...; 2分:...; 1分:...",
      "source": "bank/invented"
    }}
  ]
}}
```

## 约束
- 维度数量：4-8个
- 权重和=1
- 输出要精简，不要冗余描述

请直接输出JSON。
"""
    
    # 基于图片的 prompt 模板
    PROMPT_TEMPLATE_WITH_IMAGE = """你是一个维度规划专家，负责为物品相似度排序任务规划评估维度。

## 输入信息

### 查询图片
【图片已附在本条消息中】
补充描述：{query_description}

### 维度银行
{dimension_bank_formatted}

## 任务要求

1. 分析图片中物体的类别和特征
2. 推断使用场景
3. 从维度银行召回、筛选、优化维度
4. 分配权重（和为1）
5. 制定简洁的评分标准（5分制）

## 输出格式（精简模式）

```json
{{
  "query_object_analysis": "物体类别+关键特征（一句话）",
  "inferred_scenario": "使用场景（一句话）",
  "dimensions": [
    {{
      "name": "维度名称",
      "description": "一句话描述",
      "weight": 0.XX,
      "scoring_criteria": "5分:...; 4分:...; 3分:...; 2分:...; 1分:...",
      "source": "bank/invented"
    }}
  ]
}}
```

## 约束
- 维度数量：4-8个
- 权重和=1
- 输出要精简

请直接输出JSON。
"""
    
    def __init__(self):
        super().__init__(
            name="DimensionPlannerSkill",
            description="维度规划技能：根据查询生成用于排序的评估维度集合"
        )
    
    def get_prompt_template(self) -> str:
        return self.PROMPT_TEMPLATE
    
    def get_input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "用户查询，包含目标物品描述和使用场景"
                },
                "dimension_bank": {
                    "type": "object",
                    "description": "维度银行，包含可用的评估维度"
                }
            },
            "required": ["query", "dimension_bank"]
        }
    
    def get_output_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "inferred_scenario": {
                    "type": "string",
                    "description": "推断出的使用场景"
                },
                "scenario_reasoning": {
                    "type": "string",
                    "description": "场景推断的理由"
                },
                "dimensions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                            "weight": {"type": "number"},
                            "scoring_criteria": {"type": "string"},
                            "source": {"type": "string", "enum": ["bank", "invented"]}
                        }
                    }
                }
            }
        }
    
    def _format_dimension_bank(self, dimension_bank: dict) -> str:
        """格式化维度银行为可读字符串"""
        formatted_parts = []
        for category, dimensions in dimension_bank.items():
            formatted_parts.append(f"\n### {category.upper()} 类维度")
            for dim in dimensions:
                formatted_parts.append(f"""
- **{dim['name']}**
  - 描述：{dim['description']}
  - 适用场景：{', '.join(dim['applicable_scenarios'])}""")
        return "\n".join(formatted_parts)
    
    def run(self, input_data: dict) -> dict:
        """
        执行维度规划（文本模式）
        
        Args:
            input_data: 包含 query 和 dimension_bank 的字典
            
        Returns:
            dict: 包含 inferred_scenario 和 dimensions 的结果
        """
        self._log_execution("input", input_data)
        
        # 提取输入
        query = input_data.get("query", "")
        dimension_bank = input_data.get("dimension_bank", get_dimension_bank())
        
        # 格式化维度银行
        dimension_bank_formatted = self._format_dimension_bank(dimension_bank)
        
        # 构造 prompt
        prompt = self.PROMPT_TEMPLATE.format(
            query=query,
            dimension_bank_formatted=dimension_bank_formatted
        )
        
        self._log_execution("prompt_constructed", {"prompt_length": len(prompt)})
        
        # 调用 LLM
        response = call_llm(prompt, expected_format="json")
        
        self._log_execution("llm_response", {"response_length": len(response)})
        
        # 解析响应
        result = self._parse_json_response(response)
        
        # 验证权重和
        self._validate_weights(result.get("dimensions", []))
        
        self._log_execution("output", result)
        
        return result
    
    def run_with_image(self, input_data: dict) -> dict:
        """
        执行维度规划（图片模式）
        
        Args:
            input_data: 包含 query_image, query_description, dimension_bank 的字典
            
        Returns:
            dict: 包含 inferred_scenario 和 dimensions 的结果
        """
        self._log_execution("input", input_data)
        
        # 提取输入
        query_image = input_data.get("query_image", "")  # 图片路径
        query_description = input_data.get("query_description", "无补充描述")
        dimension_bank = input_data.get("dimension_bank", get_dimension_bank())
        
        # 验证图片存在
        if not os.path.exists(query_image):
            raise FileNotFoundError(f"查询图片不存在: {query_image}")
        
        # 格式化维度银行
        dimension_bank_formatted = self._format_dimension_bank(dimension_bank)
        
        # 构造 prompt
        prompt = self.PROMPT_TEMPLATE_WITH_IMAGE.format(
            query_description=query_description,
            dimension_bank_formatted=dimension_bank_formatted
        )
        
        self._log_execution("prompt_constructed", {"prompt_length": len(prompt)})
        
        # 调用 LLM（带图片）
        response = call_llm_with_images(prompt, [query_image], expected_format="json")
        
        self._log_execution("llm_response", {"response_length": len(response)})
        
        # 解析响应
        result = self._parse_json_response(response)
        
        # 验证权重和
        self._validate_weights(result.get("dimensions", []))
        
        self._log_execution("output", result)
        
        return result
    
    def _validate_weights(self, dimensions: list) -> None:
        """验证权重和是否为 1"""
        total_weight = sum(dim.get("weight", 0) for dim in dimensions)
        if abs(total_weight - 1.0) > 0.01:
            # 如果权重和不等于 1，进行归一化
            for dim in dimensions:
                if total_weight > 0:
                    dim["weight"] = dim["weight"] / total_weight


class DescriptorSkill(BaseSkill):
    """
    候选描述 Skill
    
    功能：针对每个 candidate，在每个维度上生成可对比、客观的描述。
    支持文本描述和图片描述两种模式。
    """
    
    PROMPT_TEMPLATE = """你是物品描述专家，对候选物品进行多维度描述。

## 输入
候选ID：{candidate_id}
候选信息：{candidate_info}

## 评估维度
{dimensions_formatted}

## 输出格式（精简模式）

```json
{{
  "candidate_id": "{candidate_id}",
  "descriptions": {{
    "维度1": "20-50字的客观描述",
    "维度2": "20-50字的客观描述"
  }}
}}
```

要求：描述精简、客观、可对比。请直接输出JSON。
"""

    # 基于图片的 prompt 模板
    PROMPT_TEMPLATE_WITH_IMAGE = """你是物品描述专家，基于图片对候选物品进行多维度描述。

## 输入
候选ID：{candidate_id}
图片：【已附在消息中】

## 评估维度
{dimensions_formatted}

## 输出格式（精简模式）

```json
{{
  "candidate_id": "{candidate_id}",
  "descriptions": {{
    "维度1": "20-50字基于图片的客观描述",
    "维度2": "20-50字基于图片的客观描述"
  }}
}}
```

要求：只描述可见特征，精简、客观。请直接输出JSON。
"""
    
    def __init__(self):
        super().__init__(
            name="DescriptorSkill",
            description="候选描述技能：对每个候选物品在各维度上生成客观描述"
        )
    
    def get_prompt_template(self) -> str:
        return self.PROMPT_TEMPLATE
    
    def get_input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "candidate_id": {
                    "type": "string",
                    "description": "候选物品的唯一标识"
                },
                "candidate_info": {
                    "type": "string",
                    "description": "候选物品的附加信息（可选）"
                },
                "dimensions": {
                    "type": "array",
                    "description": "评估维度列表"
                }
            },
            "required": ["candidate_id", "dimensions"]
        }
    
    def get_output_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "candidate_id": {"type": "string"},
                "descriptions": {
                    "type": "object",
                    "additionalProperties": {"type": "string"}
                }
            }
        }
    
    def _format_dimensions(self, dimensions: list) -> str:
        """格式化维度列表为可读字符串"""
        formatted_parts = []
        for i, dim in enumerate(dimensions, 1):
            formatted_parts.append(f"""
{i}. **{dim['name']}**
   - 描述：{dim['description']}
   - 评分标准参考：{dim.get('scoring_criteria', '无')}""")
        return "\n".join(formatted_parts)
    
    def run(self, input_data: dict) -> dict:
        """
        执行候选描述（文本模式）
        
        Args:
            input_data: 包含 candidate_id 和 dimensions 的字典
            
        Returns:
            dict: 包含 candidate_id 和 descriptions 的结果
        """
        self._log_execution("input", input_data)
        
        # 提取输入
        candidate_id = input_data.get("candidate_id", "")
        candidate_info = input_data.get("candidate_info", "暂无附加信息")
        dimensions = input_data.get("dimensions", [])
        
        # 格式化维度列表
        dimensions_formatted = self._format_dimensions(dimensions)
        
        # 构造 prompt
        prompt = self.PROMPT_TEMPLATE.format(
            candidate_id=candidate_id,
            candidate_info=candidate_info,
            dimensions_formatted=dimensions_formatted
        )
        
        self._log_execution("prompt_constructed", {"prompt_length": len(prompt)})
        
        # 调用 LLM
        response = call_llm(prompt, expected_format="json")
        
        self._log_execution("llm_response", {"response_length": len(response)})
        
        # 解析响应
        result = self._parse_json_response(response)
        
        # 确保 candidate_id 正确
        result["candidate_id"] = candidate_id
        
        self._log_execution("output", result)
        
        return result
    
    def run_with_image(self, input_data: dict) -> dict:
        """
        执行候选描述（图片模式）
        
        Args:
            input_data: 包含 candidate_id, candidate_image, dimensions 的字典
            
        Returns:
            dict: 包含 candidate_id 和 descriptions 的结果
        """
        self._log_execution("input", input_data)
        
        # 提取输入
        candidate_id = input_data.get("candidate_id", "")
        candidate_image = input_data.get("candidate_image", "")  # 图片路径
        dimensions = input_data.get("dimensions", [])
        
        # 验证图片存在
        if not os.path.exists(candidate_image):
            raise FileNotFoundError(f"候选物品图片不存在: {candidate_image}")
        
        # 格式化维度列表
        dimensions_formatted = self._format_dimensions(dimensions)
        
        # 构造 prompt
        prompt = self.PROMPT_TEMPLATE_WITH_IMAGE.format(
            candidate_id=candidate_id,
            dimensions_formatted=dimensions_formatted
        )
        
        self._log_execution("prompt_constructed", {"prompt_length": len(prompt)})
        
        # 调用 LLM（带图片）
        response = call_llm_with_images(prompt, [candidate_image], expected_format="json")
        
        self._log_execution("llm_response", {"response_length": len(response)})
        
        # 解析响应
        result = self._parse_json_response(response)
        
        # 确保 candidate_id 正确
        result["candidate_id"] = candidate_id
        
        self._log_execution("output", result)
        
        return result
    
    # 批量处理的 prompt 模板
    PROMPT_TEMPLATE_BATCH = """你是物品描述专家，基于图片对多个候选物品进行多维度描述。

## 图片说明
- 图片1：查询图片（参考）
- 图片2-N：候选物品图片

候选物品列表：
{candidate_list}

## 评估维度
{dimensions_formatted}

## 输出格式（精简模式）

```json
{{
  "batch_descriptions": {{
    "candidate_id_1": {{
      "descriptions": {{"维度1": "20-50字描述", "维度2": "20-50字描述"}}
    }},
    "candidate_id_2": {{
      "descriptions": {{"维度1": "20-50字描述", "维度2": "20-50字描述"}}
    }}
  }}
}}
```

要求：只描述可见特征，精简、客观。请直接输出JSON。
"""
    
    def run_batch_with_images(self, input_data: dict) -> dict:
        """
        批量执行候选描述（一次性发送所有图片）
        
        Args:
            input_data: 包含 query_image, candidate_images, dimensions 的字典
                - query_image: 查询图片路径
                - candidate_images: 候选图片字典 {candidate_id: image_path}
                - dimensions: 评估维度列表
            
        Returns:
            dict: 包含所有候选物品描述的字典 {candidate_id: {descriptions: {...}}}
        """
        self._log_execution("batch_input", input_data)
        
        # 提取输入
        query_image = input_data.get("query_image", "")
        candidate_images = input_data.get("candidate_images", {})
        dimensions = input_data.get("dimensions", [])
        
        # 验证图片存在
        if not os.path.exists(query_image):
            raise FileNotFoundError(f"查询图片不存在: {query_image}")
        
        for cid, img_path in candidate_images.items():
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"候选物品图片不存在: {cid} -> {img_path}")
        
        # 格式化维度列表
        dimensions_formatted = self._format_dimensions(dimensions)
        
        # 格式化候选物品列表
        candidate_list = "\n".join([
            f"- 图片 {i+2}: {cid}" 
            for i, cid in enumerate(candidate_images.keys())
        ])
        
        # 构造 prompt
        prompt = self.PROMPT_TEMPLATE_BATCH.format(
            candidate_list=candidate_list,
            dimensions_formatted=dimensions_formatted
        )
        
        self._log_execution("batch_prompt_constructed", {"prompt_length": len(prompt)})
        
        # 调用 LLM（一次性发送所有图片），带重试机制
        max_retries = 3
        last_error = None
        
        for retry in range(max_retries):
            try:
                response = call_llm_with_all_images(
                    prompt=prompt,
                    query_image=query_image,
                    candidate_images=candidate_images,
                    expected_format="json"
                )
                
                self._log_execution("batch_llm_response", {"response_length": len(response), "retry": retry})
                
                # 解析响应
                result = self._parse_json_response(response)
                
                # 提取批量描述结果
                batch_descriptions = result.get("batch_descriptions", {})
                
                # 确保每个 candidate_id 正确
                for cid in candidate_images.keys():
                    if cid in batch_descriptions:
                        batch_descriptions[cid]["candidate_id"] = cid
                    else:
                        # 如果某个候选物品没有描述，创建空描述
                        batch_descriptions[cid] = {
                            "candidate_id": cid,
                            "descriptions": {}
                        }
                
                self._log_execution("batch_output", {"candidates_described": len(batch_descriptions)})
                
                return batch_descriptions
                
            except (json.JSONDecodeError, ValueError) as e:
                last_error = e
                print(f"[DescriptorSkill] 批量描述解析失败 (重试 {retry + 1}/{max_retries}): {str(e)[:200]}")
                if retry < max_retries - 1:
                    import time
                    time.sleep(1)  # 等待1秒后重试
        
        # 所有重试都失败
        raise ValueError(f"批量描述在 {max_retries} 次重试后仍然失败: {last_error}")


class JudgeSkill(BaseSkill):
    """
    相似度判断 Skill
    
    功能：判断 query 与 candidate 在每个维度上的相似度。
    支持文本和图片两种模式。
    """
    
    PROMPT_TEMPLATE = """你是相似度评判专家，评估候选物品与查询目标的相似度。

## 输入
用户查询：{query}
候选ID：{candidate_id}

候选物品各维度描述：
{candidate_descriptions_formatted}

评估维度及评分标准：
{dimensions_formatted}

## 输出格式（精简模式）

```json
{{
  "candidate_id": "{candidate_id}",
  "scores": {{"维度1": 0.8, "维度2": 0.6}}
}}
```

评分指南：1.0=完全匹配, 0.8=高度相似, 0.6=中等, 0.4=部分相似, 0.2=很低
请直接输出JSON，只输出分数不需要理由。
"""

    # 基于图片的 prompt 模板
    PROMPT_TEMPLATE_WITH_IMAGES = """你是视觉相似度评判专家，基于图片评估候选物品与查询目标的相似度。

## 图片说明
- 图片1：查询图片（目标物品）
- 图片2：候选物品图片

候选ID：{candidate_id}

## 评估维度及评分标准
{dimensions_formatted}

## 输出格式（精简模式）

```json
{{
  "candidate_id": "{candidate_id}",
  "scores": {{"维度1": 0.8, "维度2": 0.6}}
}}
```

评分指南：1.0=完全匹配, 0.8=高度相似, 0.6=中等, 0.4=部分相似, 0.2=很低
请直接输出JSON，只输出分数不需要理由。
"""
    
    def __init__(self):
        super().__init__(
            name="JudgeSkill",
            description="相似度判断技能：评估候选物品与查询目标在各维度上的相似度"
        )
    
    def get_prompt_template(self) -> str:
        return self.PROMPT_TEMPLATE
    
    def get_input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "candidate_id": {"type": "string"},
                "candidate_descriptions": {"type": "object"},
                "dimensions": {"type": "array"}
            },
            "required": ["query", "candidate_id", "candidate_descriptions", "dimensions"]
        }
    
    def get_output_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "candidate_id": {"type": "string"},
                "scores": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {
                            "score": {"type": "number"},
                            "reason": {"type": "string"}
                        }
                    }
                }
            }
        }
    
    def _format_candidate_descriptions(self, descriptions: dict) -> str:
        """格式化候选物品描述"""
        formatted_parts = []
        for dim_name, desc in descriptions.items():
            formatted_parts.append(f"- **{dim_name}**：{desc}")
        return "\n".join(formatted_parts)
    
    def _format_dimensions_with_criteria(self, dimensions: list) -> str:
        """格式化维度及其评分标准"""
        formatted_parts = []
        for i, dim in enumerate(dimensions, 1):
            formatted_parts.append(f"""
{i}. **{dim['name']}**（权重：{dim.get('weight', 0):.2f}）
   - 描述：{dim['description']}
   - 评分标准：{dim.get('scoring_criteria', '根据相似程度给出 0-1 分')}""")
        return "\n".join(formatted_parts)
    
    def run(self, input_data: dict) -> dict:
        """
        执行相似度判断（文本模式）
        
        Args:
            input_data: 包含 query, candidate_id, candidate_descriptions, dimensions 的字典
            
        Returns:
            dict: 包含 candidate_id 和 scores 的结果
        """
        self._log_execution("input", input_data)
        
        # 提取输入
        query = input_data.get("query", "")
        candidate_id = input_data.get("candidate_id", "")
        candidate_descriptions = input_data.get("candidate_descriptions", {})
        dimensions = input_data.get("dimensions", [])
        
        # 格式化描述和维度
        candidate_descriptions_formatted = self._format_candidate_descriptions(candidate_descriptions)
        dimensions_formatted = self._format_dimensions_with_criteria(dimensions)
        
        # 构造 prompt
        prompt = self.PROMPT_TEMPLATE.format(
            query=query,
            candidate_id=candidate_id,
            candidate_descriptions_formatted=candidate_descriptions_formatted,
            dimensions_formatted=dimensions_formatted
        )
        
        self._log_execution("prompt_constructed", {"prompt_length": len(prompt)})
        
        # 调用 LLM
        response = call_llm(prompt, expected_format="json")
        
        self._log_execution("llm_response", {"response_length": len(response)})
        
        # 解析响应
        result = self._parse_json_response(response)
        
        # 确保 candidate_id 正确
        result["candidate_id"] = candidate_id
        
        # 验证分数范围
        self._validate_scores(result.get("scores", {}))
        
        self._log_execution("output", result)
        
        return result
    
    def run_with_images(self, input_data: dict) -> dict:
        """
        执行相似度判断（图片模式）
        
        Args:
            input_data: 包含 query_image, candidate_image, candidate_id, dimensions 的字典
            
        Returns:
            dict: 包含 candidate_id 和 scores 的结果
        """
        self._log_execution("input", input_data)
        
        # 提取输入
        query_image = input_data.get("query_image", "")  # 查询图片路径
        candidate_image = input_data.get("candidate_image", "")  # 候选图片路径
        candidate_id = input_data.get("candidate_id", "")
        dimensions = input_data.get("dimensions", [])
        
        # 验证图片存在
        if not os.path.exists(query_image):
            raise FileNotFoundError(f"查询图片不存在: {query_image}")
        if not os.path.exists(candidate_image):
            raise FileNotFoundError(f"候选图片不存在: {candidate_image}")
        
        # 格式化维度
        dimensions_formatted = self._format_dimensions_with_criteria(dimensions)
        
        # 构造 prompt
        prompt = self.PROMPT_TEMPLATE_WITH_IMAGES.format(
            candidate_id=candidate_id,
            dimensions_formatted=dimensions_formatted
        )
        
        self._log_execution("prompt_constructed", {"prompt_length": len(prompt)})
        
        # 调用 LLM（带两张图片：query 和 candidate）
        image_paths = [query_image, candidate_image]
        response = call_llm_with_images(prompt, image_paths, expected_format="json")
        
        self._log_execution("llm_response", {"response_length": len(response)})
        
        # 解析响应
        result = self._parse_json_response(response)
        
        # 确保 candidate_id 正确
        result["candidate_id"] = candidate_id
        
        # 验证分数范围
        self._validate_scores(result.get("scores", {}))
        
        self._log_execution("output", result)
        
        return result
    
    def _validate_scores(self, scores: dict) -> None:
        """验证分数是否在有效范围内，并将简化格式转换为标准格式"""
        for dim_name, score_data in list(scores.items()):
            # 处理简化格式：分数直接是数字
            if isinstance(score_data, (int, float)):
                score = float(score_data)
                if score < 0:
                    score = 0
                elif score > 1:
                    score = 1
                scores[dim_name] = {"score": score, "reason": ""}
            # 处理标准格式：分数是对象
            elif isinstance(score_data, dict) and "score" in score_data:
                score = score_data["score"]
                if score < 0:
                    score_data["score"] = 0
                elif score > 1:
                    score_data["score"] = 1
    
    # 批量处理的 prompt 模板（纯文本描述版本，不需要图片）
    PROMPT_TEMPLATE_BATCH_TEXT_ONLY = """你是一个专业的相似度评判专家，负责基于预先生成的文本描述批量评估多个候选物品与查询目标之间的相似度。

## 任务描述
你将根据以下信息进行评分：
1. **查询物品的分析描述**：描述了用户想要查找的目标物品的特征
2. **各候选物品的特征描述**：每个候选物品在各个评估维度上的特征描述
3. **评估维度及评分标准**：你必须严格按照这些标准进行评分

## 输入信息

### 查询物品分析
{query_analysis}

### 候选物品列表及其特征描述
{candidate_descriptions}

### 评估维度及评分标准
{dimensions_with_criteria}

## 评判要求

对于每个候选物品的每个维度，严格依据评分标准给出 0-1 之间的分数：
- 1.0 = 完全匹配（5分标准）
- 0.8 = 高度匹配（4分标准）
- 0.6 = 中等匹配（3分标准）
- 0.4 = 低度匹配（2分标准）
- 0.2 = 几乎不匹配（1分标准）

## 输出格式（极简模式，必须严格遵守）

**重要**：为了确保输出完整，请使用以下精简格式，每个维度只输出分数，不需要理由！

```json
{{
  "batch_scores": {{
    "candidate_id_1": {{
      "scores": {{"维度1": 0.8, "维度2": 0.6, "维度3": 0.7}}
    }},
    "candidate_id_2": {{
      "scores": {{"维度1": 0.5, "维度2": 0.4, "维度3": 0.6}}
    }}
  }}
}}
```

## 注意事项
- **输出必须精简**：只输出分数，不要输出理由，以确保JSON完整
- 分数精确到小数点后一位即可（如0.8, 0.6）
- 请确保为所有候选物品的所有维度都生成评分
- 如果候选描述中某维度缺失，给予0.5分
- **确保JSON格式完整**，所有括号必须正确闭合

请开始评分，直接输出JSON，不要有任何其他文字。
"""
    
    def run_batch_with_images(self, input_data: dict) -> dict:
        """
        批量执行相似度判断（一次性发送所有图片）
        
        Args:
            input_data: 包含 query_image, candidate_images, dimensions, candidate_descriptions 的字典
                - query_image: 查询图片路径
                - candidate_images: 候选图片字典 {candidate_id: image_path}
                - dimensions: 评估维度列表
                - candidate_descriptions: 候选物品描述字典（可选，来自阶段2的输出）
            
        Returns:
            dict: 包含所有候选物品评分的字典 {candidate_id: {scores: {...}}}
        """
        self._log_execution("batch_input", input_data)
        
        # 提取输入
        query_image = input_data.get("query_image", "")
        candidate_images = input_data.get("candidate_images", {})
        dimensions = input_data.get("dimensions", [])
        candidate_descriptions = input_data.get("candidate_descriptions", {})
        
        # 验证图片存在
        if not os.path.exists(query_image):
            raise FileNotFoundError(f"查询图片不存在: {query_image}")
        
        for cid, img_path in candidate_images.items():
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"候选物品图片不存在: {cid} -> {img_path}")
        
        # 格式化维度
        dimensions_formatted = self._format_dimensions_with_criteria(dimensions)
        
        # 格式化候选物品列表
        candidate_list = "\n".join([
            f"- 图片 {i+2}: {cid}" 
            for i, cid in enumerate(candidate_images.keys())
        ])
        
        # 格式化候选物品描述（如果有的话）
        if candidate_descriptions:
            desc_lines = []
            for cid, desc_data in candidate_descriptions.items():
                descriptions = desc_data.get("descriptions", {})
                if descriptions:
                    desc_lines.append(f"\n**{cid}**:")
                    for dim_name, dim_desc in descriptions.items():
                        # 截取前200字符避免过长
                        short_desc = dim_desc[:200] + "..." if len(dim_desc) > 200 else dim_desc
                        desc_lines.append(f"  - {dim_name}: {short_desc}")
            candidate_descriptions_formatted = "\n".join(desc_lines) if desc_lines else "（无预生成描述）"
        else:
            candidate_descriptions_formatted = "（无预生成描述）"
        
        # 构造 prompt
        prompt = self.PROMPT_TEMPLATE_BATCH.format(
            candidate_list=candidate_list,
            candidate_descriptions=candidate_descriptions_formatted,
            dimensions_formatted=dimensions_formatted
        )
        
        self._log_execution("batch_prompt_constructed", {"prompt_length": len(prompt)})
        
        # 调用 LLM（一次性发送所有图片），带重试机制
        max_retries = 3
        last_error = None
        
        for retry in range(max_retries):
            try:
                response = call_llm_with_all_images(
                    prompt=prompt,
                    query_image=query_image,
                    candidate_images=candidate_images,
                    expected_format="json"
                )
                
                self._log_execution("batch_llm_response", {"response_length": len(response), "retry": retry})
                
                # 解析响应
                result = self._parse_json_response(response)
                
                # 提取批量评分结果
                batch_scores = result.get("batch_scores", {})
        
                # 验证并修正每个候选物品的评分
                for cid in candidate_images.keys():
                    if cid in batch_scores:
                        batch_scores[cid]["candidate_id"] = cid
                        self._validate_scores(batch_scores[cid].get("scores", {}))
                    else:
                        # 如果某个候选物品没有评分，创建空评分
                        batch_scores[cid] = {
                            "candidate_id": cid,
                            "scores": {}
                        }
                
                self._log_execution("batch_output", {"candidates_judged": len(batch_scores)})
                
                return batch_scores
                
            except (json.JSONDecodeError, ValueError) as e:
                last_error = e
                print(f"[JudgeSkill] 批量评分解析失败 (重试 {retry + 1}/{max_retries}): {str(e)[:200]}")
                if retry < max_retries - 1:
                    import time
                    time.sleep(1)  # 等待1秒后重试
        
        # 所有重试都失败
        raise ValueError(f"批量评分在 {max_retries} 次重试后仍然失败: {last_error}")

    def run_batch_with_text_only(self, input_data: dict) -> dict:
        """
        批量执行相似度判断（纯文本模式，不需要图片）
        
        只使用第二阶段生成的文本描述进行评分，不输入任何图像。
        严格按照第一阶段产生的评分标准（scoring_criteria）进行打分。
        
        Args:
            input_data: 包含以下内容的字典
                - query_analysis: 查询物品的分析描述（来自阶段1）
                - candidate_descriptions: 候选物品描述字典（来自阶段2的输出）
                  格式：{candidate_id: {"descriptions": {dim_name: description, ...}}}
                - dimensions: 评估维度列表（来自阶段1，包含评分标准）
            
        Returns:
            dict: 包含所有候选物品评分的字典 {candidate_id: {scores: {...}}}
        """
        self._log_execution("batch_text_only_input", input_data)
        
        # 提取输入
        query_analysis = input_data.get("query_analysis", "")
        candidate_descriptions = input_data.get("candidate_descriptions", {})
        dimensions = input_data.get("dimensions", [])
        
        if not candidate_descriptions:
            raise ValueError("候选物品描述不能为空")
        
        if not dimensions:
            raise ValueError("评估维度列表不能为空")
        
        # 格式化候选物品描述（完整版，不截断）
        desc_lines = []
        for cid, desc_data in candidate_descriptions.items():
            descriptions = desc_data.get("descriptions", {})
            if descriptions:
                desc_lines.append(f"\n### 候选物品: {cid}")
                for dim_name, dim_desc in descriptions.items():
                    desc_lines.append(f"**{dim_name}**: {dim_desc}")
        candidate_descriptions_formatted = "\n".join(desc_lines) if desc_lines else "（无候选描述）"
        
        # 格式化维度及其评分标准（包含完整的 scoring_criteria）
        dimensions_with_criteria_lines = []
        for i, dim in enumerate(dimensions, 1):
            dim_name = dim.get('name', f'维度{i}')
            dim_desc = dim.get('description', '')
            dim_weight = dim.get('weight', 0)
            scoring_criteria = dim.get('scoring_criteria', '根据相似程度给出 0-1 分')
            
            dimensions_with_criteria_lines.append(f"""
### {i}. {dim_name}（权重：{dim_weight:.2f}）
- **描述**：{dim_desc}
- **评分标准**：{scoring_criteria}
""")
        dimensions_with_criteria_formatted = "\n".join(dimensions_with_criteria_lines)
        
        # 构造 prompt
        prompt = self.PROMPT_TEMPLATE_BATCH_TEXT_ONLY.format(
            query_analysis=query_analysis,
            candidate_descriptions=candidate_descriptions_formatted,
            dimensions_with_criteria=dimensions_with_criteria_formatted
        )
        
        self._log_execution("batch_text_only_prompt_constructed", {"prompt_length": len(prompt)})
        
        # 调用 LLM（纯文本模式，不发送图片），带重试机制
        max_retries = 3
        last_error = None
        
        for retry in range(max_retries):
            try:
                # 调用纯文本LLM接口
                response = call_llm(prompt, expected_format="json")
                
                self._log_execution("batch_text_only_llm_response", {"response_length": len(response), "retry": retry})
                
                # 尝试解析响应（带截断修复）
                result = self._parse_json_response_with_recovery(response)
                
                # 提取批量评分结果
                batch_scores = result.get("batch_scores", {})
        
                # 验证并修正每个候选物品的评分（支持简化格式）
                for cid in candidate_descriptions.keys():
                    if cid in batch_scores:
                        batch_scores[cid]["candidate_id"] = cid
                        # 处理简化格式：将直接的分数转换为带score的对象
                        scores = batch_scores[cid].get("scores", {})
                        normalized_scores = {}
                        for dim_name, score_value in scores.items():
                            if isinstance(score_value, (int, float)):
                                # 简化格式：分数直接是数字
                                normalized_scores[dim_name] = {
                                    "score": float(score_value),
                                    "reason": ""
                                }
                            elif isinstance(score_value, dict):
                                # 完整格式：分数是对象
                                normalized_scores[dim_name] = score_value
                            else:
                                normalized_scores[dim_name] = {"score": 0.5, "reason": ""}
                        batch_scores[cid]["scores"] = normalized_scores
                        self._validate_scores(batch_scores[cid].get("scores", {}))
                    else:
                        # 如果某个候选物品没有评分，创建空评分
                        batch_scores[cid] = {
                            "candidate_id": cid,
                            "scores": {}
                        }
                
                self._log_execution("batch_text_only_output", {"candidates_judged": len(batch_scores)})
                
                return batch_scores
                
            except (json.JSONDecodeError, ValueError) as e:
                last_error = e
                print(f"[JudgeSkill] 纯文本批量评分解析失败 (重试 {retry + 1}/{max_retries}): {str(e)[:200]}")
                if retry < max_retries - 1:
                    import time
                    time.sleep(1)  # 等待1秒后重试
        
        # 所有重试都失败
        raise ValueError(f"纯文本批量评分在 {max_retries} 次重试后仍然失败: {last_error}")
    
    def _parse_json_response_with_recovery(self, response: str) -> dict:
        """
        解析JSON响应，带有截断修复功能
        
        Args:
            response: LLM的原始响应
            
        Returns:
            dict: 解析后的JSON对象
        """
        # 首先尝试正常解析
        try:
            return self._parse_json_response(response)
        except (json.JSONDecodeError, ValueError) as original_error:
            print(f"[JudgeSkill] JSON解析失败，尝试修复截断的JSON...")
            
            # 提取JSON部分
            json_str = response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            
            # 尝试修复截断的JSON
            repaired_json = self._try_repair_truncated_json(json_str)
            if repaired_json:
                try:
                    result = json.loads(repaired_json)
                    print(f"[JudgeSkill] JSON修复成功！")
                    return result
                except json.JSONDecodeError:
                    pass
            
            # 如果修复失败，抛出原始错误
            raise original_error
    
    def _try_repair_truncated_json(self, json_str: str) -> str:
        """
        尝试修复截断的JSON字符串
        
        Args:
            json_str: 可能被截断的JSON字符串
            
        Returns:
            str: 修复后的JSON字符串，如果无法修复则返回None
        """
        if not json_str:
            return None
        
        # 统计未闭合的括号
        stack = []
        in_string = False
        escape_next = False
        
        for char in json_str:
            if escape_next:
                escape_next = False
                continue
            
            if char == '\\':
                escape_next = True
                continue
                
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            
            if in_string:
                continue
            
            if char in '{[':
                stack.append(char)
            elif char == '}':
                if stack and stack[-1] == '{':
                    stack.pop()
            elif char == ']':
                if stack and stack[-1] == '[':
                    stack.pop()
        
        # 如果在字符串中被截断，先关闭字符串
        if in_string:
            json_str += '"'
        
        # 添加缺失的闭合括号
        closing_chars = []
        for bracket in reversed(stack):
            if bracket == '{':
                closing_chars.append('}')
            elif bracket == '[':
                closing_chars.append(']')
        
        if closing_chars:
            # 处理可能的不完整值（如 "score": 0. 或 "score": 1.00,）
            # 去除末尾的逗号和不完整的数值
            json_str = json_str.rstrip()
            while json_str and json_str[-1] in ',.:':
                json_str = json_str[:-1].rstrip()
            
            json_str += ''.join(closing_chars)
        
        return json_str


class ValidateSkill(BaseSkill):
    # ... existing code ...
    """
    排序与一致性校验 Skill
    
    功能：
    1. 根据维度权重，对各 candidate 进行加权汇总
    2. 生成初始倒序排序
    3. 执行全局一致性校验
    4. 若发现问题，给出修正理由并返回修正后的排序
    """
    
    PROMPT_TEMPLATE = """你是排序验证专家，负责验证和校正候选物品的排序结果。

## 输入
评估维度及权重：
{dimensions_formatted}

各候选物品评分：
{all_scores_formatted}

加权总分：
{weighted_scores_formatted}

## 验证任务
1. 验证排序是否符合相似度递减趋势
2. 检查是否存在语义反转
3. 检查相邻排名的分数差距是否合理

## 输出格式（精简模式）

```json
{{
  "initial_ranking": ["id1", "id2", ...],
  "final_ranking": ["id1", "id2", ...],
  "confidence_score": 0.XX,
  "adjustments_count": 0
}}
```

注意：只有在发现明确问题时才调整排序。请直接输出JSON。
"""
    
    def __init__(self):
        super().__init__(
            name="ValidateSkill",
            description="排序校验技能：验证排序结果的一致性并进行必要的修正"
        )
    
    def get_prompt_template(self) -> str:
        return self.PROMPT_TEMPLATE
    
    def get_input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "all_candidate_scores": {
                    "type": "object",
                    "description": "所有候选物品的维度评分"
                },
                "dimensions": {
                    "type": "array",
                    "description": "评估维度列表（含权重）"
                }
            },
            "required": ["all_candidate_scores", "dimensions"]
        }
    
    def get_output_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "initial_ranking": {"type": "array"},
                "validation_checks": {"type": "array"},
                "adjustments_made": {"type": "array"},
                "final_ranking": {"type": "array"},
                "validation_notes": {"type": "string"},
                "confidence_score": {"type": "number"}
            }
        }
    
    def _format_dimensions(self, dimensions: list) -> str:
        """格式化维度列表"""
        formatted_parts = []
        for dim in dimensions:
            formatted_parts.append(f"- **{dim['name']}**：权重 {dim.get('weight', 0):.2f}")
        return "\n".join(formatted_parts)
    
    def _format_all_scores(self, all_scores: dict, dimensions: list) -> str:
        """格式化所有候选物品的评分"""
        formatted_parts = []
        for candidate_id, score_data in all_scores.items():
            formatted_parts.append(f"\n### {candidate_id}")
            dim_scores = score_data.get("dimension_scores", score_data.get("scores", {}))
            for dim in dimensions:
                dim_name = dim["name"]
                if dim_name in dim_scores:
                    score_info = dim_scores[dim_name]
                    if isinstance(score_info, dict):
                        score = score_info.get("score", 0)
                        reason = score_info.get("reason", "")
                    else:
                        score = score_info
                        reason = ""
                    formatted_parts.append(f"- {dim_name}: {score:.2f}" + (f" ({reason[:50]}...)" if reason else ""))
        return "\n".join(formatted_parts)
    
    def _calculate_weighted_scores(self, all_scores: dict, dimensions: list) -> dict:
        """计算加权总分"""
        weighted_scores = {}
        
        for candidate_id, score_data in all_scores.items():
            dim_scores = score_data.get("dimension_scores", score_data.get("scores", {}))
            total_score = 0
            score_breakdown = []
            
            for dim in dimensions:
                dim_name = dim["name"]
                weight = dim.get("weight", 0)
                
                if dim_name in dim_scores:
                    score_info = dim_scores[dim_name]
                    if isinstance(score_info, dict):
                        score = score_info.get("score", 0)
                    else:
                        score = score_info
                    
                    weighted_contribution = score * weight
                    total_score += weighted_contribution
                    score_breakdown.append({
                        "dimension": dim_name,
                        "score": score,
                        "weight": weight,
                        "contribution": weighted_contribution
                    })
            
            weighted_scores[candidate_id] = {
                "total_score": round(total_score, 4),
                "breakdown": score_breakdown
            }
        
        return weighted_scores
    
    def _format_weighted_scores(self, weighted_scores: dict) -> str:
        """格式化加权总分"""
        # 按总分排序
        sorted_scores = sorted(
            weighted_scores.items(),
            key=lambda x: x[1]["total_score"],
            reverse=True
        )
        
        formatted_parts = []
        for rank, (candidate_id, score_data) in enumerate(sorted_scores, 1):
            formatted_parts.append(f"{rank}. **{candidate_id}**：总分 {score_data['total_score']:.4f}")
        
        return "\n".join(formatted_parts)
    
    def run(self, input_data: dict) -> dict:
        """
        执行排序验证
        
        Args:
            input_data: 包含 all_candidate_scores 和 dimensions 的字典
            
        Returns:
            dict: 包含验证结果和最终排序的字典
        """
        self._log_execution("input", input_data)
        
        # 提取输入
        all_candidate_scores = input_data.get("all_candidate_scores", {})
        dimensions = input_data.get("dimensions", [])
        
        # 计算加权总分
        weighted_scores = self._calculate_weighted_scores(all_candidate_scores, dimensions)
        
        self._log_execution("weighted_scores", weighted_scores)
        
        # 格式化各部分
        dimensions_formatted = self._format_dimensions(dimensions)
        all_scores_formatted = self._format_all_scores(all_candidate_scores, dimensions)
        weighted_scores_formatted = self._format_weighted_scores(weighted_scores)
        
        # 构造 prompt
        prompt = self.PROMPT_TEMPLATE.format(
            dimensions_formatted=dimensions_formatted,
            all_scores_formatted=all_scores_formatted,
            weighted_scores_formatted=weighted_scores_formatted
        )
        
        self._log_execution("prompt_constructed", {"prompt_length": len(prompt)})
        
        # 调用 LLM
        response = call_llm(prompt, expected_format="json")
        
        self._log_execution("llm_response", {"response_length": len(response)})
        
        # 解析响应
        result = self._parse_json_response(response)
        
        # 如果 LLM 没有返回完整的排序，使用计算的排序
        if not result.get("final_ranking"):
            sorted_candidates = sorted(
                weighted_scores.keys(),
                key=lambda x: weighted_scores[x]["total_score"],
                reverse=True
            )
            result["final_ranking"] = sorted_candidates
            result["initial_ranking"] = sorted_candidates
        
        # 添加加权分数详情
        result["weighted_scores"] = weighted_scores
        
        self._log_execution("output", result)
        
        return result


# 导出所有 Skill 类
__all__ = [
    "BaseSkill",
    "DimensionPlannerSkill",
    "DescriptorSkill",
    "JudgeSkill",
    "ValidateSkill"
]
