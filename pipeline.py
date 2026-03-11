"""
排序流水线模块

实现 RankingPipeline 类，负责协调所有 Skills 的执行。
支持文本模式和图片模式两种输入方式。
支持一次性发送所有图像和中间结果缓存功能。
"""

import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime

from dimension_bank import get_dimension_bank
from skills import (
    DimensionPlannerSkill,
    DescriptorSkill,
    JudgeSkill,
    ValidateSkill
)
from llm_interface import get_llm_interface, get_result_cache, get_persistent_cache


class RankingPipeline:
    """
    排序流水线
    
    负责协调所有 Skills 的顺序执行，完成从 query 到最终排序的完整流程。
    支持文本模式和图片模式两种输入方式。
    支持一次性发送所有图像和中间结果缓存功能。
    
    执行流程：
    1. DimensionPlannerSkill: 规划评估维度
    2. DescriptorSkill: 对所有候选物品生成描述
    3. JudgeSkill: 对所有候选物品进行相似度评判
    4. ValidateSkill: 汇总结果并验证排序
    """
    
    def __init__(
        self, 
        dimension_bank: Optional[dict] = None, 
        verbose: bool = True,
        use_batch_mode: bool = True,
        use_cache: bool = True,
        source_json_path: str = None,
        skip_validation: bool = True
    ):
        """
        初始化排序流水线
        
        Args:
            dimension_bank: 维度银行（可选，默认使用内置维度银行）
            verbose: 是否输出详细日志
            use_batch_mode: 是否使用批量模式（一次性发送所有图像）
            use_cache: 是否使用中间结果缓存
            source_json_path: 源 JSON 文件路径（用于持久化缓存，如果提供则启用阶段级持久化缓存）
            skip_validation: 是否跳过验证步骤（默认 True，直接使用加权分数排序）
        """
        self.dimension_bank = dimension_bank or get_dimension_bank()
        self.verbose = verbose
        self.use_batch_mode = use_batch_mode
        self.use_cache = use_cache
        self.source_json_path = source_json_path
        self.skip_validation = skip_validation
        
        # 初始化所有 Skills
        self.dimension_planner = DimensionPlannerSkill()
        self.descriptor = DescriptorSkill()
        self.judge = JudgeSkill()
        self.validator = ValidateSkill()
        
        # 执行记录
        self.execution_log = []
        self.intermediate_results = {}
        
        # 获取结果缓存（内存缓存）
        self._result_cache = get_result_cache() if use_cache else None
        
        # 获取持久化缓存（基于 JSON 文件路径）
        self._persistent_cache = get_persistent_cache() if source_json_path else None
        
        if source_json_path:
            self._log(f"📁 已启用阶段级持久化缓存，源文件: {os.path.basename(source_json_path)}")
    
    def _log(self, message: str, data: Any = None):
        """记录日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = {
            "timestamp": timestamp,
            "message": message,
            "data": data
        }
        self.execution_log.append(log_entry)
        
        if self.verbose:
            print(f"[{timestamp}] {message}")
            if data and isinstance(data, dict):
                # 只打印摘要信息
                summary = self._summarize_data(data)
                if summary:
                    print(f"    -> {summary}")
    
    def _summarize_data(self, data: dict) -> str:
        """生成数据摘要"""
        if "dimensions" in data:
            return f"维度数量: {len(data['dimensions'])}"
        elif "descriptions" in data:
            return f"描述维度数量: {len(data['descriptions'])}"
        elif "scores" in data:
            return f"评分维度数量: {len(data['scores'])}"
        elif "final_ranking" in data:
            return f"最终排序: {data['final_ranking']}"
        return ""
    
    def _get_cached_result(self, skill_name: str, input_data: dict) -> Optional[dict]:
        """从缓存获取结果（先检查持久化缓存，再检查内存缓存）"""
        # 先检查持久化缓存
        if self._persistent_cache and self.source_json_path:
            result = self._persistent_cache.get_stage(self.source_json_path, skill_name)
            if result is not None:
                return result
        
        # 再检查内存缓存
        if not self.use_cache or self._result_cache is None:
            return None
        return self._result_cache.get(skill_name, input_data)
    
    def _cache_result(self, skill_name: str, input_data: dict, result: dict) -> None:
        """缓存结果（同时保存到内存缓存和持久化缓存）"""
        # 保存到内存缓存
        if self.use_cache and self._result_cache is not None:
            self._result_cache.set(skill_name, input_data, result)
        
        # 保存到持久化缓存（立即保存，即使后续步骤失败也能恢复）
        if self._persistent_cache and self.source_json_path:
            self._persistent_cache.set_stage(self.source_json_path, skill_name, result)
    
    def _calculate_ranking_from_scores(
        self, 
        all_scores: dict, 
        dimensions: List[dict]
    ) -> dict:
        """
        根据评分直接计算加权总分和排序（跳过验证步骤时使用）
        
        Args:
            all_scores: 所有候选物品的评分 {candidate_id: {"dimension_scores": {...}}}
            dimensions: 维度列表，包含权重信息
            
        Returns:
            dict: 包含 final_ranking 和 weighted_scores
        """
        weighted_scores = {}
        
        for candidate_id, score_data in all_scores.items():
            dim_scores = score_data.get("dimension_scores", score_data.get("scores", {}))
            total_score = 0.0
            dim_weighted = {}
            
            for dim in dimensions:
                dim_name = dim["name"]
                weight = dim.get("weight", 0)
                
                if dim_name in dim_scores:
                    score_info = dim_scores[dim_name]
                    # 支持简化格式（分数直接是数字）和标准格式（分数是对象）
                    if isinstance(score_info, (int, float)):
                        score = float(score_info)
                    elif isinstance(score_info, dict):
                        score = score_info.get("score", 0)
                    else:
                        score = 0
                    
                    weighted_score = score * weight
                    total_score += weighted_score
                    dim_weighted[dim_name] = {
                        "score": score,
                        "weight": weight,
                        "weighted_score": weighted_score
                    }
            
            weighted_scores[candidate_id] = {
                "total_score": total_score,
                "dimension_weighted_scores": dim_weighted
            }
        
        # 按总分降序排序
        final_ranking = sorted(
            weighted_scores.keys(),
            key=lambda x: weighted_scores[x]["total_score"],
            reverse=True
        )
        
        return {
            "final_ranking": final_ranking,
            "weighted_scores": weighted_scores,
            "initial_ranking": final_ranking,  # 无验证时，初始排序即最终排序
            "confidence_score": 1.0,  # 无验证时默认置信度为1
            "adjustments_count": 0
        }
    
    def run(self, query: str, candidate_ids: List[str], candidate_info: Optional[Dict[str, str]] = None) -> dict:
        """
        执行完整的排序流水线（文本模式）
        
        Args:
            query: 用户查询，包含目标物品描述和使用场景
            candidate_ids: 候选物品 ID 列表
            candidate_info: 候选物品附加信息（可选）
            
        Returns:
            dict: 包含完整排序结果和解释路径的字典
        """
        self._log("=" * 60)
        self._log("开始执行排序流水线（文本模式）")
        self._log(f"Query: {query}")
        self._log(f"候选物品数量: {len(candidate_ids)}")
        self._log("=" * 60)
        
        candidate_info = candidate_info or {}
        
        # ============ 阶段 1: 维度规划 ============
        self._log("\n>>> 阶段 1: 执行维度规划 (DimensionPlannerSkill)")
        
        dimension_input = {
            "query": query,
            "dimension_bank": self.dimension_bank
        }
        
        # 尝试从缓存获取
        dimension_result = self._get_cached_result("DimensionPlannerSkill", dimension_input)
        if dimension_result is None:
            dimension_result = self.dimension_planner.run(dimension_input)
            self._cache_result("DimensionPlannerSkill", dimension_input, dimension_result)
        
        self.intermediate_results["dimension_planning"] = dimension_result
        self._log("维度规划完成", dimension_result)
        
        dimensions = dimension_result.get("dimensions", [])
        inferred_scenario = dimension_result.get("inferred_scenario", "")
        
        self._log(f"推断场景: {inferred_scenario}")
        self._log(f"规划维度: {[d['name'] for d in dimensions]}")
        
        # ============ 阶段 2: 候选描述 ============
        self._log("\n>>> 阶段 2: 执行候选描述 (DescriptorSkill)")
        
        all_descriptions = {}
        for candidate_id in candidate_ids:
            self._log(f"  - 描述候选物品: {candidate_id}")
            
            descriptor_input = {
                "candidate_id": candidate_id,
                "candidate_info": candidate_info.get(candidate_id, ""),
                "dimensions": dimensions
            }
            
            # 尝试从缓存获取
            descriptor_result = self._get_cached_result("DescriptorSkill", descriptor_input)
            if descriptor_result is None:
                descriptor_result = self.descriptor.run(descriptor_input)
                self._cache_result("DescriptorSkill", descriptor_input, descriptor_result)
            
            all_descriptions[candidate_id] = descriptor_result
        
        self.intermediate_results["descriptions"] = all_descriptions
        self._log("候选描述完成", {"candidates_described": len(all_descriptions)})
        
        # ============ 阶段 3: 相似度判断 ============
        self._log("\n>>> 阶段 3: 执行相似度判断 (JudgeSkill)")
        
        all_scores = {}
        for candidate_id in candidate_ids:
            self._log(f"  - 评判候选物品: {candidate_id}")
            
            candidate_descriptions = all_descriptions[candidate_id].get("descriptions", {})
            
            judge_input = {
                "query": query,
                "candidate_id": candidate_id,
                "candidate_descriptions": candidate_descriptions,
                "dimensions": dimensions
            }
            
            # 尝试从缓存获取
            judge_result = self._get_cached_result("JudgeSkill", judge_input)
            if judge_result is None:
                judge_result = self.judge.run(judge_input)
                self._cache_result("JudgeSkill", judge_input, judge_result)
            
            all_scores[candidate_id] = {
                "dimension_scores": judge_result.get("scores", {})
            }
        
        self.intermediate_results["scores"] = all_scores
        self._log("相似度判断完成", {"candidates_judged": len(all_scores)})
        
        # ============ 阶段 4: 排序验证（可选）============
        if self.skip_validation:
            self._log("\n>>> 阶段 4: 跳过排序验证，直接计算加权排序")
            validate_result = self._calculate_ranking_from_scores(all_scores, dimensions)
            self._log(f"加权排序完成: {validate_result['final_ranking']}")
        else:
            self._log("\n>>> 阶段 4: 执行排序验证 (ValidateSkill)")
            
            validate_input = {
                "all_candidate_scores": all_scores,
                "dimensions": dimensions
            }
            
            # 尝试从缓存获取
            validate_result = self._get_cached_result("ValidateSkill", validate_input)
            if validate_result is None:
                validate_result = self.validator.run(validate_input)
                self._cache_result("ValidateSkill", validate_input, validate_result)
            
            self._log("排序验证完成", validate_result)
        
        self.intermediate_results["validation"] = validate_result
        
        # ============ 构建最终结果 ============
        self._log("\n>>> 构建最终结果")
        
        final_result = self._build_final_result(
            query=query,
            candidate_ids=candidate_ids,
            dimension_result=dimension_result,
            all_descriptions=all_descriptions,
            all_scores=all_scores,
            validate_result=validate_result
        )
        
        self._log("=" * 60)
        self._log("排序流水线执行完成")
        self._log(f"最终排序: {final_result['final_ranking']}")
        self._log("=" * 60)
        
        return final_result
    
    def run_with_images(
        self, 
        query_image: str,
        candidate_images: Dict[str, str],
        query_description: str = ""
    ) -> dict:
        """
        执行完整的排序流水线（图片模式）
        
        Args:
            query_image: 查询图片路径
            candidate_images: 候选物品图片字典，格式为 {candidate_id: image_path}
            query_description: 用户对查询图片的补充描述（可选）
            
        Returns:
            dict: 包含完整排序结果和解释路径的字典
        """
        self._log("=" * 60)
        self._log("开始执行排序流水线（图片模式）")
        self._log(f"Query 图片: {query_image}")
        self._log(f"候选物品数量: {len(candidate_images)}")
        self._log(f"批量模式: {'启用' if self.use_batch_mode else '禁用'}")
        self._log(f"结果缓存: {'启用' if self.use_cache else '禁用'}")
        self._log("=" * 60)
        
        # 验证图片文件存在
        if not os.path.exists(query_image):
            raise FileNotFoundError(f"查询图片不存在: {query_image}")
        
        for cid, img_path in candidate_images.items():
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"候选物品图片不存在: {cid} -> {img_path}")
        
        candidate_ids = list(candidate_images.keys())
        
        # 预加载所有图像（如果使用批量模式）
        if self.use_batch_mode:
            self._log("\n>>> 预加载所有图像...")
            llm = get_llm_interface()
            all_images = [query_image] + list(candidate_images.values())
            llm.preload_images(all_images)
        
        # 根据批量模式选择执行方式
        if self.use_batch_mode:
            return self._run_with_images_batch_mode(
                query_image=query_image,
                candidate_images=candidate_images,
                query_description=query_description
            )
        else:
            return self._run_with_images_sequential_mode(
                query_image=query_image,
                candidate_images=candidate_images,
                query_description=query_description
            )
    
    def _run_with_images_batch_mode(
        self,
        query_image: str,
        candidate_images: Dict[str, str],
        query_description: str = ""
    ) -> dict:
        """
        执行排序流水线（图片模式 - 批量模式）
        
        在批量模式下，会一次性发送所有图像进行处理，减少 API 调用次数。
        """
        candidate_ids = list(candidate_images.keys())
        
        # ============ 阶段 1: 维度规划（基于查询图片）============
        self._log("\n>>> 阶段 1: 执行维度规划 (DimensionPlannerSkill) - 图片模式")
        
        dimension_input = {
            "query_image": query_image,
            "query_description": query_description,
            "dimension_bank": self.dimension_bank
        }
        
        # 构造缓存键（不包含图片路径，只包含图片特征）
        cache_key_input = {
            "query_image_name": os.path.basename(query_image),
            "query_description": query_description,
            "dimension_bank_hash": hash(str(self.dimension_bank))
        }
        
        # 尝试从缓存获取
        dimension_result = self._get_cached_result("DimensionPlannerSkill_Image", cache_key_input)
        if dimension_result is None:
            dimension_result = self.dimension_planner.run_with_image(dimension_input)
            self._cache_result("DimensionPlannerSkill_Image", cache_key_input, dimension_result)
        
        self.intermediate_results["dimension_planning"] = dimension_result
        self._log("维度规划完成", dimension_result)
        
        dimensions = dimension_result.get("dimensions", [])
        inferred_scenario = dimension_result.get("inferred_scenario", "")
        query_object_analysis = dimension_result.get("query_object_analysis", "")
        
        self._log(f"图片分析: {query_object_analysis}")
        self._log(f"推断场景: {inferred_scenario}")
        self._log(f"规划维度: {[d['name'] for d in dimensions]}")
        
        # ============ 阶段 2: 批量候选描述 ============
        self._log("\n>>> 阶段 2: 执行批量候选描述 (DescriptorSkill) - 图片模式")
        
        # 构造缓存键
        batch_desc_cache_key = {
            "candidate_ids": candidate_ids,
            "dimensions_hash": hash(str(dimensions))
        }
        
        # 尝试从缓存获取
        all_descriptions = self._get_cached_result("BatchDescriptorSkill", batch_desc_cache_key)
        if all_descriptions is None:
            all_descriptions = self.descriptor.run_batch_with_images({
                "query_image": query_image,
                "candidate_images": candidate_images,
                "dimensions": dimensions
            })
            self._cache_result("BatchDescriptorSkill", batch_desc_cache_key, all_descriptions)
        
        self.intermediate_results["descriptions"] = all_descriptions
        self._log("批量候选描述完成", {"candidates_described": len(all_descriptions)})
        
        # ============ 阶段 3: 批量相似度判断（纯文本模式）============
        self._log("\n>>> 阶段 3: 执行批量相似度判断 (JudgeSkill) - 纯文本模式（基于描述和评分标准）")
        
        # 构造缓存键
        batch_judge_cache_key = {
            "query_analysis_hash": hash(query_object_analysis),
            "candidate_ids": candidate_ids,
            "dimensions_hash": hash(str(dimensions)),
            "descriptions_hash": hash(str(all_descriptions))
        }
        
        # 尝试从缓存获取
        batch_scores = self._get_cached_result("BatchJudgeSkill", batch_judge_cache_key)
        if batch_scores is None:
            # 使用纯文本评分方法，只传入描述和评分标准，不传入图像
            batch_scores = self.judge.run_batch_with_text_only({
                "query_analysis": query_object_analysis,  # 阶段1产生的查询物品分析
                "candidate_descriptions": all_descriptions,  # 阶段2产生的候选描述
                "dimensions": dimensions  # 阶段1产生的维度及评分标准
            })
            self._cache_result("BatchJudgeSkill", batch_judge_cache_key, batch_scores)
        
        # 转换为标准格式
        all_scores = {}
        for candidate_id, score_data in batch_scores.items():
            all_scores[candidate_id] = {
                "dimension_scores": score_data.get("scores", {})
            }
        
        self.intermediate_results["scores"] = all_scores
        self._log("批量相似度判断完成", {"candidates_judged": len(all_scores)})
        
        # ============ 阶段 4: 排序验证（可选）============
        if self.skip_validation:
            self._log("\n>>> 阶段 4: 跳过排序验证，直接计算加权排序")
            validate_result = self._calculate_ranking_from_scores(all_scores, dimensions)
            self._log(f"加权排序完成: {validate_result['final_ranking']}")
        else:
            self._log("\n>>> 阶段 4: 执行排序验证 (ValidateSkill)")
            
            validate_input = {
                "all_candidate_scores": all_scores,
                "dimensions": dimensions
            }
            
            # 尝试从缓存获取
            validate_result = self._get_cached_result("ValidateSkill", validate_input)
            if validate_result is None:
                validate_result = self.validator.run(validate_input)
                self._cache_result("ValidateSkill", validate_input, validate_result)
            
            self._log("排序验证完成", validate_result)
        
        self.intermediate_results["validation"] = validate_result
        
        # ============ 构建最终结果 ============
        self._log("\n>>> 构建最终结果")
        
        final_result = self._build_final_result_with_images(
            query_image=query_image,
            candidate_images=candidate_images,
            dimension_result=dimension_result,
            all_descriptions=all_descriptions,
            all_scores=all_scores,
            validate_result=validate_result
        )
        
        self._log("=" * 60)
        self._log("排序流水线执行完成（批量模式）")
        self._log(f"最终排序: {final_result['final_ranking']}")
        self._log("=" * 60)
        
        return final_result
    
    def _run_with_images_sequential_mode(
        self,
        query_image: str,
        candidate_images: Dict[str, str],
        query_description: str = ""
    ) -> dict:
        """
        执行排序流水线（图片模式 - 顺序模式）
        
        在顺序模式下，逐个处理每个候选物品。
        """
        candidate_ids = list(candidate_images.keys())
        
        # ============ 阶段 1: 维度规划（基于查询图片）============
        self._log("\n>>> 阶段 1: 执行维度规划 (DimensionPlannerSkill) - 图片模式")
        
        dimension_input = {
            "query_image": query_image,
            "query_description": query_description,
            "dimension_bank": self.dimension_bank
        }
        
        # 构造缓存键
        cache_key_input = {
            "query_image_name": os.path.basename(query_image),
            "query_description": query_description,
            "dimension_bank_hash": hash(str(self.dimension_bank))
        }
        
        # 尝试从缓存获取
        dimension_result = self._get_cached_result("DimensionPlannerSkill_Image", cache_key_input)
        if dimension_result is None:
            dimension_result = self.dimension_planner.run_with_image(dimension_input)
            self._cache_result("DimensionPlannerSkill_Image", cache_key_input, dimension_result)
        
        self.intermediate_results["dimension_planning"] = dimension_result
        self._log("维度规划完成", dimension_result)
        
        dimensions = dimension_result.get("dimensions", [])
        inferred_scenario = dimension_result.get("inferred_scenario", "")
        query_object_analysis = dimension_result.get("query_object_analysis", "")
        
        self._log(f"图片分析: {query_object_analysis}")
        self._log(f"推断场景: {inferred_scenario}")
        self._log(f"规划维度: {[d['name'] for d in dimensions]}")
        
        # ============ 阶段 2: 候选描述（基于候选图片）============
        self._log("\n>>> 阶段 2: 执行候选描述 (DescriptorSkill) - 图片模式")
        
        all_descriptions = {}
        for candidate_id, candidate_image in candidate_images.items():
            self._log(f"  - 描述候选物品: {candidate_id} (图片: {os.path.basename(candidate_image)})")
            
            descriptor_input = {
                "candidate_id": candidate_id,
                "candidate_image": candidate_image,
                "dimensions": dimensions
            }
            
            # 构造缓存键
            cache_key = {
                "candidate_id": candidate_id,
                "candidate_image_name": os.path.basename(candidate_image),
                "dimensions_hash": hash(str(dimensions))
            }
            
            # 尝试从缓存获取
            descriptor_result = self._get_cached_result("DescriptorSkill_Image", cache_key)
            if descriptor_result is None:
                descriptor_result = self.descriptor.run_with_image(descriptor_input)
                self._cache_result("DescriptorSkill_Image", cache_key, descriptor_result)
            
            all_descriptions[candidate_id] = descriptor_result
        
        self.intermediate_results["descriptions"] = all_descriptions
        self._log("候选描述完成", {"candidates_described": len(all_descriptions)})
        
        # ============ 阶段 3: 相似度判断（纯文本模式，基于描述和评分标准）============
        self._log("\n>>> 阶段 3: 执行相似度判断 (JudgeSkill) - 纯文本模式（基于描述和评分标准）")
        
        # 构造缓存键
        cache_key = {
            "query_analysis_hash": hash(query_object_analysis),
            "candidate_ids": list(candidate_images.keys()),
            "dimensions_hash": hash(str(dimensions)),
            "descriptions_hash": hash(str(all_descriptions))
        }
        
        # 尝试从缓存获取批量评分
        batch_scores = self._get_cached_result("JudgeSkill_TextOnly", cache_key)
        if batch_scores is None:
            # 使用纯文本评分方法，只传入描述和评分标准，不传入图像
            batch_scores = self.judge.run_batch_with_text_only({
                "query_analysis": query_object_analysis,  # 阶段1产生的查询物品分析
                "candidate_descriptions": all_descriptions,  # 阶段2产生的候选描述
                "dimensions": dimensions  # 阶段1产生的维度及评分标准
            })
            self._cache_result("JudgeSkill_TextOnly", cache_key, batch_scores)
        
        # 转换为标准格式
        all_scores = {}
        for candidate_id, score_data in batch_scores.items():
            all_scores[candidate_id] = {
                "dimension_scores": score_data.get("scores", {})
            }
        
        self.intermediate_results["scores"] = all_scores
        self._log("相似度判断完成", {"candidates_judged": len(all_scores)})
        
        # ============ 阶段 4: 排序验证（可选）============
        if self.skip_validation:
            self._log("\n>>> 阶段 4: 跳过排序验证，直接计算加权排序")
            validate_result = self._calculate_ranking_from_scores(all_scores, dimensions)
            self._log(f"加权排序完成: {validate_result['final_ranking']}")
        else:
            self._log("\n>>> 阶段 4: 执行排序验证 (ValidateSkill)")
            
            validate_input = {
                "all_candidate_scores": all_scores,
                "dimensions": dimensions
            }
            
            # 尝试从缓存获取
            validate_result = self._get_cached_result("ValidateSkill", validate_input)
            if validate_result is None:
                validate_result = self.validator.run(validate_input)
                self._cache_result("ValidateSkill", validate_input, validate_result)
            
            self._log("排序验证完成", validate_result)
        
        self.intermediate_results["validation"] = validate_result
        
        # ============ 构建最终结果 ============
        self._log("\n>>> 构建最终结果")
        
        final_result = self._build_final_result_with_images(
            query_image=query_image,
            candidate_images=candidate_images,
            dimension_result=dimension_result,
            all_descriptions=all_descriptions,
            all_scores=all_scores,
            validate_result=validate_result
        )
        
        self._log("=" * 60)
        self._log("排序流水线执行完成（顺序模式）")
        self._log(f"最终排序: {final_result['final_ranking']}")
        self._log("=" * 60)
        
        return final_result
    
    def _build_final_result(
        self,
        query: str,
        candidate_ids: List[str],
        dimension_result: dict,
        all_descriptions: dict,
        all_scores: dict,
        validate_result: dict
    ) -> dict:
        # ... existing code ...
        """
        构建最终结果（文本模式）
        """
        dimensions = dimension_result.get("dimensions", [])
        final_ranking = validate_result.get("final_ranking", candidate_ids)
        weighted_scores = validate_result.get("weighted_scores", {})
        
        # 构建每个候选物品的详细报告
        candidate_reports = []
        for rank, candidate_id in enumerate(final_ranking, 1):
            report = {
                "rank": rank,
                "candidate_id": candidate_id,
                "weighted_total_score": weighted_scores.get(candidate_id, {}).get("total_score", 0),
                "dimension_scores": {},
                "descriptions": all_descriptions.get(candidate_id, {}).get("descriptions", {})
            }
            
            # 添加各维度的详细评分
            score_data = all_scores.get(candidate_id, {})
            dim_scores = score_data.get("dimension_scores", score_data.get("scores", {}))
            
            for dim in dimensions:
                dim_name = dim["name"]
                if dim_name in dim_scores:
                    score_info = dim_scores[dim_name]
                    # 支持简化格式（分数直接是数字）和标准格式（分数是对象）
                    if isinstance(score_info, (int, float)):
                        score = float(score_info)
                        report["dimension_scores"][dim_name] = {
                            "score": score,
                            "weight": dim.get("weight", 0),
                            "weighted_score": score * dim.get("weight", 0),
                            "reason": ""
                        }
                    elif isinstance(score_info, dict):
                        report["dimension_scores"][dim_name] = {
                            "score": score_info.get("score", 0),
                            "weight": dim.get("weight", 0),
                            "weighted_score": score_info.get("score", 0) * dim.get("weight", 0),
                            "reason": score_info.get("reason", "")
                        }
            
            candidate_reports.append(report)
        
        # 构建完整结果
        result = {
            "mode": "text",
            "query": query,
            "inferred_scenario": dimension_result.get("inferred_scenario", ""),
            "scenario_reasoning": dimension_result.get("scenario_reasoning", ""),
            "dimensions": [
                {
                    "name": dim["name"],
                    "description": dim["description"],
                    "weight": dim.get("weight", 0),
                    "scoring_criteria": dim.get("scoring_criteria", ""),
                    "source": dim.get("source", "bank")
                }
                for dim in dimensions
            ],
            "final_ranking": final_ranking,
            "candidate_reports": candidate_reports,
            "validation": {
                "skipped": self.skip_validation,
                "validation_checks": validate_result.get("validation_checks", []),
                "adjustments_made": validate_result.get("adjustments_made", []),
                "adjustments_count": validate_result.get("adjustments_count", 0),
                "validation_notes": validate_result.get("validation_notes", "跳过验证步骤" if self.skip_validation else ""),
                "confidence_score": validate_result.get("confidence_score", 1.0)
            },
            "execution_summary": {
                "total_candidates": len(candidate_ids),
                "dimensions_used": len(dimensions),
                "execution_log_entries": len(self.execution_log)
            }
        }
        
        return result
    
    def _build_final_result_with_images(
        self,
        query_image: str,
        candidate_images: Dict[str, str],
        dimension_result: dict,
        all_descriptions: dict,
        all_scores: dict,
        validate_result: dict
    ) -> dict:
        # ... existing code ...
        """
        构建最终结果（图片模式）
        """
        candidate_ids = list(candidate_images.keys())
        dimensions = dimension_result.get("dimensions", [])
        final_ranking = validate_result.get("final_ranking", candidate_ids)
        weighted_scores = validate_result.get("weighted_scores", {})
        
        # 构建每个候选物品的详细报告
        candidate_reports = []
        for rank, candidate_id in enumerate(final_ranking, 1):
            report = {
                "rank": rank,
                "candidate_id": candidate_id,
                "candidate_image": candidate_images.get(candidate_id, ""),
                "weighted_total_score": weighted_scores.get(candidate_id, {}).get("total_score", 0),
                "dimension_scores": {},
                "descriptions": all_descriptions.get(candidate_id, {}).get("descriptions", {})
            }
            
            # 添加各维度的详细评分
            score_data = all_scores.get(candidate_id, {})
            dim_scores = score_data.get("dimension_scores", score_data.get("scores", {}))
            
            for dim in dimensions:
                dim_name = dim["name"]
                if dim_name in dim_scores:
                    score_info = dim_scores[dim_name]
                    # 支持简化格式（分数直接是数字）和标准格式（分数是对象）
                    if isinstance(score_info, (int, float)):
                        score = float(score_info)
                        report["dimension_scores"][dim_name] = {
                            "score": score,
                            "weight": dim.get("weight", 0),
                            "weighted_score": score * dim.get("weight", 0),
                            "reason": ""
                        }
                    elif isinstance(score_info, dict):
                        report["dimension_scores"][dim_name] = {
                            "score": score_info.get("score", 0),
                            "weight": dim.get("weight", 0),
                            "weighted_score": score_info.get("score", 0) * dim.get("weight", 0),
                            "reason": score_info.get("reason", "")
                        }
            
            candidate_reports.append(report)
        
        # 获取缓存统计
        cache_stats = None
        if self._result_cache:
            cache_stats = self._result_cache.get_stats()
        
        # 构建完整结果
        result = {
            "mode": "image",
            "batch_mode": self.use_batch_mode,
            "query_image": query_image,
            "query_object_analysis": dimension_result.get("query_object_analysis", ""),
            "inferred_scenario": dimension_result.get("inferred_scenario", ""),
            "scenario_reasoning": dimension_result.get("scenario_reasoning", ""),
            "dimensions": [
                {
                    "name": dim["name"],
                    "description": dim["description"],
                    "weight": dim.get("weight", 0),
                    "scoring_criteria": dim.get("scoring_criteria", ""),
                    "source": dim.get("source", "bank")
                }
                for dim in dimensions
            ],
            "final_ranking": final_ranking,
            "candidate_reports": candidate_reports,
            "validation": {
                "skipped": self.skip_validation,
                "validation_checks": validate_result.get("validation_checks", []),
                "adjustments_made": validate_result.get("adjustments_made", []),
                "adjustments_count": validate_result.get("adjustments_count", 0),
                "validation_notes": validate_result.get("validation_notes", "跳过验证步骤" if self.skip_validation else ""),
                "confidence_score": validate_result.get("confidence_score", 1.0)
            },
            "execution_summary": {
                "total_candidates": len(candidate_ids),
                "dimensions_used": len(dimensions),
                "execution_log_entries": len(self.execution_log),
                "cache_stats": cache_stats
            }
        }
        
        return result
    
    def get_execution_log(self) -> List[dict]:
        """
        获取执行日志
        
        Returns:
            List[dict]: 执行日志列表
        """
        return self.execution_log
    
    def get_intermediate_results(self) -> dict:
        """
        获取中间结果
        
        Returns:
            dict: 各阶段的中间结果
        """
        return self.intermediate_results
    
    def export_result(self, result: dict, filepath: str) -> None:
        """
        导出结果到文件
        
        Args:
            result: 排序结果
            filepath: 输出文件路径
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        if self.verbose:
            print(f"结果已导出到: {filepath}")
    
    def export_intermediate_results(self, filepath: str) -> None:
        """
        导出中间结果到文件
        
        Args:
            filepath: 输出文件路径
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.intermediate_results, f, ensure_ascii=False, indent=2)
        
        if self.verbose:
            print(f"中间结果已导出到: {filepath}")
    
    def generate_explanation_report(self, result: dict) -> str:
        # ... existing code ...
        """
        生成可读的解释报告
        
        Args:
            result: 排序结果
            
        Returns:
            str: 格式化的解释报告
        """
        lines = []
        lines.append("=" * 70)
        lines.append("排序结果解释报告")
        lines.append("=" * 70)
        
        # 根据模式显示不同的查询信息
        if result.get("mode") == "image":
            lines.append(f"\n## 查询图片\n{result.get('query_image', '无')}")
            lines.append(f"\n### 图片分析\n{result.get('query_object_analysis', '无')}")
            if result.get("batch_mode"):
                lines.append("\n### 执行模式\n批量模式（一次性发送所有图像）")
        else:
            lines.append(f"\n## 查询内容\n{result.get('query', '无')}")
        
        lines.append(f"\n## 推断场景\n{result.get('inferred_scenario', '无')}")
        lines.append(f"\n### 场景推断理由\n{result.get('scenario_reasoning', '无')}")
        
        lines.append("\n## 使用的评估维度")
        for dim in result.get('dimensions', []):
            lines.append(f"\n### {dim['name']} (权重: {dim['weight']:.2f}, 来源: {dim['source']})")
            lines.append(f"描述: {dim['description']}")
            lines.append(f"评分标准: {dim['scoring_criteria']}")
        
        lines.append("\n## 最终排序结果")
        lines.append(f"排序（从最相似到最不相似）: {' > '.join(result.get('final_ranking', []))}")
        
        lines.append("\n## 各候选物品详细报告")
        for report in result.get('candidate_reports', []):
            lines.append(f"\n### 排名 {report['rank']}: {report['candidate_id']}")
            if result.get("mode") == "image":
                lines.append(f"图片: {report.get('candidate_image', '无')}")
            lines.append(f"加权总分: {report['weighted_total_score']:.4f}")
            lines.append("\n各维度评分:")
            for dim_name, score_data in report.get('dimension_scores', {}).items():
                lines.append(f"  - {dim_name}: {score_data['score']:.2f} × {score_data['weight']:.2f} = {score_data['weighted_score']:.4f}")
                if score_data.get('reason'):
                    lines.append(f"    理由: {score_data['reason']}")
        
        lines.append("\n## 验证信息")
        validation = result.get('validation', {})
        lines.append(f"验证说明: {validation.get('validation_notes', '无')}")
        lines.append(f"置信度: {validation.get('confidence_score', 0):.2f}")
        
        if validation.get('adjustments_made'):
            lines.append("\n调整记录:")
            for adj in validation['adjustments_made']:
                lines.append(f"  - {adj}")
        
        # 缓存统计
        exec_summary = result.get('execution_summary', {})
        if exec_summary.get('cache_stats'):
            cache_stats = exec_summary['cache_stats']
            lines.append("\n## 缓存统计")
            lines.append(f"  - 缓存大小: {cache_stats.get('cache_size', 0)}")
            lines.append(f"  - 命中次数: {cache_stats.get('hit_count', 0)}")
            lines.append(f"  - 未命中次数: {cache_stats.get('miss_count', 0)}")
            lines.append(f"  - 命中率: {cache_stats.get('hit_rate', 0):.2%}")
        
        lines.append("\n" + "=" * 70)
        
        return "\n".join(lines)


class PipelineConfig:
    """
    流水线配置类
    
    用于配置流水线的各项参数。
    """
    
    def __init__(
        self,
        use_custom_dimension_bank: bool = False,
        custom_dimension_bank: Optional[dict] = None,
        verbose: bool = True,
        export_intermediate_results: bool = False,
        output_format: str = "json",
        use_batch_mode: bool = True,
        use_cache: bool = True,
        skip_validation: bool = True
    ):
        """
        初始化配置
        
        Args:
            use_custom_dimension_bank: 是否使用自定义维度银行
            custom_dimension_bank: 自定义维度银行
            verbose: 是否输出详细日志
            export_intermediate_results: 是否导出中间结果
            output_format: 输出格式 ("json" 或 "text")
            use_batch_mode: 是否使用批量模式（一次性发送所有图像）
            use_cache: 是否使用中间结果缓存
            skip_validation: 是否跳过验证步骤（默认 True，直接使用加权分数排序）
        """
        self.use_custom_dimension_bank = use_custom_dimension_bank
        self.custom_dimension_bank = custom_dimension_bank
        self.verbose = verbose
        self.export_intermediate_results = export_intermediate_results
        self.output_format = output_format
        self.use_batch_mode = use_batch_mode
        self.use_cache = use_cache
        self.skip_validation = skip_validation
    
    def get_dimension_bank(self) -> dict:
        """获取维度银行"""
        if self.use_custom_dimension_bank and self.custom_dimension_bank:
            return self.custom_dimension_bank
        return get_dimension_bank()


def create_pipeline(config: Optional[PipelineConfig] = None) -> RankingPipeline:
    """
    创建排序流水线实例
    
    Args:
        config: 流水线配置（可选）
        
    Returns:
        RankingPipeline: 流水线实例
    """
    if config is None:
        config = PipelineConfig()
    
    return RankingPipeline(
        dimension_bank=config.get_dimension_bank(),
        verbose=config.verbose,
        use_batch_mode=config.use_batch_mode,
        use_cache=config.use_cache,
        skip_validation=config.skip_validation
    )
