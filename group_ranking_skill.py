"""
分组排序技能模块

用于处理预分组的 JSON 文件，每个 group 包含若干物体，
随机选择一个作为 query，其余作为候选，调用 ranking pipeline 进行排序。

JSON 分组文件格式：
{
    "metadata": {
        "group_size": 10,
        "total_groups": N,
        ...
    },
    "groups": [
        {
            "group_id": "Category_0",
            "category": "Category",
            "group_index": 0,
            "objects": [
                {
                    "image_path": "...",
                    "description": "...",
                    "object_id": "...",
                    "category": "..."
                },
                ...
            ]
        },
        ...
    ]
}
"""

import os
import sys
import json
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime

# 确保模块路径正确
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_interface import get_llm_interface, get_persistent_cache
from pipeline import RankingPipeline


# ==================== 数据结构 ====================

@dataclass
class GroupObject:
    """分组中的单个物体"""
    object_id: str
    image_path: str
    description: str
    category: str
    mesh_path: str = ""
    llm_category: str = ""
    
    def is_valid(self) -> bool:
        """检查图片路径是否有效"""
        return os.path.exists(self.image_path)


@dataclass
class ObjectGroup:
    """一个物体分组"""
    group_id: str
    category: str
    group_index: int
    objects: List[GroupObject] = field(default_factory=list)
    
    @property
    def size(self) -> int:
        return len(self.objects)
    
    def get_valid_objects(self) -> List[GroupObject]:
        """获取有效的物体列表（图片路径存在）"""
        return [obj for obj in self.objects if obj.is_valid()]
    
    @property
    def valid_count(self) -> int:
        return len(self.get_valid_objects())


@dataclass
class GroupedData:
    """分组数据"""
    source_file: str
    group_size: int
    total_groups: int
    total_categories: int
    groups: List[ObjectGroup] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    category_statistics: Dict = field(default_factory=dict)


@dataclass
class GroupRankingResult:
    """单个分组的排序结果"""
    group_id: str
    category: str
    query_object: GroupObject
    candidate_objects: List[GroupObject]
    predicted_ranking: List[str]
    ranking_scores: Dict[str, Dict]  # 每个候选的详细评分
    pipeline_result: Dict  # 完整的 pipeline 结果
    execution_time: float = 0.0


@dataclass
class BatchRankingResult:
    """批量排序结果"""
    source_file: str
    total_groups: int
    processed_groups: int
    successful_groups: int
    failed_groups: int
    group_results: List[GroupRankingResult] = field(default_factory=list)
    failed_group_ids: List[str] = field(default_factory=list)
    total_execution_time: float = 0.0
    application_scenario: str = ""


# ==================== 分组数据加载器 ====================

class GroupDataLoader:
    """分组数据加载器"""
    
    def __init__(self, image_base_path: str = None):
        """
        初始化分组数据加载器
        
        Args:
            image_base_path: 图片基础路径（用于路径转换）
                如果为 None，则直接使用 JSON 中的路径
        """
        self.image_base_path = image_base_path
    
    def load_grouped_json(self, json_path: str) -> GroupedData:
        """
        加载分组 JSON 文件
        
        Args:
            json_path: JSON 文件路径
            
        Returns:
            GroupedData 对象
        """
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"分组 JSON 文件不存在: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 解析 metadata
        metadata = data.get("metadata", {})
        group_size = metadata.get("group_size", 10)
        total_groups = metadata.get("total_groups", 0)
        total_categories = metadata.get("total_categories", 0)
        category_statistics = data.get("category_statistics", {})
        
        # 解析 groups
        groups = []
        for group_data in data.get("groups", []):
            group_id = group_data.get("group_id", "")
            category = group_data.get("category", "")
            group_index = group_data.get("group_index", 0)
            
            # 解析物体列表
            objects = []
            for obj_data in group_data.get("objects", []):
                obj = GroupObject(
                    object_id=obj_data.get("object_id", ""),
                    image_path=self._resolve_image_path(obj_data.get("image_path", "")),
                    description=obj_data.get("description", ""),
                    category=obj_data.get("category", ""),
                    mesh_path=obj_data.get("mesh_path", ""),
                    llm_category=obj_data.get("llm_category", "")
                )
                objects.append(obj)
            
            group = ObjectGroup(
                group_id=group_id,
                category=category,
                group_index=group_index,
                objects=objects
            )
            groups.append(group)
        
        return GroupedData(
            source_file=json_path,
            group_size=group_size,
            total_groups=total_groups,
            total_categories=total_categories,
            groups=groups,
            metadata=metadata,
            category_statistics=category_statistics
        )
    
    def _resolve_image_path(self, original_path: str) -> str:
        """
        解析图片路径
        
        如果设置了 image_base_path，则进行路径转换
        
        Args:
            original_path: 原始路径（可能是服务器路径）
            
        Returns:
            解析后的本地路径
        """
        if not original_path:
            return ""
        
        if self.image_base_path:
            # 提取相对路径部分并拼接到本地基础路径
            # 原始路径格式: /apdcephfs/.../hf-objaverse-v1/xxx/yyy/basecolor/front.png
            # 尝试提取 hf-objaverse-v1 之后的部分
            path_parts = original_path.replace("\\", "/").split("/")
            
            # 查找 hf-objaverse-v1 或类似的关键目录
            key_dirs = ["hf-objaverse-v1", "objaverse", "basecolor"]
            start_index = -1
            for key_dir in key_dirs:
                if key_dir in path_parts:
                    start_index = path_parts.index(key_dir)
                    break
            
            if start_index >= 0:
                relative_path = "/".join(path_parts[start_index:])
                return os.path.join(self.image_base_path, relative_path)
        
        # 直接返回原始路径
        return original_path
    
    def prepare_group_for_ranking(
        self, 
        group: ObjectGroup, 
        query_index: int = None,
        random_seed: int = None
    ) -> Tuple[GroupObject, List[GroupObject], Dict[str, str]]:
        """
        准备单个分组的排序数据
        
        Args:
            group: 物体分组
            query_index: 指定的 query 索引（None 则随机选择）
            random_seed: 随机种子（用于复现）
            
        Returns:
            (query_object, candidate_objects, candidate_images_dict)
        """
        valid_objects = group.get_valid_objects()
        
        if len(valid_objects) < 2:
            raise ValueError(f"分组 {group.group_id} 有效物体数量不足（至少需要2个）")
        
        # 选择 query
        if query_index is not None:
            if query_index >= len(valid_objects):
                raise ValueError(f"query_index {query_index} 超出范围")
            query_idx = query_index
        else:
            if random_seed is not None:
                random.seed(random_seed)
            query_idx = random.randint(0, len(valid_objects) - 1)
        
        query_object = valid_objects[query_idx]
        
        # 其余作为 candidates
        candidate_objects = [obj for i, obj in enumerate(valid_objects) if i != query_idx]
        
        # 构建候选图片字典
        candidate_images = {obj.object_id: obj.image_path for obj in candidate_objects}
        
        return query_object, candidate_objects, candidate_images


# ==================== 分组排序执行器 ====================

class GroupRankingExecutor:
    """分组排序执行器"""
    
    def __init__(
        self,
        llm_mode: str = "api",
        model_name: str = None,
        application_scenario: str = "游戏场景中的相似资产检索",
        verbose: bool = True,
        use_cache: bool = True
    ):
        """
        初始化分组排序执行器
        
        Args:
            llm_mode: LLM 模式（"api", "qwen", "mock"）
            model_name: 模型名称
            application_scenario: 应用场景描述
            verbose: 是否输出详细信息
            use_cache: 是否使用缓存
        """
        self.llm_mode = llm_mode
        self.model_name = model_name
        self.application_scenario = application_scenario
        self.verbose = verbose
        self.use_cache = use_cache
        
        # 初始化 LLM
        self._init_llm()
    
    def _init_llm(self):
        """初始化 LLM 接口"""
        from llm_interface import get_llm_interface
        
        # 根据模式选择默认模型
        if self.model_name is None:
            if self.llm_mode == "qwen":
                self.model_name = "qwen3-vl-235b-a22b-thinking"
            else:
                self.model_name = "gemini-3-flash-preview"
        
        self.llm = get_llm_interface(
            mode=self.llm_mode,
            model=self.model_name,
            force_new=True
        )
        
        if self.verbose:
            print(f"✅ LLM 初始化完成: 模式={self.llm_mode}, 模型={self.model_name}")
    
    def run_single_group(
        self,
        group: ObjectGroup,
        query_index: int = None,
        random_seed: int = None,
        source_json_path: str = None
    ) -> GroupRankingResult:
        """
        运行单个分组的排序
        
        Args:
            group: 物体分组
            query_index: 指定的 query 索引（None 则随机选择）
            random_seed: 随机种子
            source_json_path: 源 JSON 路径（用于缓存）
            
        Returns:
            GroupRankingResult 对象
        """
        import time
        start_time = time.time()
        
        # 准备数据
        loader = GroupDataLoader()
        query_object, candidate_objects, candidate_images = loader.prepare_group_for_ranking(
            group, query_index, random_seed
        )
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"🎯 处理分组: {group.group_id}")
            print(f"{'='*60}")
            print(f"  Query: {query_object.object_id}")
            print(f"  Query 描述: {query_object.description[:50]}...")
            print(f"  Candidates 数量: {len(candidate_objects)}")
        
        # 构建 query 描述
        full_description = f"【应用场景】{self.application_scenario}"
        full_description += f"\n【Query 描述】{query_object.description}"
        full_description += f"\n【类别】{query_object.category}"
        
        # 创建并运行 pipeline
        cache_key = f"{source_json_path}_{group.group_id}" if source_json_path else None
        pipeline = RankingPipeline(verbose=self.verbose, source_json_path=cache_key)
        
        try:
            result = pipeline.run_with_images(
                query_image=query_object.image_path,
                candidate_images=candidate_images,
                query_description=full_description
            )
            
            execution_time = time.time() - start_time
            
            return GroupRankingResult(
                group_id=group.group_id,
                category=group.category,
                query_object=query_object,
                candidate_objects=candidate_objects,
                predicted_ranking=result.get("final_ranking", []),
                ranking_scores=result.get("scoring_result", {}).get("scores", {}),
                pipeline_result=result,
                execution_time=execution_time
            )
            
        except Exception as e:
            if self.verbose:
                print(f"❌ 分组 {group.group_id} 排序失败: {e}")
            raise
    
    def run_batch(
        self,
        grouped_data: GroupedData,
        max_groups: int = None,
        group_indices: List[int] = None,
        random_seed: int = 42,
        save_intermediate: bool = True,
        output_dir: str = None
    ) -> BatchRankingResult:
        """
        批量运行分组排序
        
        Args:
            grouped_data: 分组数据
            max_groups: 最大处理分组数
            group_indices: 指定处理的分组索引列表
            random_seed: 随机种子（用于复现）
            save_intermediate: 是否保存中间结果
            output_dir: 输出目录
            
        Returns:
            BatchRankingResult 对象
        """
        import time
        start_time = time.time()
        
        # 确定要处理的分组
        groups_to_process = grouped_data.groups
        
        if group_indices:
            groups_to_process = [grouped_data.groups[i] for i in group_indices 
                                if i < len(grouped_data.groups)]
        
        if max_groups:
            groups_to_process = groups_to_process[:max_groups]
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"🚀 批量分组排序")
            print(f"{'='*70}")
            print(f"  源文件: {grouped_data.source_file}")
            print(f"  总分组数: {grouped_data.total_groups}")
            print(f"  待处理分组数: {len(groups_to_process)}")
            print(f"  应用场景: {self.application_scenario}")
        
        # 初始化结果
        batch_result = BatchRankingResult(
            source_file=grouped_data.source_file,
            total_groups=grouped_data.total_groups,
            processed_groups=0,
            successful_groups=0,
            failed_groups=0,
            application_scenario=self.application_scenario
        )
        
        # 设置输出目录
        if output_dir is None:
            output_dir = os.path.dirname(grouped_data.source_file)
        
        # 逐个处理分组
        for i, group in enumerate(groups_to_process):
            if self.verbose:
                print(f"\n📦 处理分组 {i+1}/{len(groups_to_process)}: {group.group_id}")
            
            # 为每个分组使用不同的随机种子（但可复现）
            group_seed = random_seed + group.group_index if random_seed else None
            
            try:
                result = self.run_single_group(
                    group=group,
                    random_seed=group_seed,
                    source_json_path=grouped_data.source_file
                )
                
                batch_result.group_results.append(result)
                batch_result.successful_groups += 1
                
                if self.verbose:
                    print(f"  ✅ 排序完成: {result.predicted_ranking}")
                
            except Exception as e:
                batch_result.failed_groups += 1
                batch_result.failed_group_ids.append(group.group_id)
                if self.verbose:
                    print(f"  ❌ 排序失败: {e}")
            
            batch_result.processed_groups += 1
            
            # 保存中间结果
            if save_intermediate and (i + 1) % 10 == 0:
                self._save_intermediate_result(batch_result, output_dir)
        
        batch_result.total_execution_time = time.time() - start_time
        
        # 打印汇总
        if self.verbose:
            self._print_batch_summary(batch_result)
        
        return batch_result
    
    def _save_intermediate_result(self, result: BatchRankingResult, output_dir: str):
        """保存中间结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"group_ranking_intermediate_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        # 转换为可序列化的字典
        result_dict = self._result_to_dict(result)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=2)
        
        if self.verbose:
            print(f"  💾 中间结果已保存: {filepath}")
    
    def _result_to_dict(self, result: BatchRankingResult) -> Dict:
        """将结果转换为字典"""
        return {
            "source_file": result.source_file,
            "total_groups": result.total_groups,
            "processed_groups": result.processed_groups,
            "successful_groups": result.successful_groups,
            "failed_groups": result.failed_groups,
            "failed_group_ids": result.failed_group_ids,
            "total_execution_time": result.total_execution_time,
            "application_scenario": result.application_scenario,
            "group_results": [
                {
                    "group_id": gr.group_id,
                    "category": gr.category,
                    "query_object_id": gr.query_object.object_id,
                    "query_description": gr.query_object.description,
                    "candidate_object_ids": [obj.object_id for obj in gr.candidate_objects],
                    "predicted_ranking": gr.predicted_ranking,
                    "execution_time": gr.execution_time
                }
                for gr in result.group_results
            ]
        }
    
    def _print_batch_summary(self, result: BatchRankingResult):
        """打印批量结果汇总"""
        print(f"\n{'='*70}")
        print(f"📊 批量排序汇总")
        print(f"{'='*70}")
        print(f"  处理分组数: {result.processed_groups}")
        print(f"  成功: {result.successful_groups}")
        print(f"  失败: {result.failed_groups}")
        print(f"  总耗时: {result.total_execution_time:.2f} 秒")
        if result.processed_groups > 0:
            avg_time = result.total_execution_time / result.processed_groups
            print(f"  平均每组耗时: {avg_time:.2f} 秒")
        
        if result.failed_group_ids:
            print(f"\n❌ 失败的分组:")
            for gid in result.failed_group_ids[:10]:
                print(f"    - {gid}")
            if len(result.failed_group_ids) > 10:
                print(f"    ... 还有 {len(result.failed_group_ids) - 10} 个")
    
    def save_result(self, result: BatchRankingResult, output_path: str = None):
        """
        保存最终结果
        
        Args:
            result: 批量排序结果
            output_path: 输出路径（None 则自动生成）
        """
        if output_path is None:
            output_dir = os.path.dirname(result.source_file)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = os.path.splitext(os.path.basename(result.source_file))[0]
            output_path = os.path.join(output_dir, f"{base_name}_ranking_result_{timestamp}.json")
        
        result_dict = self._result_to_dict(result)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=2)
        
        if self.verbose:
            print(f"\n💾 最终结果已保存: {output_path}")
        
        return output_path


# ==================== 便捷函数 ====================

def run_group_ranking(
    json_file: str,
    max_groups: int = None,
    group_indices: List[int] = None,
    llm_mode: str = "api",
    model_name: str = None,
    application_scenario: str = "游戏场景中的相似资产检索",
    random_seed: int = 42,
    image_base_path: str = None,
    verbose: bool = True,
    save_result: bool = True
) -> BatchRankingResult:
    """
    运行分组排序的便捷函数
    
    Args:
        json_file: 分组 JSON 文件路径
        max_groups: 最大处理分组数
        group_indices: 指定处理的分组索引列表
        llm_mode: LLM 模式
        model_name: 模型名称
        application_scenario: 应用场景
        random_seed: 随机种子
        image_base_path: 图片基础路径
        verbose: 是否输出详细信息
        save_result: 是否保存结果
        
    Returns:
        BatchRankingResult 对象
    """
    # 加载数据
    loader = GroupDataLoader(image_base_path=image_base_path)
    grouped_data = loader.load_grouped_json(json_file)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"📂 加载分组数据")
        print(f"{'='*70}")
        print(f"  文件: {json_file}")
        print(f"  分组大小: {grouped_data.group_size}")
        print(f"  总分组数: {grouped_data.total_groups}")
        print(f"  类别数: {grouped_data.total_categories}")
    
    # 创建执行器并运行
    executor = GroupRankingExecutor(
        llm_mode=llm_mode,
        model_name=model_name,
        application_scenario=application_scenario,
        verbose=verbose
    )
    
    result = executor.run_batch(
        grouped_data=grouped_data,
        max_groups=max_groups,
        group_indices=group_indices,
        random_seed=random_seed
    )
    
    # 保存结果
    if save_result:
        executor.save_result(result)
    
    # 显示 Token 统计
    if verbose:
        show_token_statistics()
    
    return result


def show_token_statistics():
    """显示 Token 统计信息"""
    from llm_interface import get_llm_interface
    
    llm = get_llm_interface()
    stats = llm.get_call_statistics()
    
    print(f"\n{'='*60}")
    print(f"📊 LLM 调用统计")
    print(f"{'='*60}")
    print(f"  总调用次数: {stats['total_calls']}")
    print(f"  运行模式: {stats['mode']}")
    
    print(f"\n📝 Token 使用统计:")
    print(f"  总输入 Token: {stats.get('total_input_tokens', 0):,}")
    print(f"  总输出 Token: {stats.get('total_output_tokens', 0):,}")
    print(f"  总 Token: {stats.get('total_tokens', 0):,}")
    
    token_history = stats.get('token_history', [])
    if token_history:
        avg_input = stats.get('total_input_tokens', 0) / len(token_history)
        avg_output = stats.get('total_output_tokens', 0) / len(token_history)
        print(f"  平均输入 Token/次: {avg_input:,.0f}")
        print(f"  平均输出 Token/次: {avg_output:,.0f}")


# ==================== 交互式界面 ====================

def interactive_main():
    """交互式主函数"""
    print("\n" + "=" * 70)
    print("🎯 分组排序技能 - Group Ranking Skill")
    print("=" * 70)
    
    print("""
请选择操作:

1. 运行单个分组排序（测试）
2. 运行批量分组排序
3. 查看分组 JSON 文件信息
4. 返回

请输入选项 (1-4): """, end="")
    
    try:
        choice = input().strip()
    except:
        return
    
    if choice == "1":
        # 单个分组测试
        print("\n请输入分组 JSON 文件路径: ", end="")
        json_file = input().strip()
        print("请输入要测试的分组索引 (默认 0): ", end="")
        group_idx_input = input().strip()
        group_idx = int(group_idx_input) if group_idx_input else 0
        
        # 选择 LLM 模式
        print("\n请选择 LLM 模式:")
        print("  1. Gemini (默认)")
        print("  2. QWEN")
        print("  3. Mock (模拟)")
        print("请输入选项 (1-3, 默认 1): ", end="")
        llm_choice = input().strip()
        
        llm_mode = "api"
        if llm_choice == "2":
            llm_mode = "qwen"
        elif llm_choice == "3":
            llm_mode = "mock"
        
        run_group_ranking(
            json_file=json_file,
            group_indices=[group_idx],
            llm_mode=llm_mode,
            verbose=True
        )
    
    elif choice == "2":
        # 批量排序
        print("\n请输入分组 JSON 文件路径: ", end="")
        json_file = input().strip()
        print("请输入最大处理分组数 (直接回车处理全部): ", end="")
        max_groups_input = input().strip()
        max_groups = int(max_groups_input) if max_groups_input else None
        
        # 选择 LLM 模式
        print("\n请选择 LLM 模式:")
        print("  1. Gemini (默认)")
        print("  2. QWEN")
        print("  3. Mock (模拟)")
        print("请输入选项 (1-3, 默认 1): ", end="")
        llm_choice = input().strip()
        
        llm_mode = "api"
        if llm_choice == "2":
            llm_mode = "qwen"
        elif llm_choice == "3":
            llm_mode = "mock"
        
        # 选择应用场景
        print("\n请输入应用场景 (直接回车使用默认: 游戏场景中的相似资产检索): ", end="")
        scenario = input().strip()
        if not scenario:
            scenario = "游戏场景中的相似资产检索"
        
        run_group_ranking(
            json_file=json_file,
            max_groups=max_groups,
            llm_mode=llm_mode,
            application_scenario=scenario,
            verbose=True
        )
    
    elif choice == "3":
        # 查看文件信息
        print("\n请输入分组 JSON 文件路径: ", end="")
        json_file = input().strip()
        
        loader = GroupDataLoader()
        grouped_data = loader.load_grouped_json(json_file)
        
        print(f"\n{'='*60}")
        print(f"📂 分组文件信息")
        print(f"{'='*60}")
        print(f"  分组大小: {grouped_data.group_size}")
        print(f"  总分组数: {grouped_data.total_groups}")
        print(f"  类别数: {grouped_data.total_categories}")
        
        print(f"\n📊 类别统计:")
        for cat, stats in grouped_data.category_statistics.items():
            print(f"  {cat}:")
            print(f"    物体总数: {stats.get('total_objects', 0)}")
            print(f"    分组数: {stats.get('num_groups', 0)}")
        
        print(f"\n📦 前5个分组:")
        for group in grouped_data.groups[:5]:
            valid_count = group.valid_count
            print(f"  {group.group_id}: {group.size} 个物体, {valid_count} 个有效")
        
        if len(grouped_data.groups) > 5:
            print(f"  ... 还有 {len(grouped_data.groups) - 5} 个分组")


def demo():
    """演示函数"""
    print("\n" + "=" * 70)
    print("🎯 分组排序技能演示 (Mock 模式)")
    print("=" * 70)
    
    # 使用 mock 模式演示
    demo_json = r"d:\3d-object-数据集\1000条黄金数据\agent_skills_for_bcmk\raw_objects_group\objaverse_golden_character.json"
    
    if not os.path.exists(demo_json):
        print(f"❌ 演示文件不存在: {demo_json}")
        return
    
    # 加载并显示数据信息
    loader = GroupDataLoader()
    grouped_data = loader.load_grouped_json(demo_json)
    
    print(f"\n📂 加载分组数据:")
    print(f"  分组大小: {grouped_data.group_size}")
    print(f"  总分组数: {grouped_data.total_groups}")
    
    # 使用 mock 模式运行一个分组
    print(f"\n🚀 使用 Mock 模式测试第一个分组...")
    
    run_group_ranking(
        json_file=demo_json,
        group_indices=[0],  # 只处理第一个分组
        llm_mode="mock",
        verbose=True
    )


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo()
    else:
        interactive_main()
