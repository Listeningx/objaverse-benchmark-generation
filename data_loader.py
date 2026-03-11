"""
数据加载模块

用于从 JSON 文件加载评测数据，并根据 ID 找到对应的正视图路径。
支持 ESB、GSO、MN40、NTU 四种数据集。
"""

import os
import json
import re
import glob
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


# ==================== 配置 ====================

# 评测数据集根目录
EVAL_ROOT = r"D:\3d-object-数据集\评测"

# 数据集路径映射
DATASET_PATHS = {
    "ESB": os.path.join(EVAL_ROOT, "OS-ESB-core"),
    "MN40": os.path.join(EVAL_ROOT, "OS-MN40-core"),
    "NTU": os.path.join(EVAL_ROOT, "OS-NTU-core"),
    "GSO": os.path.join(EVAL_ROOT, "GSO_resample"),
}

# 正视图文件名（h_0 表示水平角度为0度的正视图）
FRONT_VIEW_FILENAME = "h_0.jpg"

# GSO 数据集的正视图文件名
GSO_FRONT_VIEW_FILENAME = "0.jpg"


@dataclass
class DataItem:
    """数据项"""
    id: str
    score: float
    image_path: Optional[str] = None
    is_valid: bool = False
    error_message: str = ""


@dataclass
class LoadedData:
    """加载的数据"""
    dataset_type: str  # ESB, GSO, MN40, NTU
    source_file: str
    total_count: int
    query_item: Optional[DataItem]  # 第一项作为query
    candidate_items: List[DataItem]  # 其余项作为candidates
    valid_count: int
    invalid_count: int
    

class DatasetPathResolver:
    """数据集路径解析器"""
    
    def __init__(self, eval_root: str = EVAL_ROOT):
        """
        初始化路径解析器
        
        Args:
            eval_root: 评测数据集根目录
        """
        self.eval_root = eval_root
        self.dataset_paths = {
            "ESB": os.path.join(eval_root, "OS-ESB-core"),
            "MN40": os.path.join(eval_root, "OS-MN40-core"),
            "NTU": os.path.join(eval_root, "OS-NTU-core"),
            "GSO": os.path.join(eval_root, "GSO_resample"),
        }
        
        # 缓存已找到的路径
        self._path_cache: Dict[str, str] = {}
        
        # 为每个数据集预建索引
        self._build_indices()
    
    def _build_indices(self):
        """预建数据集索引，加速路径查找"""
        self._esb_index = {}
        self._mn40_index = {}
        self._ntu_index = {}
        self._gso_index = {}
        
        # 构建 ESB 索引
        esb_path = self.dataset_paths.get("ESB", "")
        if os.path.exists(esb_path):
            self._build_esb_index(esb_path)
        
        # 构建 MN40 索引
        mn40_path = self.dataset_paths.get("MN40", "")
        if os.path.exists(mn40_path):
            self._build_mn40_index(mn40_path)
        
        # 构建 NTU 索引
        ntu_path = self.dataset_paths.get("NTU", "")
        if os.path.exists(ntu_path):
            self._build_ntu_index(ntu_path)
        
        # 构建 GSO 索引
        gso_path = self.dataset_paths.get("GSO", "")
        if os.path.exists(gso_path):
            self._build_gso_index(gso_path)
    
    def _build_esb_index(self, base_path: str):
        """构建 ESB 数据集索引"""
        # ESB 结构: OS-ESB-core/{type}/{id}/image/h_0.jpg
        # 或: OS-ESB-core/{type}/{category}/{id}/image/h_0.jpg
        for type_dir in ["target", "query", "train"]:
            type_path = os.path.join(base_path, type_dir)
            if not os.path.exists(type_path):
                continue
            
            for item in os.listdir(type_path):
                item_path = os.path.join(type_path, item)
                if os.path.isdir(item_path):
                    # 检查是否是直接的 ID 目录
                    image_dir = os.path.join(item_path, "image")
                    if os.path.exists(image_dir):
                        # 直接是 ID 目录
                        front_view = os.path.join(image_dir, FRONT_VIEW_FILENAME)
                        if os.path.exists(front_view):
                            self._esb_index[item.lower()] = front_view
                    else:
                        # 可能是分类目录
                        for sub_item in os.listdir(item_path):
                            sub_item_path = os.path.join(item_path, sub_item)
                            if os.path.isdir(sub_item_path):
                                sub_image_dir = os.path.join(sub_item_path, "image")
                                if os.path.exists(sub_image_dir):
                                    front_view = os.path.join(sub_image_dir, FRONT_VIEW_FILENAME)
                                    if os.path.exists(front_view):
                                        self._esb_index[sub_item.lower()] = front_view
    
    def _build_mn40_index(self, base_path: str):
        """构建 MN40 数据集索引"""
        # MN40 结构: OS-MN40-core/{type}/{category}/{id}/image/h_0.jpg
        # 或: OS-MN40-core/{type}/{id}/image/h_0.jpg
        for type_dir in ["target", "query", "train"]:
            type_path = os.path.join(base_path, type_dir)
            if not os.path.exists(type_path):
                continue
            
            for item in os.listdir(type_path):
                item_path = os.path.join(type_path, item)
                if os.path.isdir(item_path):
                    # 检查是否是直接的 ID 目录
                    image_dir = os.path.join(item_path, "image")
                    if os.path.exists(image_dir):
                        # 直接是 ID 目录
                        front_view = os.path.join(image_dir, FRONT_VIEW_FILENAME)
                        if os.path.exists(front_view):
                            self._mn40_index[item.lower()] = front_view
                    else:
                        # 可能是分类目录
                        for sub_item in os.listdir(item_path):
                            sub_item_path = os.path.join(item_path, sub_item)
                            if os.path.isdir(sub_item_path):
                                sub_image_dir = os.path.join(sub_item_path, "image")
                                if os.path.exists(sub_image_dir):
                                    front_view = os.path.join(sub_image_dir, FRONT_VIEW_FILENAME)
                                    if os.path.exists(front_view):
                                        self._mn40_index[sub_item.lower()] = front_view
    
    def _build_ntu_index(self, base_path: str):
        """构建 NTU 数据集索引"""
        # NTU 结构类似 MN40
        for type_dir in ["target", "query", "train"]:
            type_path = os.path.join(base_path, type_dir)
            if not os.path.exists(type_path):
                continue
            
            for item in os.listdir(type_path):
                item_path = os.path.join(type_path, item)
                if os.path.isdir(item_path):
                    # 检查是否是直接的 ID 目录
                    image_dir = os.path.join(item_path, "image")
                    if os.path.exists(image_dir):
                        # 直接是 ID 目录
                        front_view = os.path.join(image_dir, FRONT_VIEW_FILENAME)
                        if os.path.exists(front_view):
                            self._ntu_index[item.lower()] = front_view
                    else:
                        # 可能是分类目录
                        for sub_item in os.listdir(item_path):
                            sub_item_path = os.path.join(item_path, sub_item)
                            if os.path.isdir(sub_item_path):
                                sub_image_dir = os.path.join(sub_item_path, "image")
                                if os.path.exists(sub_image_dir):
                                    front_view = os.path.join(sub_image_dir, FRONT_VIEW_FILENAME)
                                    if os.path.exists(front_view):
                                        self._ntu_index[sub_item.lower()] = front_view
    
    def _build_gso_index(self, base_path: str):
        """构建 GSO 数据集索引"""
        # GSO 结构: GSO_resample/{name}/thumbnails/0.jpg
        # GSO 的 ID 格式与其他数据集不同，可能是物品名称
        if not os.path.exists(base_path):
            return
        
        for item in os.listdir(base_path):
            item_path = os.path.join(base_path, item)
            if os.path.isdir(item_path):
                thumbnails_dir = os.path.join(item_path, "thumbnails")
                if os.path.exists(thumbnails_dir):
                    front_view = os.path.join(thumbnails_dir, GSO_FRONT_VIEW_FILENAME)
                    if os.path.exists(front_view):
                        # 使用目录名作为索引键
                        self._gso_index[item.lower()] = front_view
    
    def find_image_path(self, item_id: str, dataset_type: str) -> Optional[str]:
        """
        根据 ID 和数据集类型查找图片路径
        
        Args:
            item_id: 数据项 ID
            dataset_type: 数据集类型（ESB, GSO, MN40, NTU）
            
        Returns:
            图片路径，如果找不到返回 None
        """
        cache_key = f"{dataset_type}:{item_id}"
        if cache_key in self._path_cache:
            return self._path_cache[cache_key]
        
        result = None
        id_lower = item_id.lower()
        
        if dataset_type == "ESB":
            result = self._esb_index.get(id_lower)
        elif dataset_type == "MN40":
            result = self._mn40_index.get(id_lower)
        elif dataset_type == "NTU":
            result = self._ntu_index.get(id_lower)
        elif dataset_type == "GSO":
            result = self._gso_index.get(id_lower)
        
        if result:
            self._path_cache[cache_key] = result
        
        return result
    
    def get_index_stats(self) -> Dict[str, int]:
        """获取索引统计信息"""
        return {
            "ESB": len(self._esb_index),
            "MN40": len(self._mn40_index),
            "NTU": len(self._ntu_index),
            "GSO": len(self._gso_index),
        }


class DataLoader:
    """数据加载器"""
    
    def __init__(self, eval_root: str = EVAL_ROOT):
        """
        初始化数据加载器
        
        Args:
            eval_root: 评测数据集根目录
        """
        self.eval_root = eval_root
        self.path_resolver = DatasetPathResolver(eval_root)
    
    def detect_dataset_type(self, json_path: str) -> str:
        """
        从 JSON 文件路径检测数据集类型
        
        Args:
            json_path: JSON 文件路径
            
        Returns:
            数据集类型（ESB, GSO, MN40, NTU）
        """
        # 从路径中检测
        path_lower = json_path.lower()
        
        # 检查路径中是否包含数据集名称
        if "esb" in path_lower or "\\esb\\" in path_lower:
            return "ESB"
        elif "gso" in path_lower or "\\gso\\" in path_lower:
            return "GSO"
        elif "mn40" in path_lower or "\\mn40\\" in path_lower:
            return "MN40"
        elif "ntu" in path_lower or "\\ntu\\" in path_lower:
            return "NTU"
        
        # 从文件名检测
        filename = os.path.basename(json_path).lower()
        
        if filename.startswith("search_lists"):
            return "ESB"
        elif filename.startswith("image_lists"):
            return "GSO"
        elif filename.startswith("query_"):
            return "MN40"
        elif filename.startswith("search_queries"):
            return "NTU"
        
        # 默认尝试从 JSON 内容中的 source_file 检测
        return "UNKNOWN"
    
    def detect_dataset_from_source_file(self, source_file: str) -> str:
        """
        从 source_file 字段检测数据集类型
        
        Args:
            source_file: JSON 中的 source_file 字段值
            
        Returns:
            数据集类型
        """
        source_lower = source_file.lower()
        
        if "search_list" in source_lower:
            return "ESB"
        elif "image_lists" in source_lower or "bag" in source_lower or "shoe" in source_lower or "toys" in source_lower:
            return "GSO"
        elif "query_" in source_lower and ("airplane" in source_lower or "chair" in source_lower or 
                                           "sofa" in source_lower or "desk" in source_lower or
                                           any(cat in source_lower for cat in ["person", "vase", "stool", "piano", "cup", "door", "flower_pot", "monitor", "bench", "range_hood", "mantel", "bed", "table", "xbox", "plant", "tent", "car", "wardrobe", "tv_stand", "bottle", "stairs", "sink", "cone", "night_stand", "curtain", "radio", "bathtub", "glass_box", "lamp", "dresser", "bowl", "keyboard"])):
            return "MN40"
        elif "search_queries" in source_lower:
            return "NTU"
        
        return "UNKNOWN"
    
    def load_json(self, json_path: str, use_first_as_query: bool = True) -> LoadedData:
        """
        加载 JSON 文件并解析数据
        
        Args:
            json_path: JSON 文件路径
            use_first_as_query: 是否将第一项作为 query（分数最高的项）
            
        Returns:
            LoadedData 对象
        """
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON 文件不存在: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        source_file = data.get("source_file", "")
        total_count = data.get("total_count", 0)
        items = data.get("data", [])
        
        # 检测数据集类型
        dataset_type = self.detect_dataset_type(json_path)
        if dataset_type == "UNKNOWN":
            dataset_type = self.detect_dataset_from_source_file(source_file)
        
        if dataset_type == "UNKNOWN":
            print(f"⚠️  无法检测数据集类型，将尝试所有数据集")
        
        # 解析数据项
        data_items = []
        for item in items:
            item_id = item.get("id", "")
            score = item.get("score", 0.0)
            
            # 查找图片路径
            image_path = None
            error_message = ""
            
            if dataset_type != "UNKNOWN":
                image_path = self.path_resolver.find_image_path(item_id, dataset_type)
            else:
                # 尝试所有数据集
                for ds_type in ["ESB", "MN40", "NTU", "GSO"]:
                    image_path = self.path_resolver.find_image_path(item_id, ds_type)
                    if image_path:
                        break
            
            if not image_path:
                error_message = f"未找到 ID '{item_id}' 对应的图片"
            
            data_items.append(DataItem(
                id=item_id,
                score=score,
                image_path=image_path,
                is_valid=image_path is not None,
                error_message=error_message
            ))
        
        # 分离 query 和 candidates
        query_item = None
        candidate_items = []
        
        if use_first_as_query and data_items:
            query_item = data_items[0]
            candidate_items = data_items[1:]
        else:
            candidate_items = data_items
        
        # 统计有效/无效数量
        valid_count = sum(1 for item in data_items if item.is_valid)
        invalid_count = len(data_items) - valid_count
        
        return LoadedData(
            dataset_type=dataset_type,
            source_file=source_file,
            total_count=total_count,
            query_item=query_item,
            candidate_items=candidate_items,
            valid_count=valid_count,
            invalid_count=invalid_count
        )
    
    def prepare_for_ranking(self, loaded_data: LoadedData, 
                           max_candidates: int = None,
                           only_valid: bool = True) -> Tuple[str, Dict[str, str]]:
        """
        准备排序所需的数据格式
        
        Args:
            loaded_data: 加载的数据
            max_candidates: 最大候选数量限制
            only_valid: 是否只包含有效（有图片路径）的项
            
        Returns:
            (query_image_path, candidate_images_dict)
        """
        if not loaded_data.query_item or not loaded_data.query_item.is_valid:
            raise ValueError("没有有效的 query 项")
        
        query_image = loaded_data.query_item.image_path
        
        # 筛选候选项
        candidates = loaded_data.candidate_items
        if only_valid:
            candidates = [c for c in candidates if c.is_valid]
        
        if max_candidates:
            candidates = candidates[:max_candidates]
        
        # 构建候选图片字典
        candidate_images = {}
        for item in candidates:
            if item.is_valid:
                candidate_images[item.id] = item.image_path
        
        return query_image, candidate_images
    
    def get_ground_truth_ranking(self, loaded_data: LoadedData, 
                                  only_valid: bool = True) -> List[Tuple[str, float]]:
        """
        获取 ground truth 排序（基于分数）
        
        Args:
            loaded_data: 加载的数据
            only_valid: 是否只包含有效项
            
        Returns:
            按分数降序排列的 (id, score) 列表
        """
        items = loaded_data.candidate_items
        if only_valid:
            items = [item for item in items if item.is_valid]
        
        # 按分数降序排序
        sorted_items = sorted(items, key=lambda x: x.score, reverse=True)
        return [(item.id, item.score) for item in sorted_items]


def print_loaded_data_summary(loaded_data: LoadedData):
    """打印加载数据的摘要信息"""
    print("\n" + "=" * 60)
    print("📊 数据加载摘要")
    print("=" * 60)
    print(f"  数据集类型: {loaded_data.dataset_type}")
    print(f"  源文件: {loaded_data.source_file}")
    print(f"  总数据量: {loaded_data.total_count}")
    print(f"  有效项: {loaded_data.valid_count}")
    print(f"  无效项: {loaded_data.invalid_count}")
    
    if loaded_data.query_item:
        print(f"\n📷 Query 项:")
        print(f"    ID: {loaded_data.query_item.id}")
        print(f"    分数: {loaded_data.query_item.score}")
        print(f"    有效: {'✅' if loaded_data.query_item.is_valid else '❌'}")
        if loaded_data.query_item.image_path:
            print(f"    路径: {loaded_data.query_item.image_path}")
    
    print(f"\n📦 Candidate 项 (前5个):")
    for i, item in enumerate(loaded_data.candidate_items[:5]):
        status = '✅' if item.is_valid else '❌'
        print(f"    {i+1}. {status} {item.id} (分数: {item.score})")
    
    if len(loaded_data.candidate_items) > 5:
        print(f"    ... 还有 {len(loaded_data.candidate_items) - 5} 项")


def demo():
    """演示数据加载功能"""
    print("\n" + "=" * 70)
    print("🎯 数据加载模块演示")
    print("=" * 70)
    
    # 初始化数据加载器
    loader = DataLoader()
    
    # 打印索引统计
    stats = loader.path_resolver.get_index_stats()
    print("\n📊 数据集索引统计:")
    for ds_type, count in stats.items():
        print(f"    {ds_type}: {count} 项")
    
    # 测试加载 ESB 数据
    test_json = r"d:\3d-object-数据集\1000条黄金数据\ESB\search_lists_01_search_list_01_adjusted_id_score.json"
    
    if os.path.exists(test_json):
        print(f"\n📂 测试加载: {test_json}")
        loaded_data = loader.load_json(test_json)
        print_loaded_data_summary(loaded_data)
        
        # 准备排序数据
        try:
            query_image, candidate_images = loader.prepare_for_ranking(loaded_data, max_candidates=10)
            print(f"\n✅ 成功准备排序数据:")
            print(f"    Query 图片: {query_image}")
            print(f"    Candidate 数量: {len(candidate_images)}")
        except ValueError as e:
            print(f"\n❌ 准备数据失败: {e}")
    else:
        print(f"\n⚠️  测试文件不存在: {test_json}")


if __name__ == "__main__":
    demo()
