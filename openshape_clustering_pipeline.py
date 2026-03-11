#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenShape 聚类精排 Pipeline

完整流程：
1. 从 glb 文件提取点云，使用 OpenShape 编码成 embedding
2. 对 embedding 进行聚类分析，每个 cluster 至少包含 30 个物体
3. 为每个 cluster 生成精排 case（基于 agent 的精排 + 余弦相似度补充）
4. 保存最终结果

Author: Auto-generated
Date: 2024-02-09
"""

import os
import sys
import json
import random
import pickle
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from tqdm import tqdm

import torch
import torch.nn.functional as F
import trimesh
import open3d as o3d
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import normalize

# 添加 OpenShape 代码路径
sys.path.insert(0, '/apdcephfs/share_303565425/DCC3/serenasnliu/GAR/OpenShape_code/src')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from collections import OrderedDict
import re

try:
    from param import parse_args
    import models
    import MinkowskiEngine as ME
    from utils.data import normalize_pc
    from utils.misc import load_config
    from huggingface_hub import hf_hub_download
    OPENSHAPE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ OpenShape 依赖未完全安装: {e}")
    OPENSHAPE_AVAILABLE = False


# ==================== 配置 ====================

# GLB 文件基础路径
GLB_BASE_PATH = "/apdcephfs/share_303565425/DCC3/data_all/glbs_objaverse_all"

# OpenShape 配置
OPENSHAPE_CONFIG = '/apdcephfs/share_303565425/DCC3/serenasnliu/GAR/OpenShape_code/src/configs/train.yaml'
OPENSHAPE_MODEL = 'OpenShape/openshape-spconv-all'

# 点云参数
NUM_POINTS = 10000

# 聚类参数
MIN_CLUSTER_SIZE = 30  # 每个 cluster 最少物体数量
DEFAULT_NUM_CLUSTERS = 100  # 默认聚类数量

# 精排参数
RANKING_BATCH_SIZE = 20  # 精排物体总数（前10个来自同cluster相似度最高，后10个从相邻cluster随机选取）
TOP_SIMILAR_COUNT = 10  # 从同cluster选取相似度最高的物体数
NEIGHBOR_RANDOM_COUNT = 10  # 从相邻cluster随机选取的物体数（参与精排）
DISTANT_CLUSTER_OBJECTS = 20  # 从较远cluster随机选取的物体数（不参与精排）
OTHER_CATEGORY_OBJECTS = 9  # 从其他类别随机选取的物体数（不参与精排）

# 所有支持的类别及默认缓存基础目录
ALL_CATEGORIES = ["Character", "Object", "Building", "Weapon", "Vehicle", "Animal"]
DEFAULT_CACHE_BASE_DIR = "/apdcephfs/share_303565425/DCC3/serenasnliu/GAR/Uni3D/data/test_datasets/objaverse"

# 其他类别数据源
CATEGORIZED_JSON_PATH = "/apdcephfs/share_303565425/DCC3/serenasnliu/GAR/Uni3D/data/test_datasets/objaverse/categorized_objaverse_golden.json"

# 专门的输出目录
FINAL_OUTPUT_DIR = "/apdcephfs/share_303565425/DCC3/serenasnliu/GAR/Uni3D/data/test_datasets/objaverse/openshape_ranking_results_qwen3.5max/character"
INTERMEDIATE_CACHE_DIR = "/apdcephfs/share_303565425/DCC3/serenasnliu/GAR/Uni3D/data/test_datasets/objaverse/openshape_intermediate_cache"


# ==================== 数据结构 ====================

@dataclass
class ObjectInfo:
    """单个 3D 物体信息"""
    object_id: str
    mesh_path: str
    image_path: str
    description: str
    category: str
    llm_category: str = ""
    embedding: Optional[np.ndarray] = None
    cluster_id: int = -1
    
    def get_glb_path(self) -> str:
        """获取 GLB 文件完整路径"""
        # mesh_path 格式: objaverse/hf-objaverse-v1/000-117/xxx
        # GLB 路径格式: GLB_BASE_PATH/hf-objaverse-v1/000-117/xxx/xxx.glb
        rel_path = self.mesh_path.replace("objaverse/", "")
        obj_name = os.path.basename(rel_path)
        return os.path.join(GLB_BASE_PATH, rel_path, f"{obj_name}.glb")


@dataclass
class ClusterInfo:
    """聚类信息"""
    cluster_id: int
    centroid: np.ndarray
    object_ids: List[str] = field(default_factory=list)
    neighbor_cluster_ids: List[int] = field(default_factory=list)  # 按距离排序的邻居 cluster
    
    @property
    def size(self) -> int:
        return len(self.object_ids)


@dataclass
class RankingCase:
    """单个排序 case"""
    case_id: str
    query_object_id: str
    cluster_id: int
    category: str
    
    # 精排结果（20个物体：10个同cluster相似度最高 + 10个相邻cluster随机）
    reranked_objects: List[str] = field(default_factory=list)  # 精排后的排序结果
    
    # 精排前的候选物体来源
    top_similar_objects: List[str] = field(default_factory=list)  # 同cluster相似度最高的10个
    neighbor_random_objects: List[str] = field(default_factory=list)  # 相邻cluster随机选取的10个
    
    # 精排前的余弦相似度排序（用于对比）
    pre_rerank_cosine_order: List[str] = field(default_factory=list)  # 精排前按余弦相似度排序的20个物体
    pre_rerank_cosine_scores: Dict[str, float] = field(default_factory=dict)  # 精排前每个物体的余弦相似度分数
    
    # 较远cluster的物体（不参与精排，按余弦相似度排序）
    distant_ranking: List[str] = field(default_factory=list)  # 较远cluster的20个物体
    
    # 其他类别的物体（不参与精排，按随机顺序）
    other_category_objects: List[str] = field(default_factory=list)  # 其他类别的9个物体
    
    # 合并后的最终排序（包含 query 本身）
    # 结构: [query] + [精排20个] + [较远cluster20个] + [其他类别9个] = 50个物体
    final_ranking: List[str] = field(default_factory=list)
    
    # 加权分数（来自 agent 精排系统，仅对精排的20个物体有效）
    # 格式: {object_id: {"total_score": float, "dimension_scores": {...}}}
    weighted_scores: Dict[str, Dict] = field(default_factory=dict)
    
    # 较远cluster物体的余弦相似度分数
    # 格式: {object_id: float}
    distant_similarity_scores: Dict[str, float] = field(default_factory=dict)
    
    # 其他类别物体的信息
    # 格式: {object_id: {"category": str, ...}}
    other_category_info: Dict[str, Dict] = field(default_factory=dict)
    
    # 元数据
    objects_info: Dict[str, Dict] = field(default_factory=dict)  # 每个物体的详细信息


# ==================== 点云提取 ====================

def extract_pointcloud_from_glb(glb_path: str, num_points: int = 10000) -> Optional[np.ndarray]:
    """
    从 GLB 文件提取点云（包含颜色）
    
    Args:
        glb_path: GLB 文件路径
        num_points: 采样点数
        
    Returns:
        点云数组 [num_points, 6]，包含 xyz + rgb
    """
    if not os.path.exists(glb_path):
        return None
    
    try:
        # 使用 trimesh 加载 GLB 文件
        mesh = trimesh.load(glb_path, force='mesh')
        
        if isinstance(mesh, trimesh.Scene):
            # 如果是场景，合并所有 mesh
            meshes = []
            for name, geom in mesh.geometry.items():
                if isinstance(geom, trimesh.Trimesh):
                    meshes.append(geom)
            if not meshes:
                return None
            mesh = trimesh.util.concatenate(meshes)
        
        if not isinstance(mesh, trimesh.Trimesh):
            return None
        
        # 采样点云
        points, face_indices = mesh.sample(num_points, return_index=True)
        
        # 获取颜色
        if mesh.visual.kind == 'vertex':
            # 顶点颜色
            colors = mesh.visual.vertex_colors[mesh.faces[face_indices]].mean(axis=1)[:, :3] / 255.0
        elif mesh.visual.kind == 'texture':
            # 纹理颜色
            try:
                colors = mesh.visual.to_color().vertex_colors[mesh.faces[face_indices]].mean(axis=1)[:, :3] / 255.0
            except:
                colors = np.ones((num_points, 3)) * 0.5  # 默认灰色
        else:
            # 默认颜色
            colors = np.ones((num_points, 3)) * 0.5
        
        # 合并 xyz 和 rgb
        pointcloud = np.concatenate([points, colors], axis=1)
        
        return pointcloud.astype(np.float32)
        
    except Exception as e:
        print(f"⚠️ 提取点云失败 {glb_path}: {e}")
        return None


# ==================== OpenShape 编码 ====================

class OpenShapeEncoder:
    """OpenShape 点云编码器"""
    
    def __init__(self, config_path: str = OPENSHAPE_CONFIG, model_name: str = OPENSHAPE_MODEL):
        """
        初始化 OpenShape 编码器
        
        Args:
            config_path: 配置文件路径
            model_name: HuggingFace 模型名称
        """
        self.config_path = config_path
        self.model_name = model_name
        self.model = None
        self.config = None
        
        self._load_model()
    
    def _load_model(self):
        """加载 OpenShape 模型"""
        if not OPENSHAPE_AVAILABLE:
            raise RuntimeError("OpenShape 依赖未安装")
        
        print("\n加载 OpenShape 模型...")
        
        cli_args, extras = parse_args([])
        self.config = load_config(self.config_path, cli_args=vars(cli_args), extra_args=extras)
        
        self.model = models.make(self.config).cuda()
        
        if self.config.model.name.startswith('Mink'):
            self.model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(self.model)
        else:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        
        print(f"  从 HuggingFace 下载模型: {self.model_name}")
        checkpoint = torch.load(hf_hub_download(repo_id=self.model_name, filename="model.pt"))
        
        model_dict = OrderedDict()
        pattern = re.compile('module.')
        for k, v in checkpoint['state_dict'].items():
            if re.search("module", k):
                model_dict[re.sub(pattern, '', k)] = v
            else:
                model_dict[k] = v
        
        self.model.load_state_dict(model_dict)
        self.model.eval()
        
        print("  OpenShape 模型加载完成！")
    
    def _prepare_pointcloud(self, pointcloud: np.ndarray, y_up: bool = True):
        """
        准备点云数据用于 OpenShape 编码
        
        Args:
            pointcloud: 点云数组 [N, 6] (xyz + rgb)
            y_up: 是否将 Y 轴朝上
            
        Returns:
            (xyz_coords, features) 用于 MinkowskiEngine
        """
        xyz = pointcloud[:, :3].copy()
        rgb = pointcloud[:, 3:6].copy()
        
        if y_up:
            # 交换 Y 和 Z 轴
            xyz[:, [1, 2]] = xyz[:, [2, 1]]
        
        # 归一化点云
        xyz = normalize_pc(xyz)
        
        # 组合特征
        features = np.concatenate([xyz, rgb], axis=1)
        
        xyz = torch.from_numpy(xyz).type(torch.float32)
        features = torch.from_numpy(features).type(torch.float32)
        
        return ME.utils.batched_coordinates([xyz], dtype=torch.float32), features
    
    @torch.no_grad()
    def encode(self, pointcloud: np.ndarray) -> np.ndarray:
        """
        编码单个点云
        
        Args:
            pointcloud: 点云数组 [N, 6]
            
        Returns:
            embedding 向量 [D,]
        """
        xyz, features = self._prepare_pointcloud(pointcloud)
        xyz = xyz.cuda()
        features = features.cuda()
        
        shape_feat = self.model(
            xyz, 
            features, 
            device='cuda', 
            quantization_size=self.config.model.voxel_size
        )
        
        shape_feat = F.normalize(shape_feat, p=2, dim=1)
        
        return shape_feat.cpu().numpy().squeeze()
    
    @torch.no_grad()
    def encode_batch(self, pointclouds: List[np.ndarray]) -> np.ndarray:
        """
        批量编码点云（由于 MinkowskiEngine 限制，实际逐个处理）
        
        Args:
            pointclouds: 点云列表
            
        Returns:
            embeddings 数组 [N, D]
        """
        embeddings = []
        for pc in tqdm(pointclouds, desc="编码点云"):
            emb = self.encode(pc)
            embeddings.append(emb)
        return np.array(embeddings)


# ==================== 数据管理 ====================

class EmbeddingManager:
    """Embedding 持久化管理"""
    
    def __init__(self, cache_dir: str):
        """
        初始化 Embedding 管理器
        
        Args:
            cache_dir: 缓存目录
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        self.embeddings_file = os.path.join(cache_dir, "openshape_embeddings.npz")
        self.metadata_file = os.path.join(cache_dir, "openshape_metadata.json")
    
    def save_embeddings(
        self, 
        object_ids: List[str], 
        embeddings: np.ndarray,
        metadata: Dict[str, Any] = None
    ):
        """
        保存 embeddings
        
        Args:
            object_ids: 物体 ID 列表
            embeddings: embedding 数组
            metadata: 额外元数据
        """
        np.savez(
            self.embeddings_file,
            object_ids=np.array(object_ids),
            embeddings=embeddings
        )
        
        if metadata is None:
            metadata = {}
        metadata["saved_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metadata["num_objects"] = len(object_ids)
        metadata["embedding_dim"] = embeddings.shape[1] if len(embeddings.shape) > 1 else 0
        
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Embeddings 已保存到: {self.embeddings_file}")
    
    def load_embeddings(self) -> Tuple[List[str], np.ndarray, Dict]:
        """
        加载 embeddings
        
        Returns:
            (object_ids, embeddings, metadata)
        """
        if not os.path.exists(self.embeddings_file):
            return [], np.array([]), {}
        
        data = np.load(self.embeddings_file, allow_pickle=True)
        object_ids = data['object_ids'].tolist()
        embeddings = data['embeddings']
        
        metadata = {}
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        
        print(f"✅ 已加载 {len(object_ids)} 个 embeddings")
        return object_ids, embeddings, metadata
    
    def has_cache(self) -> bool:
        """检查是否有缓存"""
        return os.path.exists(self.embeddings_file)


class ClusterManager:
    """聚类结果管理"""
    
    def __init__(self, cache_dir: str):
        """
        初始化聚类管理器
        
        Args:
            cache_dir: 缓存目录
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        self.clusters_file = os.path.join(cache_dir, "clusters.pkl")
        self.cluster_info_file = os.path.join(cache_dir, "cluster_info.json")
    
    def save_clusters(
        self,
        clusters: Dict[int, ClusterInfo],
        object_cluster_map: Dict[str, int],
        metadata: Dict[str, Any] = None
    ):
        """
        保存聚类结果
        
        Args:
            clusters: 聚类信息字典
            object_cluster_map: 物体到聚类的映射
            metadata: 额外元数据
        """
        # 保存完整数据
        data = {
            'clusters': clusters,
            'object_cluster_map': object_cluster_map
        }
        with open(self.clusters_file, 'wb') as f:
            pickle.dump(data, f)
        
        # 保存可读的统计信息
        if metadata is None:
            metadata = {}
        
        cluster_stats = {
            'total_clusters': len(clusters),
            'total_objects': len(object_cluster_map),
            'cluster_sizes': {str(k): v.size for k, v in clusters.items()},
            'saved_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        cluster_stats.update(metadata)
        
        with open(self.cluster_info_file, 'w', encoding='utf-8') as f:
            json.dump(cluster_stats, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 聚类结果已保存到: {self.clusters_file}")
    
    def load_clusters(self) -> Tuple[Dict[int, ClusterInfo], Dict[str, int]]:
        """
        加载聚类结果
        
        Returns:
            (clusters, object_cluster_map)
        """
        if not os.path.exists(self.clusters_file):
            return {}, {}
        
        with open(self.clusters_file, 'rb') as f:
            data = pickle.load(f)
        
        print(f"✅ 已加载 {len(data['clusters'])} 个聚类")
        return data['clusters'], data['object_cluster_map']
    
    def has_cache(self) -> bool:
        """检查是否有缓存"""
        return os.path.exists(self.clusters_file)


# ==================== 聚类算法 ====================

def perform_clustering(
    embeddings: np.ndarray,
    object_ids: List[str],
    min_cluster_size: int = MIN_CLUSTER_SIZE,
    target_num_clusters: int = None
) -> Tuple[Dict[int, ClusterInfo], Dict[str, int]]:
    """
    对 embeddings 进行聚类
    
    Args:
        embeddings: embedding 数组 [N, D]
        object_ids: 物体 ID 列表
        min_cluster_size: 每个 cluster 最少物体数
        target_num_clusters: 目标聚类数（None 则自动计算）
        
    Returns:
        (clusters, object_cluster_map)
    """
    print("\n" + "=" * 60)
    print("开始聚类分析")
    print("=" * 60)
    
    n_samples = len(object_ids)
    print(f"总物体数: {n_samples}")
    print(f"最小聚类大小: {min_cluster_size}")
    
    # 归一化 embeddings
    embeddings_norm = normalize(embeddings, norm='l2', axis=1)
    
    # 计算目标聚类数
    if target_num_clusters is None:
        # 保守估计：确保平均每个聚类至少有 min_cluster_size 个物体
        target_num_clusters = max(10, n_samples // (min_cluster_size * 2))
    
    print(f"目标聚类数: {target_num_clusters}")
    
    # 使用 KMeans 聚类
    print("运行 KMeans 聚类...")
    kmeans = KMeans(
        n_clusters=target_num_clusters,
        random_state=42,
        n_init=10,
        max_iter=300
    )
    cluster_labels = kmeans.fit_predict(embeddings_norm)
    centroids = kmeans.cluster_centers_
    
    # 统计每个聚类的大小
    cluster_sizes = defaultdict(list)
    for obj_id, label in zip(object_ids, cluster_labels):
        cluster_sizes[label].append(obj_id)
    
    print(f"\n初始聚类结果:")
    print(f"  聚类数: {len(cluster_sizes)}")
    sizes = [len(v) for v in cluster_sizes.values()]
    print(f"  聚类大小范围: {min(sizes)} - {max(sizes)}")
    print(f"  平均大小: {np.mean(sizes):.1f}")
    
    # 合并过小的聚类
    print("\n合并过小的聚类...")
    clusters, object_cluster_map = merge_small_clusters(
        cluster_sizes,
        centroids,
        embeddings_norm,
        object_ids,
        cluster_labels,
        min_cluster_size
    )
    
    # 计算每个聚类的邻居
    print("\n计算聚类邻居关系...")
    compute_cluster_neighbors(clusters)
    
    # 打印最终统计
    print(f"\n最终聚类结果:")
    print(f"  聚类数: {len(clusters)}")
    sizes = [c.size for c in clusters.values()]
    print(f"  聚类大小范围: {min(sizes)} - {max(sizes)}")
    print(f"  平均大小: {np.mean(sizes):.1f}")
    
    return clusters, object_cluster_map


def merge_small_clusters(
    cluster_sizes: Dict[int, List[str]],
    centroids: np.ndarray,
    embeddings_norm: np.ndarray,
    object_ids: List[str],
    cluster_labels: np.ndarray,
    min_size: int
) -> Tuple[Dict[int, ClusterInfo], Dict[str, int]]:
    """
    合并过小的聚类到最近的大聚类
    
    Args:
        cluster_sizes: 初始聚类大小
        centroids: 聚类中心
        embeddings_norm: 归一化后的 embeddings
        object_ids: 物体 ID 列表
        cluster_labels: 初始聚类标签
        min_size: 最小聚类大小
        
    Returns:
        (clusters, object_cluster_map)
    """
    # 找出大聚类和小聚类
    large_clusters = {k: v for k, v in cluster_sizes.items() if len(v) >= min_size}
    small_clusters = {k: v for k, v in cluster_sizes.items() if len(v) < min_size}
    
    print(f"  大聚类数 (>= {min_size}): {len(large_clusters)}")
    print(f"  小聚类数 (< {min_size}): {len(small_clusters)}")
    
    if not large_clusters:
        # 如果没有足够大的聚类，降低阈值
        print("  ⚠️ 没有足够大的聚类，将所有聚类视为有效")
        large_clusters = cluster_sizes.copy()
        small_clusters = {}
    
    # 合并小聚类
    large_centroids = np.array([centroids[k] for k in large_clusters.keys()])
    large_cluster_ids = list(large_clusters.keys())
    
    merged_clusters = {k: list(v) for k, v in large_clusters.items()}
    
    for small_id, small_objects in small_clusters.items():
        # 找最近的大聚类
        small_centroid = centroids[small_id:small_id+1]
        distances = 1 - np.dot(large_centroids, small_centroid.T).squeeze()
        nearest_large_id = large_cluster_ids[np.argmin(distances)]
        
        # 合并
        merged_clusters[nearest_large_id].extend(small_objects)
    
    # 重新编号并创建 ClusterInfo
    clusters = {}
    object_cluster_map = {}
    
    # 创建物体ID到embedding的映射
    obj_to_emb = {obj_id: emb for obj_id, emb in zip(object_ids, embeddings_norm)}
    
    for new_id, (old_id, obj_list) in enumerate(merged_clusters.items()):
        # 重新计算聚类中心
        cluster_embs = np.array([obj_to_emb[obj_id] for obj_id in obj_list])
        centroid = cluster_embs.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
        
        clusters[new_id] = ClusterInfo(
            cluster_id=new_id,
            centroid=centroid,
            object_ids=obj_list
        )
        
        for obj_id in obj_list:
            object_cluster_map[obj_id] = new_id
    
    return clusters, object_cluster_map


def compute_cluster_neighbors(clusters: Dict[int, ClusterInfo]):
    """
    计算每个聚类的邻居（按质心距离排序）
    
    Args:
        clusters: 聚类信息字典（原地修改）
    """
    cluster_ids = list(clusters.keys())
    centroids = np.array([clusters[cid].centroid for cid in cluster_ids])
    
    # 计算所有质心间的余弦相似度
    similarity_matrix = np.dot(centroids, centroids.T)
    
    for i, cid in enumerate(cluster_ids):
        # 按相似度降序排列（排除自己）
        similarities = similarity_matrix[i]
        sorted_indices = np.argsort(similarities)[::-1]
        
        neighbors = []
        for idx in sorted_indices:
            neighbor_id = cluster_ids[idx]
            if neighbor_id != cid:
                neighbors.append(neighbor_id)
        
        clusters[cid].neighbor_cluster_ids = neighbors


# ==================== 精排 Case 生成 ====================

def compute_cosine_similarity(query_emb: np.ndarray, candidate_embs: np.ndarray) -> np.ndarray:
    """
    计算余弦相似度
    
    Args:
        query_emb: 查询 embedding [D,]
        candidate_embs: 候选 embeddings [N, D]
        
    Returns:
        相似度数组 [N,]
    """
    query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
    candidate_norms = candidate_embs / (np.linalg.norm(candidate_embs, axis=1, keepdims=True) + 1e-8)
    return np.dot(candidate_norms, query_norm)


def generate_ranking_case(
    query_obj: ObjectInfo,
    cluster: ClusterInfo,
    clusters: Dict[int, ClusterInfo],
    objects_dict: Dict[str, ObjectInfo],
    embeddings_dict: Dict[str, np.ndarray],
    executor = None,
    use_agent_ranking: bool = True
) -> RankingCase:
    """
    生成单个排序 case
    
    新流程（按需求修改）：
    1. 从同 cluster 中选取与 query 余弦相似度最高的 10 个物体
    2. 从相邻 cluster 中随机选取 10 个物体
    3. 这 20 个物体 + query 送入 agent 精排
    4. 从距离较远的不相邻 cluster 中随机选取 20 个物体（不参与精排，按余弦相似度排序）
    5. 将 query 本身放到最前面
    6. 最终结构: [query] + [精排20个] + [较远cluster20个] = 41个物体
    
    Args:
        query_obj: 查询物体
        cluster: 所属聚类
        clusters: 所有聚类信息
        objects_dict: 所有物体信息
        embeddings_dict: 所有 embeddings
        executor: GroupRankingExecutor（可选，用于 agent 精排）
        use_agent_ranking: 是否使用 agent 精排
        
    Returns:
        RankingCase 对象
    """
    query_id = query_obj.object_id
    query_emb = embeddings_dict[query_id]
    
    # ========== 步骤1: 从同 cluster 中选取相似度最高的 10 个物体 ==========
    cluster_objects = [oid for oid in cluster.object_ids if oid != query_id]
    
    if len(cluster_objects) < TOP_SIMILAR_COUNT:
        print(f"  ⚠️ Cluster {cluster.cluster_id} 物体数不足 {TOP_SIMILAR_COUNT}，实际: {len(cluster_objects)}")
    
    # 计算与 query 的余弦相似度
    cluster_embs = np.array([embeddings_dict[oid] for oid in cluster_objects])
    similarities = compute_cosine_similarity(query_emb, cluster_embs)
    
    # 按相似度排序，选取前 10 个
    sorted_indices = np.argsort(similarities)[::-1]
    top_similar_objects = [cluster_objects[i] for i in sorted_indices[:TOP_SIMILAR_COUNT]]
    
    # ========== 步骤2: 从相邻 cluster 中随机选取 10 个物体（参与精排） ==========
    # Dynamically split neighbor clusters between neighbor (for reranking) and distant
    # Ensure distant always gets at least some clusters when possible
    total_neighbors = len(cluster.neighbor_cluster_ids)
    if total_neighbors <= 2:
        # Very few clusters: use 1 for neighbor, rest for distant
        neighbor_split = min(1, total_neighbors)
    elif total_neighbors <= 4:
        # Few clusters: use 2 for neighbor, rest for distant
        neighbor_split = 2
    elif total_neighbors <= 6:
        # Moderate clusters: use 3 for neighbor, rest for distant
        neighbor_split = 3
    else:
        # Enough clusters: use up to 5 for neighbor
        neighbor_split = 5
    
    neighbor_candidates = []
    neighbor_cluster_ids = cluster.neighbor_cluster_ids[:neighbor_split]
    
    for neighbor_cid in neighbor_cluster_ids:
        neighbor_cluster = clusters[neighbor_cid]
        neighbor_candidates.extend(neighbor_cluster.object_ids)
    
    # 随机选取 10 个（如果不足则全部选取）
    if len(neighbor_candidates) >= NEIGHBOR_RANDOM_COUNT:
        neighbor_random_objects = random.sample(neighbor_candidates, NEIGHBOR_RANDOM_COUNT)
    else:
        neighbor_random_objects = neighbor_candidates
        print(f"  ⚠️ 相邻cluster物体数不足 {NEIGHBOR_RANDOM_COUNT}，实际: {len(neighbor_candidates)}")
    
    # ========== 步骤3: 合并候选物体并送入精排 ==========
    # 精排候选：前10个相似度最高 + 后10个相邻cluster随机
    candidates_for_ranking = top_similar_objects + neighbor_random_objects
    
    # ========== 记录精排前的余弦相似度排序（用于对比） ==========
    all_embs = np.array([embeddings_dict[oid] for oid in candidates_for_ranking])
    all_similarities = compute_cosine_similarity(query_emb, all_embs)
    sorted_idx = np.argsort(all_similarities)[::-1]
    
    # 精排前按余弦相似度排序的物体列表
    pre_rerank_cosine_order = [candidates_for_ranking[i] for i in sorted_idx]
    # 精排前每个物体的余弦相似度分数
    pre_rerank_cosine_scores = {candidates_for_ranking[i]: float(all_similarities[i]) for i in range(len(candidates_for_ranking))}
    
    # 精排
    weighted_scores = {}
    if use_agent_ranking and executor is not None:
        # 使用 agent 精排
        reranked_objects, weighted_scores = run_agent_ranking(query_obj, candidates_for_ranking, objects_dict, executor)
    else:
        # 不使用精排时，按余弦相似度对所有候选排序
        reranked_objects = pre_rerank_cosine_order.copy()
    
    # ========== 步骤4: 从较远 cluster 随机选取 20 个物体（不参与精排） ==========
    # 较远 cluster 定义为邻居列表中排在 neighbor_split 之后的 cluster
    distant_cluster_ids = cluster.neighbor_cluster_ids[neighbor_split:]  # 跳过用于精排的邻居cluster
    
    distant_candidates = []
    for distant_cid in distant_cluster_ids:
        distant_cluster = clusters[distant_cid]
        distant_candidates.extend(distant_cluster.object_ids)
        if len(distant_candidates) >= DISTANT_CLUSTER_OBJECTS * 3:  # 多取一些用于后续筛选
            break
    
    # 随机选取 20 个
    if len(distant_candidates) >= DISTANT_CLUSTER_OBJECTS:
        distant_selected = random.sample(distant_candidates, DISTANT_CLUSTER_OBJECTS)
    else:
        distant_selected = list(distant_candidates)
        
        # Fallback: supplement from neighbor clusters (objects not already used)
        if len(distant_selected) < DISTANT_CLUSTER_OBJECTS:
            already_used = set([query_id] + top_similar_objects + neighbor_random_objects + distant_selected)
            supplement_candidates = []
            
            # First, try unused objects from neighbor clusters
            for neighbor_cid in neighbor_cluster_ids:
                neighbor_cluster = clusters[neighbor_cid]
                for oid in neighbor_cluster.object_ids:
                    if oid not in already_used:
                        supplement_candidates.append(oid)
            
            # Then, try unused objects from the same cluster
            for oid in cluster.object_ids:
                if oid not in already_used and oid not in supplement_candidates:
                    supplement_candidates.append(oid)
            
            needed = DISTANT_CLUSTER_OBJECTS - len(distant_selected)
            if supplement_candidates:
                supplement = random.sample(supplement_candidates, min(needed, len(supplement_candidates)))
                distant_selected.extend(supplement)
            
            print(f"  ⚠️ 较远cluster物体数不足 {DISTANT_CLUSTER_OBJECTS}，补充后实际: {len(distant_selected)}")
    
    # 按余弦相似度排序（用于展示）
    distant_similarity_scores = {}
    if distant_selected:
        distant_embs = np.array([embeddings_dict[oid] for oid in distant_selected])
        distant_similarities = compute_cosine_similarity(query_emb, distant_embs)
        distant_sorted_indices = np.argsort(distant_similarities)[::-1]
        distant_ranking = [distant_selected[i] for i in distant_sorted_indices]
        
        # 保存较远物体的余弦相似度分数
        for i, idx in enumerate(distant_sorted_indices):
            obj_id = distant_selected[idx]
            distant_similarity_scores[obj_id] = float(distant_similarities[idx])
    else:
        distant_ranking = []
    
    # ========== 步骤5: 合并最终排序（暂时不包含其他类别物体，由pipeline后续添加） ==========
    # 结构: [query] + [精排20个] + [较远cluster20个] = 41个物体
    # 其他类别物体将在 pipeline 中添加
    final_ranking = [query_id] + reranked_objects + distant_ranking
    
    # 收集所有物体信息
    all_object_ids = set(final_ranking)
    objects_info = {}
    for oid in all_object_ids:
        obj = objects_dict.get(oid)
        if obj:
            objects_info[oid] = {
                'object_id': oid,
                'image_path': obj.image_path,
                'description': obj.description,
                'category': obj.category,
                'mesh_path': obj.mesh_path
            }
    
    return RankingCase(
        case_id=f"case_{query_id}",
        query_object_id=query_id,
        cluster_id=cluster.cluster_id,
        category=query_obj.category,
        reranked_objects=reranked_objects,
        top_similar_objects=top_similar_objects,
        neighbor_random_objects=neighbor_random_objects,
        pre_rerank_cosine_order=pre_rerank_cosine_order,  # 精排前的余弦相似度排序
        pre_rerank_cosine_scores=pre_rerank_cosine_scores,  # 精排前的余弦相似度分数
        distant_ranking=distant_ranking,
        other_category_objects=[],  # 将在 pipeline 中填充
        final_ranking=final_ranking,
        weighted_scores=weighted_scores,
        distant_similarity_scores=distant_similarity_scores,
        other_category_info={},  # 将在 pipeline 中填充
        objects_info=objects_info
    )


def run_agent_ranking(
    query_obj: ObjectInfo,
    candidate_ids: List[str],
    objects_dict: Dict[str, ObjectInfo],
    executor
) -> Tuple[List[str], Dict[str, Dict]]:
    """
    运行 agent 精排
    
    Args:
        query_obj: 查询物体
        candidate_ids: 候选物体 ID 列表
        objects_dict: 物体信息字典
        executor: GroupRankingExecutor
        
    Returns:
        (排序后的物体 ID 列表, 加权分数字典)
        加权分数字典格式: {object_id: {"total_score": float, "dimension_scores": {...}}}
    """
    from group_ranking_skill import ObjectGroup, GroupObject
    
    # 构建临时 group
    objects = [GroupObject(
        object_id=query_obj.object_id,
        image_path=query_obj.image_path,
        description=query_obj.description,
        category=query_obj.category,
        mesh_path=query_obj.mesh_path
    )]
    
    for cid in candidate_ids:
        obj = objects_dict.get(cid)
        if obj:
            objects.append(GroupObject(
                object_id=cid,
                image_path=obj.image_path,
                description=obj.description,
                category=obj.category,
                mesh_path=obj.mesh_path
            ))
    
    temp_group = ObjectGroup(
        group_id=f"temp_{query_obj.object_id}",
        category=query_obj.category,
        group_index=0,
        objects=objects
    )
    
    try:
        result = executor.run_single_group(
            group=temp_group,
            query_index=0  # query 在第一位
        )
        
        # 提取加权分数
        weighted_scores = {}
        pipeline_result = result.pipeline_result
        
        # 从 candidate_reports 中提取分数
        for report in pipeline_result.get('candidate_reports', []):
            obj_id = report.get('candidate_id')
            if obj_id:
                weighted_scores[obj_id] = {
                    'total_score': report.get('weighted_total_score', 0.0),
                    'rank': report.get('rank', 0),
                    'dimension_scores': report.get('dimension_scores', {})
                }
        
        return result.predicted_ranking, weighted_scores
    except Exception as e:
        print(f"  ⚠️ Agent 精排失败: {e}，使用原始顺序")
        return candidate_ids, {}


# ==================== 主流程 ====================

class OpenShapeClusteringPipeline:
    """OpenShape 聚类精排主流程"""
    
    def __init__(
        self,
        input_json: str,
        output_dir: str = None,
        cache_dir: str = None,
        cache_base_dir: str = None,
        intermediate_cache_dir: str = None,
        use_agent_ranking: bool = True,
        llm_mode: str = "api",
        model_name: str = None,
        categorized_json_path: str = None,
        target_category: str = "Character"
    ):
        """
        初始化流程
        
        Args:
            input_json: 输入 JSON 文件（分组数据）
            output_dir: 最终输出目录（默认使用 FINAL_OUTPUT_DIR）
            cache_dir: 缓存目录（如果指定则直接使用，优先级最高）
            cache_base_dir: 缓存基础目录（自动根据 target_category 拼接子目录，如 cache_base_dir/openshape_cache_object/）
            intermediate_cache_dir: 中间结果缓存目录（默认使用 INTERMEDIATE_CACHE_DIR）
            use_agent_ranking: 是否使用 agent 精排
            llm_mode: LLM 模式
            model_name: 模型名称
            categorized_json_path: 分类物体 JSON 文件路径（用于获取其他类别物体）
            target_category: 当前处理的目标类别名称（如 Character, Object, Building 等）
        """
        self.input_json = input_json
        self.target_category = target_category
        self.output_dir = output_dir or FINAL_OUTPUT_DIR
        
        # 保存 cache_base_dir，用于后续加载所有类别的 embedding
        self.cache_base_dir = cache_base_dir or DEFAULT_CACHE_BASE_DIR
        
        # 缓存目录自动推断逻辑：
        # 1. 如果显式指定了 cache_dir，直接使用
        # 2. 否则根据 target_category 自动拼接子目录（复用 _get_category_cache_dir 逻辑）
        if cache_dir:
            self.cache_dir = cache_dir
        else:
            self.cache_dir = self._get_category_cache_dir(target_category)
        
        self.intermediate_cache_dir = intermediate_cache_dir or INTERMEDIATE_CACHE_DIR
        self.use_agent_ranking = use_agent_ranking
        self.llm_mode = llm_mode
        self.model_name = model_name
        self.categorized_json_path = categorized_json_path or CATEGORIZED_JSON_PATH
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.intermediate_cache_dir, exist_ok=True)
        
        # 数据管理器
        self.embedding_manager = EmbeddingManager(self.cache_dir)
        self.cluster_manager = ClusterManager(self.cache_dir)
        
        # 数据存储
        self.objects_dict: Dict[str, ObjectInfo] = {}
        self.embeddings_dict: Dict[str, np.ndarray] = {}
        self.clusters: Dict[int, ClusterInfo] = {}
        self.object_cluster_map: Dict[str, int] = {}
        
        # 其他类别物体数据（非当前目标类别）
        self.other_category_objects: Dict[str, List[Dict]] = {}  # {category: [objects]}
        
        # 所有类别的 embedding 管理器（用于加载其他类别缓存）
        self.all_category_embedding_managers: Dict[str, EmbeddingManager] = {}
        
        # OpenShape 编码器
        self.encoder: Optional[OpenShapeEncoder] = None
        
        # Agent 精排执行器
        self.executor = None
    
    def load_input_data(self):
        """加载输入数据"""
        print("\n" + "=" * 60)
        print("加载输入数据")
        print("=" * 60)
        
        with open(self.input_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 从分组数据中提取所有物体
        for group in data.get('groups', []):
            for obj_data in group.get('objects', []):
                obj = ObjectInfo(
                    object_id=obj_data.get('object_id', ''),
                    mesh_path=obj_data.get('mesh_path', ''),
                    image_path=obj_data.get('image_path', ''),
                    description=obj_data.get('description', ''),
                    category=obj_data.get('category', ''),
                    llm_category=obj_data.get('llm_category', '')
                )
                self.objects_dict[obj.object_id] = obj
        
        print(f"✅ 加载了 {len(self.objects_dict)} 个物体")
        
        # 加载其他类别物体数据
        self._load_other_category_objects()
        
        # 自动从所有类别缓存子目录加载全部 embedding
        self._load_all_category_embeddings()
    
    def _get_category_cache_dir(self, category: str) -> str:
        """
        根据类别名获取对应的缓存目录路径
        
        Args:
            category: 类别名称
            
        Returns:
            缓存目录路径
        """
        category_lower = category.lower()
        if category_lower == "character":
            return os.path.join(self.cache_base_dir, "openshape_cache")
        else:
            return os.path.join(self.cache_base_dir, f"openshape_cache_{category_lower}")
    
    def _load_all_category_embeddings(self):
        """
        自动从所有类别的缓存子目录中提取全部 embedding。
        遍历 ALL_CATEGORIES 中的每个类别，如果对应的缓存目录中存在 embedding 文件，
        则加载并合并到 self.embeddings_dict 中（当前类别的 embedding 已在 extract_and_encode 中加载，
        此方法负责补充加载其他类别的 embedding）。
        """
        print("\n" + "=" * 60)
        print("从所有类别缓存子目录加载全部 embedding")
        print("=" * 60)
        
        loaded_count = 0
        skipped_count = 0
        
        for category in ALL_CATEGORIES:
            cache_dir = self._get_category_cache_dir(category)
            
            # 当前类别的缓存已由 extract_and_encode 加载，这里标记但不跳过
            # （因为 load_input_data 在 extract_and_encode 之前被调用，所以这里需要加载所有类别）
            
            manager = EmbeddingManager(cache_dir)
            self.all_category_embedding_managers[category] = manager
            
            if not manager.has_cache():
                print(f"  ⚠️ {category}: 缓存目录不存在或无 embedding 文件 ({cache_dir})")
                skipped_count += 1
                continue
            
            try:
                object_ids, embeddings, metadata = manager.load_embeddings()
                
                category_loaded = 0
                for i, oid in enumerate(object_ids):
                    if oid not in self.embeddings_dict:
                        self.embeddings_dict[oid] = embeddings[i]
                        category_loaded += 1
                
                loaded_count += category_loaded
                print(f"  ✅ {category}: 加载了 {category_loaded} 个新 embedding（缓存中共 {len(object_ids)} 个，目录: {cache_dir}）")
                
            except Exception as e:
                print(f"  ⚠️ {category}: 加载 embedding 失败: {e}")
                skipped_count += 1
        
        print(f"\n📊 全部类别 embedding 加载完成:")
        print(f"   总 embedding 数量: {len(self.embeddings_dict)}")
        print(f"   本次新加载: {loaded_count}")
        print(f"   跳过类别数: {skipped_count}")
    
    def _load_other_category_objects(self):
        """
        从分类 JSON 文件加载其他类别（非当前目标类别）的物体数据
        用于在最终排序中添加干扰项
        """
        print("\n" + "=" * 60)
        print(f"加载其他类别物体数据（排除当前目标类别: {self.target_category}）")
        print("=" * 60)
        
        if not os.path.exists(self.categorized_json_path):
            print(f"⚠️ 分类物体文件不存在: {self.categorized_json_path}")
            return
        
        try:
            with open(self.categorized_json_path, 'r', encoding='utf-8') as f:
                categorized_data = json.load(f)
            
            # 排除当前目标类别，加载其他所有类别
            excluded_categories = {self.target_category}  # 排除当前正在处理的类别
            
            for category, objects in categorized_data.items():
                if category in excluded_categories:
                    continue
                
                # 过滤有效的物体（图片路径存在）
                valid_objects = []
                for obj in objects:
                    image_path = obj.get('image_path', '')
                    if image_path and os.path.exists(image_path):
                        # 生成 object_id（从 mesh_path 提取）
                        mesh_path = obj.get('mesh_path', '')
                        obj_id = os.path.basename(mesh_path) if mesh_path else ''
                        obj['object_id'] = obj_id
                        valid_objects.append(obj)
                
                if valid_objects:
                    self.other_category_objects[category] = valid_objects
            
            total_other = sum(len(v) for v in self.other_category_objects.values())
            print(f"✅ 加载了 {len(self.other_category_objects)} 个其他类别，共 {total_other} 个物体")
            for cat, objs in self.other_category_objects.items():
                print(f"    - {cat}: {len(objs)} 个物体")
                
        except Exception as e:
            print(f"⚠️ 加载其他类别物体失败: {e}")
            self.other_category_objects = {}
    
    def _sample_other_category_objects(self, exclude_ids: set = None, count: int = OTHER_CATEGORY_OBJECTS, query_embedding: np.ndarray = None) -> Tuple[List[str], Dict[str, Dict]]:
        """
        从其他类别中采样物体。
        优先采样有 embedding 缓存的物体。如果提供了 query_embedding，
        则对有 embedding 的物体按余弦相似度排序（取最不相似的，增加多样性）。
        
        Args:
            exclude_ids: 需要排除的物体ID集合
            count: 采样数量
            query_embedding: 查询物体的 embedding（可选，用于相似度计算）
            
        Returns:
            (object_ids, object_info_dict)
        """
        if not self.other_category_objects:
            return [], {}
        
        exclude_ids = exclude_ids or set()
        
        # 收集所有其他类别的物体，区分有无 embedding
        objects_with_emb = []  # 有 embedding 缓存的物体
        objects_without_emb = []  # 无 embedding 缓存的物体
        
        for category, objects in self.other_category_objects.items():
            for obj in objects:
                obj_id = obj.get('object_id', '')
                if obj_id and obj_id not in exclude_ids:
                    if obj_id in self.embeddings_dict:
                        objects_with_emb.append((category, obj))
                    else:
                        objects_without_emb.append((category, obj))
        
        # 优先从有 embedding 的物体中采样
        sampled = []
        remaining_count = count
        
        if objects_with_emb:
            if query_embedding is not None and len(objects_with_emb) > remaining_count:
                # 有查询 embedding 时，按余弦相似度排序后随机采样（增加多样性）
                emb_ids = [obj.get('object_id', '') for _, obj in objects_with_emb]
                emb_matrix = np.array([self.embeddings_dict[oid] for oid in emb_ids])
                similarities = compute_cosine_similarity(query_embedding, emb_matrix)
                
                # 按相似度升序排列（取最不相似的，增加干扰难度多样性）
                sorted_indices = np.argsort(similarities)
                # 从最不相似的一半中随机采样
                pool_size = max(remaining_count, len(sorted_indices) // 2)
                pool_indices = sorted_indices[:pool_size]
                selected_indices = np.random.choice(pool_indices, size=min(remaining_count, len(pool_indices)), replace=False)
                sampled.extend([objects_with_emb[i] for i in selected_indices])
            else:
                # 无查询 embedding 时，随机采样
                sample_count = min(remaining_count, len(objects_with_emb))
                sampled.extend(random.sample(objects_with_emb, sample_count))
            
            remaining_count -= len(sampled)
        
        # 如果有 embedding 的不够，从无 embedding 的物体中补充
        if remaining_count > 0 and objects_without_emb:
            supplement_count = min(remaining_count, len(objects_without_emb))
            sampled.extend(random.sample(objects_without_emb, supplement_count))
        
        if not sampled:
            return [], {}
        
        # 整理结果
        object_ids = []
        object_info = {}
        
        for category, obj in sampled:
            obj_id = obj.get('object_id', '')
            has_embedding = obj_id in self.embeddings_dict
            object_ids.append(obj_id)
            object_info[obj_id] = {
                'object_id': obj_id,
                'image_path': obj.get('image_path', ''),
                'description': obj.get('description', ''),
                'category': category,  # 使用大类别名
                'mesh_path': obj.get('mesh_path', ''),
                'llm_category': obj.get('llm_category', category),
                'is_other_category': True,  # 标记为其他类别物体
                'has_embedding': has_embedding  # 标记是否有 embedding
            }
        
        return object_ids, object_info
    
    def extract_and_encode(self, force_recompute: bool = False):
        """提取点云并编码
        
        优化逻辑：
        1. 先检查 self.embeddings_dict 中已有的 embedding（可能由 _load_all_category_embeddings 预加载）
        2. 再检查当前类别的缓存文件，补充加载缺失的 embedding
        3. 最后只对仍然缺失的物体进行实际编码（需要初始化编码器）
        4. 只有在产生了新编码时才更新缓存文件
        """
        print("\n" + "=" * 60)
        print("提取点云并编码")
        print("=" * 60)
        
        current_ids = set(self.objects_dict.keys())
        
        # === 第一步：检查 embeddings_dict 中已有的 embedding ===
        already_loaded_ids = current_ids & set(self.embeddings_dict.keys())
        missing_ids = current_ids - already_loaded_ids
        
        if already_loaded_ids:
            print(f"📦 embeddings_dict 中已有 {len(already_loaded_ids)}/{len(current_ids)} 个物体的 embedding（来自全类别缓存预加载）")
        
        if not missing_ids and not force_recompute:
            print(f"✅ 所有 {len(current_ids)} 个物体的 embedding 均已加载，无需重新编码")
            return
        
        # === 第二步：从当前类别缓存文件补充加载 ===
        if not force_recompute and missing_ids and self.embedding_manager.has_cache():
            print(f"尝试从当前类别缓存补充加载 {len(missing_ids)} 个缺失的 embedding...")
            object_ids, embeddings, metadata = self.embedding_manager.load_embeddings()
            
            cached_map = {oid: embeddings[i] for i, oid in enumerate(object_ids)}
            newly_loaded = 0
            for oid in list(missing_ids):
                if oid in cached_map:
                    self.embeddings_dict[oid] = cached_map[oid]
                    missing_ids.discard(oid)
                    newly_loaded += 1
            
            if newly_loaded > 0:
                print(f"  从当前类别缓存补充加载了 {newly_loaded} 个 embedding")
        
        # === 第三步：检查是否还有缺失 ===
        if not missing_ids:
            print(f"✅ 所有 {len(current_ids)} 个物体的 embedding 均已加载，无需编码")
            return
        
        if force_recompute:
            missing_ids = current_ids  # 强制重新编码所有物体
            self.embeddings_dict = {k: v for k, v in self.embeddings_dict.items() if k not in current_ids}
            print(f"🔄 强制重新编码 {len(missing_ids)} 个物体")
        else:
            print(f"⚠️ 仍有 {len(missing_ids)} 个物体缺少 embedding，需要编码")
        
        # === 第四步：初始化编码器，编码缺失物体 ===
        if self.encoder is None:
            print("初始化 OpenShape 编码器...")
            self.encoder = OpenShapeEncoder()
        
        failed_objects = []
        newly_encoded = 0
        
        for obj_id in tqdm(missing_ids, desc="编码物体"):
            obj = self.objects_dict.get(obj_id)
            if obj is None:
                continue
            
            glb_path = obj.get_glb_path()
            
            # 提取点云
            pointcloud = extract_pointcloud_from_glb(glb_path, NUM_POINTS)
            
            if pointcloud is None:
                failed_objects.append(obj_id)
                continue
            
            # 编码
            try:
                embedding = self.encoder.encode(pointcloud)
                self.embeddings_dict[obj_id] = embedding
                newly_encoded += 1
            except Exception as e:
                print(f"⚠️ 编码失败 {obj_id}: {e}")
                failed_objects.append(obj_id)
        
        print(f"\n✅ 新编码 {newly_encoded} 个物体")
        print(f"📊 当前类别 embedding 总数: {len(current_ids & set(self.embeddings_dict.keys()))}/{len(current_ids)}")
        if failed_objects:
            print(f"⚠️ 失败 {len(failed_objects)} 个物体")
        
        # === 第五步：仅在有新编码时保存缓存 ===
        if newly_encoded > 0:
            # 只保存当前类别的 embedding 到当前类别的缓存文件
            category_object_ids = [oid for oid in self.objects_dict.keys() if oid in self.embeddings_dict]
            category_embeddings = np.array([self.embeddings_dict[oid] for oid in category_object_ids])
            
            self.embedding_manager.save_embeddings(
                object_ids=category_object_ids,
                embeddings=category_embeddings,
                metadata={
                    'source_file': self.input_json,
                    'failed_count': len(failed_objects),
                    'newly_encoded': newly_encoded
                }
            )
        else:
            print("ℹ️ 无新编码，跳过缓存保存")
    
    def perform_clustering(self, force_recompute: bool = False, target_num_clusters: int = None):
        """执行聚类"""
        # 检查缓存
        if not force_recompute and self.cluster_manager.has_cache():
            print("\n发现缓存的聚类结果，正在加载...")
            self.clusters, self.object_cluster_map = self.cluster_manager.load_clusters()
            return
        
        # 准备数据
        object_ids = list(self.embeddings_dict.keys())
        embeddings = np.array([self.embeddings_dict[oid] for oid in object_ids])
        
        # 执行聚类
        self.clusters, self.object_cluster_map = perform_clustering(
            embeddings=embeddings,
            object_ids=object_ids,
            min_cluster_size=MIN_CLUSTER_SIZE,
            target_num_clusters=target_num_clusters
        )
        
        # 更新物体的 cluster_id
        for obj_id, cluster_id in self.object_cluster_map.items():
            if obj_id in self.objects_dict:
                self.objects_dict[obj_id].cluster_id = cluster_id
        
        # 保存聚类结果
        self.cluster_manager.save_clusters(
            clusters=self.clusters,
            object_cluster_map=self.object_cluster_map,
            metadata={'source_file': self.input_json}
        )
    
    def generate_cases(self, num_cases: int = 10, random_seed: int = 42) -> Tuple[List[RankingCase], List[Dict]]:
        """
        生成精排 cases
        
        新流程：
        1. 从同 cluster 选取相似度最高的 10 个物体
        2. 从相邻 cluster 随机选取 10 个物体
        3. 这 20 个物体 + query 送入 agent 精排
        4. 从较远 cluster 随机选取 20 个物体（不参与精排）
        5. 从其他类别（非当前目标类别）随机选取 9 个物体（不参与精排）
        6. 最终结构: [query] + [精排20个] + [较远cluster20个] + [其他类别9个] = 50个物体
        
        Args:
            num_cases: 生成的 case 数量
            random_seed: 随机种子
            
        Returns:
            (成功的 RankingCase 列表, 失败的 case 信息列表)
        """
        print("\n" + "=" * 60)
        print(f"生成 {num_cases} 个精排 cases")
        print("=" * 60)
        
        random.seed(random_seed)
        
        # 初始化 agent 精排执行器
        if self.use_agent_ranking:
            try:
                from group_ranking_skill import GroupRankingExecutor
                self.executor = GroupRankingExecutor(
                    llm_mode=self.llm_mode,
                    model_name=self.model_name,
                    verbose=False
                )
            except Exception as e:
                print(f"⚠️ 无法初始化 agent 精排: {e}")
                self.executor = None
        
        # 过滤有效的 clusters（大小 >= 30）
        valid_clusters = [c for c in self.clusters.values() if c.size >= MIN_CLUSTER_SIZE]
        print(f"有效聚类数（>= {MIN_CLUSTER_SIZE}）: {len(valid_clusters)}")
        
        if not valid_clusters:
            print("❌ 没有足够大的聚类")
            return []
        
        # 随机选择 clusters 和 query
        cases = []  # 成功的 cases
        failed_cases = []  # 失败的 cases
        
        for i in tqdm(range(num_cases), desc="生成 cases"):
            # 随机选择一个 cluster
            cluster = random.choice(valid_clusters)
            
            # 随机选择一个 query
            query_id = random.choice(cluster.object_ids)
            query_obj = self.objects_dict.get(query_id)
            
            if query_obj is None:
                continue
            
            # 生成 case（不包含其他类别物体）
            try:
                case = generate_ranking_case(
                    query_obj=query_obj,
                    cluster=cluster,
                    clusters=self.clusters,
                    objects_dict=self.objects_dict,
                    embeddings_dict=self.embeddings_dict,
                    executor=self.executor,
                    use_agent_ranking=self.use_agent_ranking and self.executor is not None
                )
                
                # 添加其他类别物体（不参与精排）
                existing_ids = set(case.final_ranking)
                # 获取 query embedding 用于其他类别物体的相似度采样
                query_emb = self.embeddings_dict.get(query_id)
                other_category_ids, other_category_info = self._sample_other_category_objects(
                    exclude_ids=existing_ids,
                    count=OTHER_CATEGORY_OBJECTS,
                    query_embedding=query_emb
                )
                
                # 更新 case
                case.other_category_objects = other_category_ids
                case.other_category_info = other_category_info
                
                # 更新最终排序：[query] + [精排20个] + [较远cluster20个] + [其他类别9个] = 50个物体
                case.final_ranking = case.final_ranking + other_category_ids
                
                # 合并物体信息
                case.objects_info.update(other_category_info)
                
                cases.append(case)
                
                # 保存中间结果到缓存目录
                self._save_intermediate_case(case, i)
                
            except Exception as e:
                # 记录失败的 case 信息
                import traceback
                error_traceback = traceback.format_exc()
                failed_case_info = {
                    "case_index": i,
                    "query_object_id": query_id,
                    "cluster_id": cluster.cluster_id,
                    "category": query_obj.category,
                    "error_message": str(e),
                    "error_traceback": error_traceback,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                failed_cases.append(failed_case_info)
                print(f"⚠️ 生成 case {i} 失败: {e}")
                traceback.print_exc()
                continue
        
        print(f"\n✅ 成功生成 {len(cases)} 个 cases")
        if failed_cases:
            print(f"❌ 失败 {len(failed_cases)} 个 cases")
        print(f"每个 case 包含 50 个物体: 1(query) + 20(精排) + 20(较远cluster) + 9(其他类别)")
        return cases, failed_cases
    
    def _save_intermediate_case(self, case: RankingCase, index: int):
        """
        保存单个 case 的中间结果到缓存目录
        
        Args:
            case: RankingCase 对象
            index: case 索引
        """
        intermediate_file = os.path.join(
            self.intermediate_cache_dir, 
            f"intermediate_case_{index:04d}_{case.query_object_id}.json"
        )
        
        case_data = {
            "case_id": case.case_id,
            "query_object_id": case.query_object_id,
            "cluster_id": case.cluster_id,
            "category": case.category,
            "final_ranking": case.final_ranking,
            "ranking_details": {
                "reranked_objects": case.reranked_objects,
                "top_similar_objects": case.top_similar_objects,
                "neighbor_random_objects": case.neighbor_random_objects,
                "distant_ranking": case.distant_ranking,
                "other_category_objects": case.other_category_objects
            },
            "weighted_scores": case.weighted_scores,
            "distant_similarity_scores": case.distant_similarity_scores,
            "other_category_info": case.other_category_info,
            "objects": case.objects_info,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(intermediate_file, 'w', encoding='utf-8') as f:
            json.dump(case_data, f, ensure_ascii=False, indent=2)
        
        # 不打印每个中间文件的保存信息，避免输出过多
    
    def save_cases(self, cases: List[RankingCase], failed_cases: List[Dict] = None, output_filename: str = None):
        """
        保存 cases 到专门的输出目录
        
        Args:
            cases: RankingCase 列表（成功的）
            failed_cases: 失败的 case 信息列表
            output_filename: 输出文件名
        """
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"ranking_cases_50objects_{timestamp}.json"
        
        output_path = os.path.join(self.output_dir, output_filename)
        
        failed_cases = failed_cases or []
        
        # 转换为兼容原有格式的 JSON
        output_data = {
            "metadata": {
                "source_file": self.input_json,
                "categorized_json": self.categorized_json_path,
                "total_cases": len(cases),
                "failed_cases_count": len(failed_cases),
                "generated_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "use_agent_ranking": self.use_agent_ranking,
                "output_dir": self.output_dir,
                "intermediate_cache_dir": self.intermediate_cache_dir,
                "ranking_structure": {
                    "rerank_size": RANKING_BATCH_SIZE,  # 精排物体总数（20）
                    "top_similar_count": TOP_SIMILAR_COUNT,  # 同cluster相似度最高的物体数（10）
                    "neighbor_random_count": NEIGHBOR_RANDOM_COUNT,  # 相邻cluster随机选取的物体数（10）
                    "distant_size": DISTANT_CLUSTER_OBJECTS,  # 较远cluster物体数（20，不参与精排）
                    "other_category_size": OTHER_CATEGORY_OBJECTS,  # 其他类别物体数（9，不参与精排）
                    "total_per_case": 1 + RANKING_BATCH_SIZE + DISTANT_CLUSTER_OBJECTS + OTHER_CATEGORY_OBJECTS  # 总数（50）
                }
            },
            "cluster_statistics": {
                "total_clusters": len(self.clusters),
                "cluster_sizes": {str(k): v.size for k, v in self.clusters.items()}
            },
            "other_category_statistics": {
                "total_categories": len(self.other_category_objects),
                "category_sizes": {cat: len(objs) for cat, objs in self.other_category_objects.items()}
            },
            "cases": [],
            "failed_cases": failed_cases  # 添加失败的 cases 信息
        }
        
        for case in cases:
            case_data = {
                "case_id": case.case_id,
                "query_object_id": case.query_object_id,
                "cluster_id": case.cluster_id,
                "category": case.category,
                "total_objects": len(case.final_ranking),
                "final_ranking": case.final_ranking,
                "ranking_details": {
                    "reranked_objects": case.reranked_objects,  # 精排后的20个物体
                    "top_similar_objects": case.top_similar_objects,  # 同cluster相似度最高的10个
                    "neighbor_random_objects": case.neighbor_random_objects,  # 相邻cluster随机选取的10个
                    "distant_ranking": case.distant_ranking,  # 较远cluster的20个物体
                    "other_category_objects": case.other_category_objects,  # 其他类别的9个物体
                    "pre_rerank_cosine_order": case.pre_rerank_cosine_order,  # 精排前的余弦相似度排序
                },
                # 添加加权分数信息（仅对精排的20个物体有效）
                "weighted_scores": case.weighted_scores,  # Agent 精排的加权分数
                "pre_rerank_cosine_scores": case.pre_rerank_cosine_scores,  # 精排前的余弦相似度分数
                "distant_similarity_scores": case.distant_similarity_scores,  # 较远物体的余弦相似度
                "other_category_info": case.other_category_info,  # 其他类别物体的详细信息
                "objects": case.objects_info
            }
            output_data["cases"].append(case_data)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ Cases 已保存到: {output_path}")
        print(f"   - 成功: {len(cases)} 个 cases")
        if failed_cases:
            print(f"   - 失败: {len(failed_cases)} 个 cases（详情见 failed_cases 字段）")
        print(f"📁 最终输出目录: {self.output_dir}")
        print(f"📁 中间结果缓存目录: {self.intermediate_cache_dir}")
        return output_path
    
    def resume_and_rerun(self, resume_file: str) -> str:
        """
        断点续跑模式：读取上次输出文件，找出 agent 精排失败的 case（weighted_scores 为空），
        仅对这些 case 重新运行 agent 精排，成功的 case 保持不变。
        
        Args:
            resume_file: 上次输出的 JSON 文件路径
            
        Returns:
            新输出文件路径
        """
        print("\n" + "=" * 70)
        print("🔄 Resume 模式：断点续跑")
        print("=" * 70)
        print(f"读取上次结果: {resume_file}")
        
        with open(resume_file, 'r', encoding='utf-8') as f:
            prev_data = json.load(f)
        
        prev_cases = prev_data.get('cases', [])
        prev_failed = prev_data.get('failed_cases', [])
        total_cases = len(prev_cases)
        
        # 找出需要重新精排的 case（weighted_scores 为空 dict）
        failed_indices = []
        success_cases_data = []
        for idx, case_data in enumerate(prev_cases):
            ws = case_data.get('weighted_scores', {})
            if not ws:  # weighted_scores 为空
                failed_indices.append(idx)
            else:
                success_cases_data.append(case_data)
        
        print(f"\n📊 上次结果统计:")
        print(f"  - 总 case 数: {total_cases}")
        print(f"  - Agent 精排成功: {len(success_cases_data)}")
        print(f"  - Agent 精排失败（需重跑）: {len(failed_indices)}")
        if prev_failed:
            print(f"  - 上次完全失败的 case: {len(prev_failed)}")
        
        if not failed_indices:
            print("\n✅ 所有 case 的 agent 精排均已成功，无需重跑！")
            return resume_file
        
        # 加载数据（需要 objects_dict 来构建 ObjectInfo）
        self.load_input_data()
        
        # 初始化 agent 精排执行器
        if self.use_agent_ranking:
            try:
                from group_ranking_skill import GroupRankingExecutor
                self.executor = GroupRankingExecutor(
                    llm_mode=self.llm_mode,
                    model_name=self.model_name,
                    verbose=False
                )
                print(f"\n✅ Agent 精排执行器已初始化 (模式: {self.llm_mode})")
            except Exception as e:
                print(f"❌ 无法初始化 agent 精排: {e}")
                return resume_file
        else:
            print("❌ Resume 模式需要开启 agent 精排（不要使用 --no_agent）")
            return resume_file
        
        # 也加载其他类别物体数据（用于构建 ObjectInfo）
        if os.path.exists(self.categorized_json_path):
            with open(self.categorized_json_path, 'r', encoding='utf-8') as f:
                categorized_data = json.load(f)
            for cat, cat_objects in categorized_data.items():
                for obj_data in cat_objects:
                    oid = obj_data.get('object_id', '')
                    if oid and oid not in self.objects_dict:
                        self.objects_dict[oid] = ObjectInfo(
                            object_id=oid,
                            mesh_path=obj_data.get('mesh_path', ''),
                            image_path=obj_data.get('image_path', ''),
                            description=obj_data.get('description', ''),
                            category=obj_data.get('category', cat),
                            llm_category=obj_data.get('llm_category', '')
                        )
        
        # 对失败的 case 逐一重新运行 agent 精排
        rerun_success = 0
        rerun_failed = 0
        
        for fail_idx in tqdm(failed_indices, desc="重新精排失败的 cases"):
            case_data = prev_cases[fail_idx]
            query_id = case_data['query_object_id']
            query_obj = self.objects_dict.get(query_id)
            
            if query_obj is None:
                print(f"  ⚠️ 找不到 query 物体 {query_id}，跳过")
                rerun_failed += 1
                continue
            
            # 从上次结果中获取精排候选列表（精排的20个物体）
            ranking_details = case_data.get('ranking_details', {})
            reranked_objects = ranking_details.get('reranked_objects', [])
            
            if not reranked_objects:
                # fallback: 使用 pre_rerank_cosine_order
                reranked_objects = case_data.get('pre_rerank_cosine_scores', {}).keys()
                reranked_objects = list(reranked_objects)
            
            if not reranked_objects:
                print(f"  ⚠️ Case {fail_idx} 没有精排候选列表，跳过")
                rerun_failed += 1
                continue
            
            # 重新运行 agent 精排
            try:
                new_reranked, new_weighted_scores = run_agent_ranking(
                    query_obj, reranked_objects, self.objects_dict, self.executor
                )
                
                if new_weighted_scores:  # 精排成功（有分数）
                    # 更新 case 数据
                    prev_cases[fail_idx]['weighted_scores'] = new_weighted_scores
                    prev_cases[fail_idx]['ranking_details']['reranked_objects'] = new_reranked
                    
                    # 更新 final_ranking: [query] + [新精排20个] + [较远cluster20个] + [其他类别9个]
                    distant_ranking = ranking_details.get('distant_ranking', [])
                    other_category_objects = ranking_details.get('other_category_objects', [])
                    prev_cases[fail_idx]['final_ranking'] = [query_id] + new_reranked + distant_ranking + other_category_objects
                    
                    rerun_success += 1
                    
                    # 更新中间结果缓存
                    intermediate_file = os.path.join(
                        self.intermediate_cache_dir,
                        f"intermediate_case_{fail_idx:04d}_{query_id}.json"
                    )
                    with open(intermediate_file, 'w', encoding='utf-8') as f:
                        json.dump(prev_cases[fail_idx], f, ensure_ascii=False, indent=2)
                else:
                    print(f"  ⚠️ Case {fail_idx} (query={query_id}) agent 精排仍然失败（无分数）")
                    rerun_failed += 1
                    
            except Exception as e:
                print(f"  ⚠️ Case {fail_idx} (query={query_id}) 重新精排异常: {e}")
                rerun_failed += 1
        
        # 统计最终结果
        final_success = sum(1 for c in prev_cases if c.get('weighted_scores', {}))
        final_empty = sum(1 for c in prev_cases if not c.get('weighted_scores', {}))
        
        print(f"\n" + "=" * 60)
        print(f"📊 Resume 结果统计:")
        print(f"  - 本次重跑成功: {rerun_success}")
        print(f"  - 本次重跑失败: {rerun_failed}")
        print(f"  - 最终精排成功: {final_success}/{total_cases}")
        print(f"  - 最终精排失败: {final_empty}/{total_cases}")
        print("=" * 60)
        
        # 更新 metadata
        prev_data['metadata']['generated_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        prev_data['metadata']['resume_info'] = {
            'resume_from': resume_file,
            'rerun_success': rerun_success,
            'rerun_failed': rerun_failed,
            'final_success': final_success,
            'final_empty': final_empty
        }
        prev_data['metadata']['total_cases'] = total_cases
        prev_data['metadata']['failed_cases_count'] = len(prev_failed)
        
        # 保存到新文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"ranking_cases_50objects_{timestamp}.json"
        output_path = os.path.join(self.output_dir, output_filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(prev_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ Resume 结果已保存到: {output_path}")
        return output_path

    def run(
        self, 
        num_cases: int = 10,
        force_recompute_embeddings: bool = False,
        force_recompute_clusters: bool = False,
        target_num_clusters: int = None,
        random_seed: int = 42
    ) -> str:
        """
        运行完整流程
        
        Args:
            num_cases: 生成的 case 数量
            force_recompute_embeddings: 强制重新计算 embeddings
            force_recompute_clusters: 强制重新聚类
            target_num_clusters: 目标聚类数
            random_seed: 随机种子
            
        Returns:
            输出文件路径
        """
        print("\n" + "=" * 70)
        print("🚀 OpenShape 聚类精排 Pipeline (50个物体/case)")
        print("=" * 70)
        print(f"输入文件: {self.input_json}")
        print(f"目标类别: {self.target_category}")
        print(f"其他类别数据: {self.categorized_json_path}")
        print(f"最终输出目录: {self.output_dir}")
        print(f"中间结果缓存目录: {self.intermediate_cache_dir}")
        print(f"缓存目录: {self.cache_dir}")
        print(f"缓存基础目录（全类别）: {self.cache_base_dir}")
        print(f"生成 case 数量: {num_cases}")
        print(f"使用 Agent 精排: {self.use_agent_ranking}")
        print(f"\n排序结构:")
        print(f"  - Query: 1")
        print(f"  - 精排物体: {RANKING_BATCH_SIZE} (同cluster最相似{TOP_SIMILAR_COUNT} + 相邻cluster随机{NEIGHBOR_RANDOM_COUNT})")
        print(f"  - 较远cluster: {DISTANT_CLUSTER_OBJECTS}")
        print(f"  - 其他类别: {OTHER_CATEGORY_OBJECTS}")
        print(f"  - 总计: {1 + RANKING_BATCH_SIZE + DISTANT_CLUSTER_OBJECTS + OTHER_CATEGORY_OBJECTS}")
        print("=" * 70)
        
        # 步骤 1: 加载数据
        self.load_input_data()
        
        # 步骤 2: 提取点云并编码
        self.extract_and_encode(force_recompute=force_recompute_embeddings)
        
        # 步骤 3: 聚类
        self.perform_clustering(
            force_recompute=force_recompute_clusters,
            target_num_clusters=target_num_clusters
        )
        
        # 步骤 4: 生成 cases
        cases, failed_cases = self.generate_cases(num_cases=num_cases, random_seed=random_seed)
        
        # 步骤 5: 保存结果（包括失败的 cases）
        output_path = self.save_cases(cases, failed_cases)
        
        print("\n" + "=" * 70)
        print("✅ Pipeline 完成！")
        print("=" * 70)
        
        return output_path


# ==================== 命令行接口 ====================

def main():
    parser = argparse.ArgumentParser(description='OpenShape 聚类精排 Pipeline (50个物体/case)')
    
    parser.add_argument('--input', type=str, default='/apdcephfs/share_303565425/DCC3/serenasnliu/GAR/Uni3D/data/test_datasets/objaverse/objaverse_golden_all_groups.json',
                        help='输入 JSON 文件（分组数据）')
    parser.add_argument('--cache_dir', type=str, default=None,
                        help='缓存目录（直接指定，优先级最高；不指定则根据 --category 自动推断）')
    parser.add_argument('--cache_base_dir', type=str, default=None,
                        help='缓存基础目录（根据 --category 自动拼接子目录，如 base/openshape_cache_object/）')
    parser.add_argument('--output_dir', type=str, default=None,
                        help=f'最终输出目录（默认: {FINAL_OUTPUT_DIR}）')
    parser.add_argument('--intermediate_cache_dir', type=str, default=None,
                        help=f'中间结果缓存目录（默认: {INTERMEDIATE_CACHE_DIR}）')
    parser.add_argument('--categorized_json', type=str, default=None,
                        help=f'分类物体 JSON 文件路径（默认: {CATEGORIZED_JSON_PATH}）')
    parser.add_argument('--category', type=str, default='Character',
                        help='当前处理的目标类别名称（如 Character, Object, Building, Weapon, Vehicle, Animal）')
    parser.add_argument('--num_cases', type=int, default=10,
                        help='生成的 case 数量')
    parser.add_argument('--num_clusters', type=int, default=None,
                        help='目标聚类数（默认自动计算）')
    parser.add_argument('--no_agent', action='store_true',
                        help='不使用 agent 精排，仅使用余弦相似度')
    parser.add_argument('--llm_mode', type=str, default='qwen',
                        choices=['api', 'qwen', 'mock'],
                        help='LLM 模式')
    parser.add_argument('--force_embeddings', action='store_true',
                        help='强制重新计算 embeddings')
    parser.add_argument('--force_clusters', action='store_true',
                        help='强制重新聚类')
    parser.add_argument('--seed', type=int, default=20260301,
                        help='随机种子')
    parser.add_argument('--resume', type=str, default=None,
                        help='断点续跑：指定上次输出的 JSON 文件路径，仅对 agent 精排失败的 case 重新运行')
    
    args = parser.parse_args()
    
    # 创建并运行 pipeline
    pipeline = OpenShapeClusteringPipeline(
        input_json=args.input,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        cache_base_dir=args.cache_base_dir,
        intermediate_cache_dir=args.intermediate_cache_dir,
        use_agent_ranking=not args.no_agent,
        llm_mode=args.llm_mode,
        categorized_json_path=args.categorized_json,
        target_category=args.category
    )
    
    if args.resume:
        # Resume 模式：断点续跑
        pipeline.resume_and_rerun(args.resume)
    else:
        # 正常模式：完整运行
        pipeline.run(
            num_cases=args.num_cases,
            force_recompute_embeddings=args.force_embeddings,
            force_recompute_clusters=args.force_clusters,
            target_num_clusters=args.num_clusters,
            random_seed=args.seed
        )


if __name__ == '__main__':
    main()
