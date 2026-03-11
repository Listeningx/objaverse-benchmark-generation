"""
使用真实 Gemini API 测试基于图片的 Agent-Skills 排序流程

运行前请确保：
1. 已安装 google-genai 库：pip install google-genai
2. 已设置 API Key（通过环境变量 GEMINI_API_KEY 或在代码中设置）
3. 准备好查询图片和候选物品图片
"""

import os
import sys
import json
import glob
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# 确保模块路径正确
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_interface import get_llm_interface, LLMInterface, get_persistent_cache
from pipeline import RankingPipeline, PipelineConfig, create_pipeline
from dimension_bank import get_dimension_bank
from data_loader import DataLoader, LoadedData, print_loaded_data_summary
from group_ranking_skill import (
    GroupDataLoader, GroupRankingExecutor, 
    run_group_ranking, GroupedData
)


# ==================== 配置区域 ====================
# 设置 API Key（如果已通过环境变量设置，可以注释掉下面这行）
# os.environ['GEMINI_API_KEY'] = 'your-api-key-here'
# os.environ['QWEN_API_KEY'] = 'your-qwen-api-key-here'

# 模型配置
MODEL_NAME = "gemini-3-flash-preview"  # Gemini 模式默认模型
QWEN_MODEL_NAME = "qwen3-vl-235b-a22b-thinking"  # QWEN 模式默认模型
DEFAULT_LLM_MODE = "api"  # 默认 LLM 模式: "api" (Gemini), "qwen", "mock"

# 应用场景配置
DEFAULT_APPLICATION_SCENARIO = "游戏场景中的相似资产检索"  # 默认应用场景

# 预设应用场景列表（可扩展）
PRESET_SCENARIOS = {
    "1": {
        "name": "游戏场景中的相似资产检索",
        "description": "用于游戏开发中查找风格、材质、功能相似的3D资产，重点关注视觉风格一致性和游戏适用性"
    },
    "2": {
        "name": "电商商品相似推荐",
        "description": "用于电商平台推荐相似商品，重点关注功能、价格区间、品牌定位的相似性"
    },
    "3": {
        "name": "工业设计参考检索",
        "description": "用于工业设计师查找参考设计，重点关注造型、结构、工艺的相似性"
    },
    "4": {
        "name": "3D模型库管理",
        "description": "用于3D模型库的去重和分类，重点关注几何形状、拓扑结构的相似性"
    },
    "5": {
        "name": "自定义场景",
        "description": "用户自定义应用场景"
    }
}


# ==================== 工具函数 ====================

def find_images_in_directory(directory: str, extensions: list = None) -> list:
    """
    在目录中查找图片文件
    
    Args:
        directory: 目录路径
        extensions: 图片扩展名列表，默认为常见图片格式
        
    Returns:
        list: 图片文件路径列表
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']
    
    image_files = []
    for ext in extensions:
        pattern = os.path.join(directory, f'*{ext}')
        image_files.extend(glob.glob(pattern))
        pattern = os.path.join(directory, f'*{ext.upper()}')
        image_files.extend(glob.glob(pattern))
    
    return sorted(list(set(image_files)))


def load_images_from_txt(txt_file: str) -> list:
    """
    从 txt 文件加载图片路径列表
    
    Args:
        txt_file: txt 文件路径，每行包含一个图片路径
        
    Returns:
        list: 图片文件路径列表
    """
    if not os.path.exists(txt_file):
        raise FileNotFoundError(f"图片列表文件不存在：{txt_file}")
    
    valid_paths = []
    with open(txt_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            img_path = line.strip()
            if not img_path:
                continue  # 跳过空行
            
            if not os.path.exists(img_path):
                print(f"⚠️  第 {line_num} 行路径无效，跳过: {img_path}")
                continue
            
            valid_paths.append(img_path)
    
    print(f"✅ 成功加载 {len(valid_paths)} 个有效图片路径")
    return valid_paths


# ==================== 测试函数 ====================

def test_api_connection():
    """
    测试 API 连接是否正常
    """
    print("\n" + "=" * 60)
    print("步骤 1: 测试 API 连接")
    print("=" * 60)
    
    try:
        # 强制创建新的 API 模式实例
        llm = get_llm_interface(
            mode="api",
            model=MODEL_NAME,
            force_new=True
        )
        
        # 发送简单测试请求
        test_prompt = "请用一句话回答：1+1等于多少？只输出答案。"
        response = llm.call_llm(test_prompt, expected_format="text")
        
        print(f"\n✅ API 连接成功！")
        print(f"测试响应: {response[:100]}...")
        return True
        
    except Exception as e:
        print(f"\n❌ API 连接失败: {e}")
        print("\n请检查:")
        print("  1. API Key 是否正确设置")
        print("  2. 网络连接是否正常")
        print("  3. google-genai 库是否已安装")
        return False


def test_qwen_connection():
    """
    测试 QWEN API 连接是否正常
    """
    print("\n" + "=" * 60)
    print("步骤 1: 测试 QWEN API 连接")
    print("=" * 60)
    
    try:
        # 强制创建新的 QWEN 模式实例
        llm = get_llm_interface(
            mode="qwen",
            model=QWEN_MODEL_NAME,
            force_new=True
        )
        
        # 发送简单测试请求
        test_prompt = "请用一句话回答：1+1等于多少？只输出答案。"
        response = llm.call_llm(test_prompt, expected_format="text")
        
        print(f"\n✅ QWEN API 连接成功！")
        print(f"测试响应: {response[:100]}...")
        return True
        
    except Exception as e:
        print(f"\n❌ QWEN API 连接失败: {e}")
        print("\n请检查:")
        print("  1. QWEN_API_KEY 是否正确设置")
        print("  2. 网络连接是否正常")
        print("  3. openai 库是否已安装")
        return False


def select_llm_mode() -> str:
    """
    交互式选择 LLM 模式
    
    Returns:
        str: LLM 模式 ("api", "qwen", "mock")
    """
    print("\n请选择 LLM 模式:")
    print("  1. Gemini (默认)")
    print("  2. QWEN (通义千问)")
    print("  3. Mock (模拟模式，无需 API)")
    print("请输入选项 (1-3, 默认 1): ", end="")
    
    llm_choice = input().strip()
    
    if llm_choice == "2":
        return "qwen"
    elif llm_choice == "3":
        return "mock"
    else:
        return "api"


def select_application_scenario() -> str:
    """
    交互式选择应用场景
    
    Returns:
        str: 应用场景描述
    """
    print("\n🎯 请选择应用场景:")
    for key, scenario in PRESET_SCENARIOS.items():
        print(f"  {key}. {scenario['name']}")
        print(f"     {scenario['description']}")
    print(f"\n当前默认场景: {DEFAULT_APPLICATION_SCENARIO}")
    print("请输入选项 (1-5, 默认使用当前默认场景，直接回车跳过): ", end="")
    
    choice = input().strip()
    
    if not choice:
        return DEFAULT_APPLICATION_SCENARIO
    
    if choice in PRESET_SCENARIOS:
        if choice == "5":
            # 自定义场景
            print("\n请输入自定义应用场景描述: ", end="")
            custom_scenario = input().strip()
            if custom_scenario:
                return custom_scenario
            else:
                return DEFAULT_APPLICATION_SCENARIO
        else:
            return PRESET_SCENARIOS[choice]["name"]
    else:
        print(f"⚠️  无效选项，使用默认场景: {DEFAULT_APPLICATION_SCENARIO}")
        return DEFAULT_APPLICATION_SCENARIO


def run_image_ranking_test(
    query_image: str,
    candidate_images: dict,
    query_description: str = "",
    use_api: bool = True,
    source_json_path: str = None,
    llm_mode: str = None,
    application_scenario: str = None
):
    """
    运行基于图片的排序测试
    
    Args:
        query_image: 查询图片路径
        candidate_images: 候选物品图片字典 {candidate_id: image_path}
        query_description: 查询图片的补充描述（可选）
        use_api: 是否使用真实 API（False 则使用 mock 模式）
        source_json_path: 源 JSON 文件路径（用于持久化缓存）
        llm_mode: LLM 模式（"api"/"qwen"/"mock"），优先级高于 use_api 参数
        application_scenario: 应用场景描述，用于指导维度规划
        
    Returns:
        排序结果
    """
    print("\n" + "=" * 60)
    print("运行基于图片的排序测试")
    print("=" * 60)
    
    # 设置应用场景（如果未指定则使用默认场景）
    if application_scenario is None:
        application_scenario = DEFAULT_APPLICATION_SCENARIO
    
    print(f"\n🎯 应用场景: {application_scenario}")
    
    # 确定 LLM 模式
    if llm_mode:
        mode = llm_mode
    else:
        mode = "api" if use_api else "mock"
    
    # 根据模式选择模型
    if mode == "qwen":
        model_name = QWEN_MODEL_NAME
    else:
        model_name = MODEL_NAME
    
    # 初始化 LLM 接口
    llm = get_llm_interface(
        mode=mode,
        model=model_name,
        force_new=True
    )
    print(f"\n✅ LLM 模式: {mode}, 模型: {model_name}")
    
    # 显示输入信息
    print(f"\n📷 查询图片: {query_image}")
    if query_description:
        print(f"📝 补充描述: {query_description}")
    print(f"📦 候选物品数量: {len(candidate_images)}")
    for cid, img_path in candidate_images.items():
        print(f"   - {cid}: {os.path.basename(img_path)}")
    
    # 组合应用场景和补充描述作为完整的 query_description
    full_description = f"【应用场景】{application_scenario}"
    if query_description:
        full_description += f"\n【补充信息】{query_description}"
    
    # 创建并运行流水线
    print("\n🚀 开始运行排序流水线...")
    pipeline = RankingPipeline(verbose=True, source_json_path=source_json_path)
    
    try:
        result = pipeline.run_with_images(
            query_image=query_image,
            candidate_images=candidate_images,
            query_description=full_description
        )
        
        # 将应用场景信息添加到结果中
        result["application_scenario"] = application_scenario
        
        # 打印结果
        print("\n" + "-" * 50)
        print("✅ 排序完成！最终结果:")
        print("-" * 50)
        for rank, cid in enumerate(result["final_ranking"], 1):
            img_path = candidate_images.get(cid, "未知")
            print(f"  {rank}. {cid}: {os.path.basename(img_path)}")
        
        # 生成解释报告
        print("\n" + "-" * 50)
        print("📄 详细解释报告:")
        print("-" * 50)
        report = pipeline.generate_explanation_report(result)
        print(report)
        
        # 保存结果（使用时间戳唯一命名）
        output_dir = os.path.dirname(os.path.abspath(__file__))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_path = os.path.join(output_dir, f"image_ranking_result_{timestamp}.json")
        pipeline.export_result(result, result_path)
        print(f"\n💾 结果已保存到: {result_path}")
        
        # 显示 Token 统计
        show_llm_statistics()
        
        return result
        
    except Exception as e:
        print(f"\n❌ 排序流程出错: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_batch_image_ranking(
    query_image: str,
    candidate_images_dir: str,
    query_description: str = "",
    max_candidates: int = 10,
    use_api: bool = True
):
    """
    批量运行图片排序（从目录加载候选图片）
    
    Args:
        query_image: 查询图片路径
        candidate_images_dir: 候选物品图片目录
        query_description: 查询图片的补充描述（可选）
        max_candidates: 最大候选数量
        use_api: 是否使用真实 API
        
    Returns:
        排序结果
    """
    print("\n" + "=" * 60)
    print("批量图片排序测试")
    print("=" * 60)
    
    # 查找候选图片
    candidate_image_files = find_images_in_directory(candidate_images_dir)
    
    if not candidate_image_files:
        print(f"❌ 在目录 {candidate_images_dir} 中未找到图片文件")
        return None
    
    print(f"✅ 在目录中找到 {len(candidate_image_files)} 个图片文件")
    
    # 限制候选数量
    if len(candidate_image_files) > max_candidates:
        print(f"⚠️  图片数量超过限制，只取前 {max_candidates} 个")
        candidate_image_files = candidate_image_files[:max_candidates]
    
    # 构建候选图片字典
    candidate_images = {}
    for i, img_path in enumerate(candidate_image_files, 1):
        candidate_id = f"candidate_{i:03d}"
        candidate_images[candidate_id] = img_path
    
    # 运行排序
    return run_image_ranking_test(
        query_image=query_image,
        candidate_images=candidate_images,
        query_description=query_description,
        use_api=use_api
    )


def run_from_txt_file(
    txt_file: str,
    query_description: str = "",
    use_api: bool = True
):
    """
    从 txt 文件加载图片路径并运行排序
    
    txt 文件格式：每行一个图片路径，第一行为查询图片，其余为候选图片
    
    Args:
        txt_file: txt 文件路径
        query_description: 查询图片的补充描述（可选）
        use_api: 是否使用真实 API
        
    Returns:
        排序结果
    """
    print("\n" + "=" * 60)
    print("从 TXT 文件加载图片并运行排序")
    print("=" * 60)
    
    # 加载图片路径
    image_paths = load_images_from_txt(txt_file)
    
    if len(image_paths) < 2:
        print("❌ 图片数量不足（至少需要 1 个查询图片和 1 个候选图片）")
        return None
    
    # 第一个为查询图片，其余为候选图片
    query_image = image_paths[0]
    candidate_image_files = image_paths[1:]
    
    print(f"📷 查询图片: {query_image}")
    print(f"📦 候选图片数量: {len(candidate_image_files)}")
    
    # 构建候选图片字典
    candidate_images = {}
    for i, img_path in enumerate(candidate_image_files, 1):
        candidate_id = f"candidate_{i:03d}"
        candidate_images[candidate_id] = img_path
    
    # 运行排序
    return run_image_ranking_test(
        query_image=query_image,
        candidate_images=candidate_images,
        query_description=query_description,
        use_api=use_api
    )


def show_llm_statistics():
    """
    显示 LLM 调用统计（包含 Token 统计）
    """
    llm = get_llm_interface()
    stats = llm.get_call_statistics()
    
    print("\n" + "=" * 60)
    print("📊 LLM 调用统计")
    print("=" * 60)
    print(f"  - 总调用次数: {stats['total_calls']}")
    print(f"  - 运行模式: {stats['mode']}")
    print(f"  - 历史记录数: {stats['history_count']}")
    
    # 显示 Token 统计
    print("\n📝 Token 使用统计:")
    print(f"  - 总输入 Token: {stats.get('total_input_tokens', 0):,}")
    print(f"  - 总输出 Token: {stats.get('total_output_tokens', 0):,}")
    print(f"  - 总 Token: {stats.get('total_tokens', 0):,}")
    
    # 计算平均值
    token_history = stats.get('token_history', [])
    if token_history:
        avg_input = stats.get('total_input_tokens', 0) / len(token_history)
        avg_output = stats.get('total_output_tokens', 0) / len(token_history)
        print(f"  - 平均输入 Token/次: {avg_input:,.0f}")
        print(f"  - 平均输出 Token/次: {avg_output:,.0f}")


# ==================== JSON 数据加载功能 ====================

def run_from_json_file(
    json_file: str,
    max_candidates: int = 10,
    use_api: bool = True,
    save_result: bool = True,
    output_dir: str = None,
    use_cache: bool = True,
    force_refresh: bool = False,
    llm_mode: str = None,
    application_scenario: str = None
) -> Optional[dict]:
    """
    从 JSON 文件加载评测数据并运行排序
    
    JSON 文件格式：
    {
        "source_file": "...",
        "total_count": N,
        "data": [
            {"id": "...", "score": 5.0},
            ...
        ]
    }
    
    第一项作为 query，其余作为 candidates
    
    Args:
        json_file: JSON 文件路径
        max_candidates: 最大候选数量
        use_api: 是否使用真实 API
        save_result: 是否保存结果
        output_dir: 输出目录（默认为 JSON 文件所在目录）
        use_cache: 是否使用持久化缓存（基于 JSON 文件路径）
        force_refresh: 是否强制刷新缓存（忽略已有缓存，重新调用 LLM）
        llm_mode: LLM 模式（"api"/"qwen"/"mock"），优先级高于 use_api 参数
        application_scenario: 应用场景描述，用于指导维度规划
        
    Returns:
        排序结果字典，包含 ground_truth 对比信息
    """
    print("\n" + "=" * 60)
    print("从 JSON 文件加载评测数据并运行排序")
    print("=" * 60)
    
    # 获取持久化缓存实例
    persistent_cache = get_persistent_cache()
    
    # 检查缓存（如果启用缓存且不强制刷新）
    if use_cache and not force_refresh:
        cached_result = persistent_cache.get(json_file)
        if cached_result is not None:
            print("\n✅ 使用缓存的排序结果")
            result = cached_result.get("result", cached_result)
            
            # 打印缓存的评估结果
            if "evaluation" in result:
                print("\n" + "=" * 60)
                print("📊 排序评估结果 (来自缓存)")
                print("=" * 60)
                eval_result = result["evaluation"]
                print(f"  Kendall's Tau: {eval_result['kendall_tau']:.4f}")
                print(f"  Spearman's Rho: {eval_result['spearman_rho']:.4f}")
                print(f"  Top-3 准确率: {eval_result['top_k_accuracy']['top_3']:.2%}")
                print(f"  Top-5 准确率: {eval_result['top_k_accuracy']['top_5']:.2%}")
                print(f"  完全匹配: {'✅' if eval_result['exact_match'] else '❌'}")
            
            return result
    
    # 初始化数据加载器
    print("\n📂 初始化数据加载器...")
    loader = DataLoader()
    
    # 打印索引统计
    stats = loader.path_resolver.get_index_stats()
    print("📊 数据集索引统计:")
    for ds_type, count in stats.items():
        print(f"    {ds_type}: {count} 项")
    
    # 加载 JSON 文件
    print(f"\n📄 加载 JSON 文件: {json_file}")
    try:
        loaded_data = loader.load_json(json_file)
        print_loaded_data_summary(loaded_data)
    except Exception as e:
        print(f"❌ 加载 JSON 文件失败: {e}")
        return None
    
    # 准备排序数据
    try:
        query_image, candidate_images = loader.prepare_for_ranking(
            loaded_data, 
            max_candidates=max_candidates,
            only_valid=True
        )
    except ValueError as e:
        print(f"❌ 准备数据失败: {e}")
        return None
    
    print(f"\n✅ 成功准备排序数据:")
    print(f"    Query 图片: {query_image}")
    print(f"    Candidate 数量: {len(candidate_images)}")
    
    # 获取 ground truth 排序
    ground_truth = loader.get_ground_truth_ranking(loaded_data, only_valid=True)
    ground_truth_ids = [item[0] for item in ground_truth[:max_candidates]]
    
    print(f"\n📋 Ground Truth 排序 (前 {min(len(ground_truth_ids), 10)} 项):")
    for i, (item_id, score) in enumerate(ground_truth[:10], 1):
        print(f"    {i}. {item_id} (分数: {score})")
    
    # 运行排序
    result = run_image_ranking_test(
        query_image=query_image,
        candidate_images=candidate_images,
        query_description=f"数据集类型: {loaded_data.dataset_type}, 源文件: {loaded_data.source_file}",
        use_api=use_api,
        source_json_path=json_file,
        llm_mode=llm_mode,
        application_scenario=application_scenario
    )
    
    if result is None:
        return None
    
    # 添加 ground truth 对比信息
    result["ground_truth"] = {
        "dataset_type": loaded_data.dataset_type,
        "source_file": loaded_data.source_file,
        "ranking": ground_truth_ids,
        "scores": {item[0]: item[1] for item in ground_truth}
    }
    
    # 计算排序一致性
    predicted_ranking = result["final_ranking"]
    result["evaluation"] = evaluate_ranking(
        predicted_ranking=predicted_ranking,
        ground_truth_ranking=ground_truth_ids
    )
    
    # 打印评估结果
    print("\n" + "=" * 60)
    print("📊 排序评估结果")
    print("=" * 60)
    eval_result = result["evaluation"]
    print(f"  Kendall's Tau: {eval_result['kendall_tau']:.4f}")
    print(f"  Spearman's Rho: {eval_result['spearman_rho']:.4f}")
    print(f"  Top-3 准确率: {eval_result['top_k_accuracy']['top_3']:.2%}")
    print(f"  Top-5 准确率: {eval_result['top_k_accuracy']['top_5']:.2%}")
    print(f"  完全匹配: {'✅' if eval_result['exact_match'] else '❌'}")
    
    # 保存结果（使用时间戳唯一命名）
    if save_result:
        if output_dir is None:
            output_dir = os.path.dirname(json_file)
        
        json_basename = os.path.splitext(os.path.basename(json_file))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_filename = f"{json_basename}_ranking_result_{timestamp}.json"
        result_path = os.path.join(output_dir, result_filename)
        
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 结果已保存到: {result_path}")
    
    # 保存到持久化缓存（如果启用）
    if use_cache:
        persistent_cache.set(json_file, result)
        cache_stats = persistent_cache.get_stats()
        print(f"\n📊 持久化缓存统计: {cache_stats['cached_count']} 个缓存文件, {cache_stats['total_size_mb']} MB")
    
    # 显示 Token 统计
    show_llm_statistics()
    
    return result


def evaluate_ranking(
    predicted_ranking: List[str],
    ground_truth_ranking: List[str]
) -> dict:
    """
    评估预测排序与 ground truth 的一致性
    
    Args:
        predicted_ranking: 预测的排序列表
        ground_truth_ranking: ground truth 排序列表
        
    Returns:
        评估结果字典
    """
    # 只比较共同的项
    common_items = set(predicted_ranking) & set(ground_truth_ranking)
    
    if len(common_items) < 2:
        return {
            "kendall_tau": 0.0,
            "spearman_rho": 0.0,
            "top_k_accuracy": {"top_3": 0.0, "top_5": 0.0, "top_10": 0.0},
            "exact_match": False,
            "common_items_count": len(common_items)
        }
    
    # 构建排名映射
    pred_rank = {item: i for i, item in enumerate(predicted_ranking)}
    gt_rank = {item: i for i, item in enumerate(ground_truth_ranking)}
    
    # 计算 Kendall's Tau
    concordant = 0
    discordant = 0
    common_list = list(common_items)
    
    for i in range(len(common_list)):
        for j in range(i + 1, len(common_list)):
            item_i, item_j = common_list[i], common_list[j]
            pred_diff = pred_rank[item_i] - pred_rank[item_j]
            gt_diff = gt_rank[item_i] - gt_rank[item_j]
            
            if pred_diff * gt_diff > 0:
                concordant += 1
            elif pred_diff * gt_diff < 0:
                discordant += 1
    
    total_pairs = concordant + discordant
    kendall_tau = (concordant - discordant) / total_pairs if total_pairs > 0 else 0.0
    
    # 计算 Spearman's Rho
    n = len(common_items)
    d_squared_sum = sum((pred_rank[item] - gt_rank[item]) ** 2 for item in common_items)
    spearman_rho = 1 - (6 * d_squared_sum) / (n * (n ** 2 - 1)) if n > 1 else 0.0
    
    # 计算 Top-K 准确率
    def top_k_accuracy(k):
        if k > len(predicted_ranking) or k > len(ground_truth_ranking):
            k = min(len(predicted_ranking), len(ground_truth_ranking))
        pred_top_k = set(predicted_ranking[:k])
        gt_top_k = set(ground_truth_ranking[:k])
        return len(pred_top_k & gt_top_k) / k if k > 0 else 0.0
    
    return {
        "kendall_tau": kendall_tau,
        "spearman_rho": spearman_rho,
        "top_k_accuracy": {
            "top_3": top_k_accuracy(3),
            "top_5": top_k_accuracy(5),
            "top_10": top_k_accuracy(10)
        },
        "exact_match": predicted_ranking == ground_truth_ranking[:len(predicted_ranking)],
        "common_items_count": len(common_items)
    }


def run_batch_from_json_directory(
    json_dir: str,
    max_candidates: int = 10,
    max_files: int = None,
    use_api: bool = True,
    output_dir: str = None,
    use_cache: bool = True,
    force_refresh: bool = False
) -> List[dict]:
    """
    批量处理目录下的所有 JSON 文件
    
    Args:
        json_dir: JSON 文件目录
        max_candidates: 每个文件的最大候选数量
        max_files: 最大处理文件数量
        use_api: 是否使用真实 API
        output_dir: 输出目录
        use_cache: 是否使用持久化缓存
        force_refresh: 是否强制刷新缓存
        
    Returns:
        所有结果的列表
    """
    print("\n" + "=" * 70)
    print("批量处理 JSON 文件")
    print("=" * 70)
    
    # 查找 JSON 文件
    json_files = glob.glob(os.path.join(json_dir, "*_id_score.json"))
    
    if not json_files:
        print(f"❌ 在目录 {json_dir} 中未找到 *_id_score.json 文件")
        return []
    
    print(f"✅ 找到 {len(json_files)} 个 JSON 文件")
    print(f"📦 缓存模式: {'启用' if use_cache else '禁用'}, 强制刷新: {'是' if force_refresh else '否'}")
    
    if max_files:
        json_files = json_files[:max_files]
        print(f"⚠️  限制处理前 {max_files} 个文件")
    
    results = []
    for i, json_file in enumerate(json_files, 1):
        print(f"\n{'='*70}")
        print(f"处理第 {i}/{len(json_files)} 个文件: {os.path.basename(json_file)}")
        print(f"{'='*70}")
        
        try:
            result = run_from_json_file(
                json_file=json_file,
                max_candidates=max_candidates,
                use_api=use_api,
                save_result=True,
                output_dir=output_dir,
                use_cache=use_cache,
                force_refresh=force_refresh
            )
            if result:
                results.append(result)
        except Exception as e:
            print(f"❌ 处理失败: {e}")
            continue
    
    # 汇总统计
    if results:
        print("\n" + "=" * 70)
        print("📊 批量处理汇总统计")
        print("=" * 70)
        
        avg_kendall = sum(r["evaluation"]["kendall_tau"] for r in results) / len(results)
        avg_spearman = sum(r["evaluation"]["spearman_rho"] for r in results) / len(results)
        avg_top3 = sum(r["evaluation"]["top_k_accuracy"]["top_3"] for r in results) / len(results)
        avg_top5 = sum(r["evaluation"]["top_k_accuracy"]["top_5"] for r in results) / len(results)
        
        print(f"  处理文件数: {len(results)}")
        print(f"  平均 Kendall's Tau: {avg_kendall:.4f}")
        print(f"  平均 Spearman's Rho: {avg_spearman:.4f}")
        print(f"  平均 Top-3 准确率: {avg_top3:.2%}")
        print(f"  平均 Top-5 准确率: {avg_top5:.2%}")
    
    return results


def list_available_json_files():
    """
    列出可用的 JSON 数据文件
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    datasets = {
        "ESB": os.path.join(base_dir, "ESB"),
        "GSO": os.path.join(base_dir, "GSO"),
        "MN40": os.path.join(base_dir, "MN40"),
        "NTU": os.path.join(base_dir, "NTU"),
    }
    
    print("\n" + "=" * 60)
    print("📂 可用的 JSON 数据文件")
    print("=" * 60)
    
    for ds_name, ds_path in datasets.items():
        if os.path.exists(ds_path):
            json_files = glob.glob(os.path.join(ds_path, "*_id_score.json"))
            print(f"\n{ds_name} ({len(json_files)} 个文件):")
            for f in json_files[:3]:
                print(f"    - {os.path.basename(f)}")
            if len(json_files) > 3:
                print(f"    ... 还有 {len(json_files) - 3} 个文件")


# Objaverse 分组文件预设路径
OBJAVERSE_GROUP_FILES = {
    "1": {
        "name": "Character 分组 (348组)",
        "path": "/apdcephfs/share_303565425/DCC3/serenasnliu/GAR/Uni3D/data/test_datasets/objaverse/objaverse_golden_character_groups.json"
    },
    "2": {
        "name": "全部分组",
        "path": "/apdcephfs/share_303565425/DCC3/serenasnliu/GAR/Uni3D/data/test_datasets/objaverse/objaverse_golden_all_groups.json"
    },
    "3": {
        "name": "100组测试集",
        "path": "/apdcephfs/share_303565425/DCC3/serenasnliu/GAR/Uni3D/data/test_datasets/objaverse/objaverse_golden_100_groups.json"
    }
}


def list_available_objaverse_group_files():
    """
    列出可用的 Objaverse 分组文件
    
    Returns:
        dict: 预设文件信息
    """
    print("\n" + "=" * 60)
    print("📂 可用的 Objaverse 分组文件")
    print("=" * 60)
    
    available_files = {}
    for key, info in OBJAVERSE_GROUP_FILES.items():
        path = info["path"]
        exists = os.path.exists(path)
        status = "✅" if exists else "❌"
        print(f"  {key}. {status} {info['name']}")
        print(f"       路径: {path}")
        if exists:
            available_files[key] = info
    
    # 扫描目录下的其他分组文件
    objaverse_dir = "/apdcephfs/share_303565425/DCC3/serenasnliu/GAR/Uni3D/data/test_datasets/objaverse"
    if os.path.exists(objaverse_dir):
        other_files = glob.glob(os.path.join(objaverse_dir, "*_groups.json"))
        preset_paths = [info["path"] for info in OBJAVERSE_GROUP_FILES.values()]
        other_files = [f for f in other_files if f not in preset_paths]
        
        if other_files:
            print(f"\n  其他分组文件 ({len(other_files)} 个):")
            for f in other_files[:5]:
                print(f"    - {os.path.basename(f)}")
            if len(other_files) > 5:
                print(f"    ... 还有 {len(other_files) - 5} 个文件")
    
    return available_files


def select_objaverse_group_file() -> Optional[str]:
    """
    交互式选择 Objaverse 分组文件
    
    Returns:
        str: 选中的文件路径，或 None 表示自定义输入
    """
    available = list_available_objaverse_group_files()
    
    print("\n请选择分组文件:")
    print("  输入数字选择预设文件")
    print("  输入 'c' 或直接输入路径使用自定义文件")
    print("  输入 'q' 返回")
    print("请输入选项: ", end="")
    
    choice = input().strip()
    
    if choice.lower() == 'q':
        return None
    
    if choice in available:
        return available[choice]["path"]
    elif choice.lower() == 'c' or os.path.exists(choice):
        if choice.lower() == 'c':
            print("请输入分组 JSON 文件路径: ", end="")
            custom_path = input().strip()
        else:
            custom_path = choice
        
        if os.path.exists(custom_path):
            return custom_path
        else:
            print(f"❌ 文件不存在: {custom_path}")
            return None
    else:
        print(f"❌ 无效选项: {choice}")
        return None


def demo_mock_mode():
    """
    使用 mock 模式演示图片排序流程（无需真实 API）
    """
    print("\n" + "=" * 70)
    print("🎯 Mock 模式演示 - 图片排序流程")
    print("=" * 70)
    
    # 使用假的图片路径进行演示
    # 注意：mock 模式下不会真正读取图片，只是演示流程
    query_image = "query_image.jpg"
    candidate_images = {
        "candidate_001": "candidate_1.jpg",
        "candidate_002": "candidate_2.jpg",
        "candidate_003": "candidate_3.jpg",
    }
    
    # 初始化 mock 模式
    llm = get_llm_interface(mode="mock", force_new=True)
    print(f"\n✅ 使用 Mock 模式（不需要真实图片和 API）")
    
    # 创建流水线
    pipeline = RankingPipeline(verbose=True)
    
    # 由于 mock 模式会检查文件存在，我们直接使用文本模式演示
    print("\n📝 使用文本模式演示排序流程...")
    
    query = "我正在寻找一个与图片中物体相似的替代品，主要用于日常办公场景。"
    
    candidate_info = {
        "candidate_001": "物品1：一个红色的金属杯子",
        "candidate_002": "物品2：一个蓝色的塑料水壶",
        "candidate_003": "物品3：一个透明的玻璃杯",
    }
    
    result = pipeline.run(
        query=query,
        candidate_ids=list(candidate_info.keys()),
        candidate_info=candidate_info
    )
    
    # 打印结果
    print("\n" + "-" * 50)
    print("✅ Mock 排序完成！最终结果:")
    print("-" * 50)
    for rank, cid in enumerate(result["final_ranking"], 1):
        print(f"  {rank}. {cid}")
    
    # 显示统计
    show_llm_statistics()
    
    return result


def run_group_ranking_interactive():
    """
    交互式运行分组排序
    
    支持的数据格式：
    - Objaverse 分组格式：image_path 字段包含完整的图片绝对路径
    - 其他分组格式：根据需要进行路径解析
    """
    print("\n" + "=" * 60)
    print("🎯 分组排序 - Group Ranking")
    print("=" * 60)
    
    print("""
请选择操作:

1. 运行单个分组排序（测试）
2. 运行批量分组排序
3. 查看分组 JSON 文件信息
4. 列出可用的 Objaverse 分组文件
5. 返回主菜单

请输入选项 (1-5): """, end="")
    
    try:
        sub_choice = input().strip()
    except:
        return
    
    if sub_choice == "1":
        # 单个分组测试
        print("\n🎯 选择分组文件...")
        json_file = select_objaverse_group_file()
        
        if json_file is None:
            return
        
        # 加载并显示分组信息
        loader = GroupDataLoader()
        try:
            grouped_data = loader.load_grouped_json(json_file)
            print(f"\n📂 分组文件信息:")
            print(f"  文件: {os.path.basename(json_file)}")
            print(f"  总分组数: {grouped_data.total_groups}")
            print(f"  分组大小: {grouped_data.group_size}")
            
            # 显示有效性检查（检查图片路径是否存在）
            sample_group = grouped_data.groups[0] if grouped_data.groups else None
            if sample_group:
                valid_count = sample_group.valid_count
                print(f"  首组有效物体: {valid_count}/{sample_group.size}")
        except Exception as e:
            print(f"❌ 加载文件失败: {e}")
            return
        
        print("\n请输入要测试的分组索引 (默认 0): ", end="")
        group_idx_input = input().strip()
        group_idx = int(group_idx_input) if group_idx_input else 0
        
        if group_idx >= len(grouped_data.groups):
            print(f"❌ 分组索引超出范围（最大: {len(grouped_data.groups) - 1}）")
            return
        
        # 选择 LLM 模式
        llm_mode = select_llm_mode()
        
        # 选择应用场景
        application_scenario = select_application_scenario()
        
        # 运行排序
        run_group_ranking(
            json_file=json_file,
            group_indices=[group_idx],
            llm_mode=llm_mode,
            application_scenario=application_scenario,
            verbose=True
        )
    
    elif sub_choice == "2":
        # 批量排序
        print("\n🎯 选择分组文件...")
        json_file = select_objaverse_group_file()
        
        if json_file is None:
            return
        
        # 加载并显示分组信息
        loader = GroupDataLoader()
        try:
            grouped_data = loader.load_grouped_json(json_file)
            print(f"\n📂 分组文件信息:")
            print(f"  文件: {os.path.basename(json_file)}")
            print(f"  总分组数: {grouped_data.total_groups}")
            print(f"  分组大小: {grouped_data.group_size}")
            
            # 显示有效性检查
            sample_group = grouped_data.groups[0] if grouped_data.groups else None
            if sample_group:
                valid_count = sample_group.valid_count
                print(f"  首组有效物体: {valid_count}/{sample_group.size}")
        except Exception as e:
            print(f"❌ 加载文件失败: {e}")
            return
        
        print("\n请输入最大处理分组数 (直接回车处理全部): ", end="")
        max_groups_input = input().strip()
        max_groups = int(max_groups_input) if max_groups_input else None
        
        print("请输入随机种子 (默认 42，用于复现结果): ", end="")
        seed_input = input().strip()
        random_seed = int(seed_input) if seed_input else 42
        
        # 选择 LLM 模式
        llm_mode = select_llm_mode()
        
        # 选择应用场景
        application_scenario = select_application_scenario()
        
        # 运行批量排序
        run_group_ranking(
            json_file=json_file,
            max_groups=max_groups,
            llm_mode=llm_mode,
            application_scenario=application_scenario,
            random_seed=random_seed,
            verbose=True
        )
    
    elif sub_choice == "3":
        # 查看文件信息
        print("\n🎯 选择分组文件...")
        json_file = select_objaverse_group_file()
        
        if json_file is None:
            return
        
        loader = GroupDataLoader()
        try:
            grouped_data = loader.load_grouped_json(json_file)
        except Exception as e:
            print(f"❌ 加载文件失败: {e}")
            return
        
        print(f"\n{'='*60}")
        print(f"📂 分组文件信息")
        print(f"{'='*60}")
        print(f"  文件: {os.path.basename(json_file)}")
        print(f"  分组大小: {grouped_data.group_size}")
        print(f"  总分组数: {grouped_data.total_groups}")
        print(f"  类别数: {grouped_data.total_categories}")
        
        print(f"\n📊 类别统计:")
        for cat, stats in grouped_data.category_statistics.items():
            print(f"  {cat}:")
            print(f"    物体总数: {stats.get('total_objects', 0)}")
            print(f"    分组数: {stats.get('num_groups', 0)}")
        
        print(f"\n📦 前10个分组:")
        for group in grouped_data.groups[:10]:
            valid_count = group.valid_count
            print(f"  {group.group_id}: {group.size} 个物体, {valid_count} 个有效")
        
        if len(grouped_data.groups) > 10:
            print(f"  ... 还有 {len(grouped_data.groups) - 10} 个分组")
        
        # 显示首个分组的详细信息（检查 image_path 是否有效）
        if grouped_data.groups:
            first_group = grouped_data.groups[0]
            print(f"\n📸 首组物体图片路径检查:")
            for obj in first_group.objects[:3]:
                exists = "✅" if obj.is_valid() else "❌"
                print(f"  {exists} {obj.object_id}")
                print(f"      {obj.image_path}")
            if first_group.size > 3:
                print(f"  ... 还有 {first_group.size - 3} 个物体")
    
    elif sub_choice == "4":
        # 列出可用的 Objaverse 分组文件
        list_available_objaverse_group_files()
    
    # 选项 5 直接返回


def manage_cache():
    """
    缓存管理 - 查看和清除 LLM 调用缓存
    """
    print("\n" + "=" * 60)
    print("📦 LLM 调用缓存管理")
    print("=" * 60)
    
    # 获取持久化缓存实例
    persistent_cache = get_persistent_cache()
    
    # 显示缓存统计
    stats = persistent_cache.get_stats()
    print(f"\n📊 缓存统计:")
    print(f"    缓存目录: {stats['cache_dir']}")
    print(f"    缓存文件数量: {stats['cached_count']}")
    print(f"    总大小: {stats['total_size_mb']} MB")
    print(f"    命中次数: {stats['hit_count']}")
    print(f"    未命中次数: {stats['miss_count']}")
    print(f"    命中率: {stats['hit_rate']:.2%}")
    
    # 列出缓存文件（区分完整缓存和阶段缓存）
    cached_files = persistent_cache.list_cached_files()
    
    # 分类显示
    complete_caches = [cf for cf in cached_files if not cf['cache_file'].endswith('_stage.json')]
    stage_caches = [cf for cf in cached_files if cf['cache_file'].endswith('_stage.json')]
    
    if complete_caches:
        print(f"\n📋 完整结果缓存 ({len(complete_caches)} 个):")
        for i, cf in enumerate(complete_caches, 1):
            size_kb = cf['size'] / 1024
            print(f"    {i}. {cf['cache_file']}")
            print(f"       源文件: {os.path.basename(cf['source_json'])}")
            print(f"       大小: {size_kb:.1f} KB")
    
    if stage_caches:
        print(f"\n📋 阶段缓存 ({len(stage_caches)} 个):")
        # 按源文件分组显示
        stage_by_source = {}
        for cf in stage_caches:
            source = cf['source_json']
            if source not in stage_by_source:
                stage_by_source[source] = []
            stage_by_source[source].append(cf)
        
        for source, stages in stage_by_source.items():
            print(f"    📁 {os.path.basename(source)}:")
            for cf in stages:
                # 从文件名中提取阶段名
                stage_name = cf['cache_file'].replace('_stage.json', '').split('_')[-1]
                size_kb = cf['size'] / 1024
                print(f"       - {stage_name} ({size_kb:.1f} KB)")
    
    if not cached_files:
        print("\n📋 当前没有缓存文件")
    
    # 缓存操作选项
    print("""
缓存操作选项：
  1. 清除所有缓存（包括完整结果和阶段缓存）
  2. 清除指定 JSON 文件的所有缓存（包括阶段缓存）
  3. 仅清除阶段缓存
  4. 返回主菜单

请输入选项 (1-4): """, end="")
    
    try:
        sub_choice = input().strip()
    except:
        return
    
    if sub_choice == "1":
        print("\n⚠️  确认清除所有缓存？(y/n): ", end="")
        confirm = input().strip().lower()
        if confirm == 'y':
            count = persistent_cache.clear_all()
            print(f"✅ 已清除 {count} 个缓存文件")
        else:
            print("已取消")
    
    elif sub_choice == "2":
        print("\n请输入源 JSON 文件路径: ", end="")
        json_path = input().strip()
        deleted = False
        if persistent_cache.has_cache(json_path):
            persistent_cache.delete(json_path)
            deleted = True
        stage_count = persistent_cache.delete_all_stages(json_path)
        if deleted or stage_count > 0:
            print(f"✅ 已删除完整缓存和 {stage_count} 个阶段缓存")
        else:
            print("❌ 未找到该文件的缓存")
    
    elif sub_choice == "3":
        print("\n⚠️  确认清除所有阶段缓存？(y/n): ", end="")
        confirm = input().strip().lower()
        if confirm == 'y':
            count = 0
            cache_dir = persistent_cache.cache_dir
            if os.path.exists(cache_dir):
                for filename in os.listdir(cache_dir):
                    if filename.endswith("_stage.json"):
                        try:
                            os.remove(os.path.join(cache_dir, filename))
                            count += 1
                        except IOError:
                            pass
            print(f"✅ 已清除 {count} 个阶段缓存文件")
        else:
            print("已取消")


def main():
    """
    主函数 - 提供交互式选项
    """
    print("\n" + "=" * 70)
    print("🎯 Agent-Skills 图片排序系统")
    print("=" * 70)
    
    print("""
请选择运行模式：

1. Mock 模式演示（无需 API Key 和真实图片）
2. API 测试 - 连接测试（Gemini / QWEN）
3. API 测试 - 从目录加载图片进行排序
4. API 测试 - 从 TXT 文件加载图片进行排序
5. API 测试 - 从 JSON 文件加载评测数据进行排序 ⭐
6. API 测试 - 批量处理 JSON 文件
7. 分组排序 - 从分组 JSON 文件运行排序 🆕
8. 查看可用的 JSON 数据文件
9. 缓存管理（查看/清除 LLM 调用缓存）
10. 自定义测试

请输入选项 (1-10): """)    
    try:
        choice = input().strip()
    except:
        choice = "1"
    
    if choice == "1":
        demo_mock_mode()
    
    elif choice == "2":
        # 选择 LLM 模式
        print("\n请选择要测试的 LLM:")
        print("  1. Gemini (默认)")
        print("  2. QWEN (通义千问)")
        print("请输入选项 (1-2, 默认 1): ", end="")
        llm_choice = input().strip()
        
        if llm_choice == "2":
            # 测试 QWEN
            api_key = os.environ.get('QWEN_API_KEY')
            if not api_key:
                print("\n⚠️  未检测到 QWEN_API_KEY 环境变量")
                print("   请设置后重试")
                return
            test_qwen_connection()
        else:
            # 测试 Gemini
            api_key = os.environ.get('GEMINI_API_KEY')
            if not api_key:
                print("\n⚠️  未检测到 GEMINI_API_KEY 环境变量")
                print("   请设置后重试")
                return
            test_api_connection()
    
    elif choice == "3":
        # 从目录加载图片
        print("\n请输入查询图片路径: ", end="")
        query_image = input().strip()
        print("请输入候选图片目录: ", end="")
        candidate_dir = input().strip()
        print("请输入查询描述（可选，直接回车跳过）: ", end="")
        description = input().strip()
        
        run_batch_image_ranking(
            query_image=query_image,
            candidate_images_dir=candidate_dir,
            query_description=description,
            use_api=True
        )
    
    elif choice == "4":
        # 从 TXT 文件加载
        print("\n请输入包含图片路径的 TXT 文件: ", end="")
        txt_file = input().strip()
        print("请输入查询描述（可选，直接回车跳过）: ", end="")
        description = input().strip()
        
        run_from_txt_file(
            txt_file=txt_file,
            query_description=description,
            use_api=True
        )
    
    elif choice == "5":
        # 从 JSON 文件加载评测数据
        list_available_json_files()
        print("\n请输入 JSON 文件路径: ", end="")
        json_file = input().strip()
        print("请输入最大候选数量（默认 10）: ", end="")
        max_candidates_input = input().strip()
        max_candidates = int(max_candidates_input) if max_candidates_input else 10
        print("是否使用缓存？(y/n, 默认 y): ", end="")
        use_cache_input = input().strip().lower()
        use_cache = use_cache_input != 'n'
        force_refresh = False
        if use_cache:
            print("是否强制刷新缓存？(y/n, 默认 n): ", end="")
            force_refresh_input = input().strip().lower()
            force_refresh = force_refresh_input == 'y'
        
        # 选择 LLM 模式
        llm_mode = select_llm_mode()
        
        # 选择应用场景
        application_scenario = select_application_scenario()
        
        run_from_json_file(
            json_file=json_file,
            max_candidates=max_candidates,
            use_api=(llm_mode != "mock"),
            use_cache=use_cache,
            force_refresh=force_refresh,
            llm_mode=llm_mode,
            application_scenario=application_scenario
        )
    
    elif choice == "6":
        # 批量处理 JSON 文件
        list_available_json_files()
        print("\n请输入 JSON 文件目录: ", end="")
        json_dir = input().strip()
        print("请输入最大候选数量（默认 10）: ", end="")
        max_candidates_input = input().strip()
        max_candidates = int(max_candidates_input) if max_candidates_input else 10
        print("请输入最大处理文件数量（直接回车处理全部）: ", end="")
        max_files_input = input().strip()
        max_files = int(max_files_input) if max_files_input else None
        print("是否使用缓存？(y/n, 默认 y): ", end="")
        use_cache_input = input().strip().lower()
        use_cache = use_cache_input != 'n'
        force_refresh = False
        if use_cache:
            print("是否强制刷新缓存？(y/n, 默认 n): ", end="")
            force_refresh_input = input().strip().lower()
            force_refresh = force_refresh_input == 'y'
        
        run_batch_from_json_directory(
            json_dir=json_dir,
            max_candidates=max_candidates,
            max_files=max_files,
            use_api=True,
            use_cache=use_cache,
            force_refresh=force_refresh
        )
    
    elif choice == "7":
        # 分组排序
        run_group_ranking_interactive()
    
    elif choice == "8":
        # 查看可用的 JSON 数据文件
        list_available_json_files()
    
    elif choice == "9":
        # 缓存管理
        manage_cache()
    
    elif choice == "10":
        # 自定义测试 - 展示如何在代码中使用
        print("""
=== 自定义测试示例 ===

方式1: 使用图片路径直接运行排序

```python
from run_image_test import run_image_ranking_test
from llm_interface import get_llm_interface

# 初始化 API 模式
get_llm_interface(mode="api", force_new=True)

# 定义查询图片和候选图片
query_image = "/path/to/query.jpg"
candidate_images = {
    "item_001": "/path/to/candidate1.jpg",
    "item_002": "/path/to/candidate2.jpg",
    "item_003": "/path/to/candidate3.jpg",
}

# 运行排序
result = run_image_ranking_test(
    query_image=query_image,
    candidate_images=candidate_images,
    query_description="可选的补充描述",
    use_api=True
)

# 结果
print("最终排序:", result["final_ranking"])
```

方式2: 从 JSON 评测数据文件加载并运行排序

```python
from run_image_test import run_from_json_file

# 直接从 JSON 文件加载数据并运行排序
result = run_from_json_file(
    json_file="d:/3d-object-数据集/1000条黄金数据/ESB/search_lists_01_search_list_01_adjusted_id_score.json",
    max_candidates=10,
    use_api=True
)

# 结果包含排序和评估信息
print("预测排序:", result["final_ranking"])
print("Ground Truth:", result["ground_truth"]["ranking"])
print("Kendall's Tau:", result["evaluation"]["kendall_tau"])
```

方式3: 批量处理目录下的所有 JSON 文件

```python
from run_image_test import run_batch_from_json_directory

results = run_batch_from_json_directory(
    json_dir="d:/3d-object-数据集/1000条黄金数据/ESB",
    max_candidates=10,
    max_files=5,  # 只处理前5个文件
    use_api=True
)
```
""")
    
    else:
        print("无效选项")


if __name__ == "__main__":
    main()
