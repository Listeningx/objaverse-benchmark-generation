"""
LLM 调用接口模块

提供统一的 LLM 调用接口，支持 mock 实现和真实 API 调用。
支持文本和图片的多模态输入。
支持一次性发送所有图像进行批量处理。
"""

from datetime import datetime
import json
import re
import os
import mimetypes
import hashlib
import requests
from typing import Optional, List, Union, Dict, Any

GENAI_API_KEY = "AIzaSyAGOSvLEhVW-Vw1KVozU1Mtu9jKY7eFBXc"  # 替换为你的 Gemini API Key
os.environ['GEMINI_API_KEY']='AIzaSyAGOSvLEhVW-Vw1KVozU1Mtu9jKY7eFBXc'

# QWEN API 配置
QWEN_API_KEY = "sk-188b54b371b54f1a9ed4cd737df7a92b"  # 替换为你的 QWEN API Key
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # 阿里云 DashScope 兼容模式
# QWEN_BASE_URL = "http://v2.open.venus.oa.com/llmproxy/chat/completions"
# ENV_VENUS_OPENAPI_SECRET_ID= "XyxAYLiQvIYXK2kbL3KOR15s"
# os.environ['ENV_VENUS_OPENAPI_SECRET_ID']=ENV_VENUS_OPENAPI_SECRET_ID

# QWEN_DEFAULT_MODEL = "qwen3-vl-235b-a22b-thinking"  # 默认使用 QWEN-VL-Max 多模态模型
QWEN_DEFAULT_MODEL = "qwen3.5-397b-a17b"
# QWEN_DEFAULT_MODEL = "qwen3.5-flash"

# QWEN 调用模式配置
# "requests": 使用 requests 库直接调用 HTTP API（推荐用于自部署模型）
# "openai_sdk": 使用 OpenAI SDK 兼容模式调用
# QWEN_CALL_MODE = "requests"  # 默认使用 requests 模式
QWEN_CALL_MODE = "openai_sdk" 

os.environ['QWEN_API_KEY'] = QWEN_API_KEY

# 尝试导入 Google GenAI 库
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    print("警告: google-genai 库未安装，Gemini API 模式将不可用。请运行: pip install google-genai")

# 尝试导入 OpenAI 库（QWEN 使用 OpenAI 兼容协议）
try:
    from openai import OpenAI
    import base64
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("警告: openai 库未安装，QWEN API 模式将不可用。请运行: pip install openai")


class ResultCache:
    """
    结果缓存类
    
    用于缓存 Skill 的中间结果，避免重复计算。
    """
    
    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._hit_count = 0
        self._miss_count = 0
    
    def _generate_key(self, skill_name: str, input_data: Any) -> str:
        """生成缓存键"""
        # 将输入数据序列化为字符串
        if isinstance(input_data, dict):
            # 对字典进行排序以保证一致性
            serialized = json.dumps(input_data, sort_keys=True, ensure_ascii=False)
        else:
            serialized = str(input_data)
        
        # 使用 MD5 生成固定长度的键
        hash_value = hashlib.md5(serialized.encode('utf-8')).hexdigest()
        return f"{skill_name}:{hash_value}"
    
    def get(self, skill_name: str, input_data: Any) -> Optional[Any]:
        """
        从缓存获取结果
        
        Args:
            skill_name: Skill 名称
            input_data: 输入数据
            
        Returns:
            缓存的结果，如果不存在返回 None
        """
        key = self._generate_key(skill_name, input_data)
        if key in self._cache:
            self._hit_count += 1
            print(f"  📦 缓存命中: {skill_name}")
            return self._cache[key]
        self._miss_count += 1
        return None
    
    def set(self, skill_name: str, input_data: Any, result: Any) -> None:
        """
        将结果存入缓存
        
        Args:
            skill_name: Skill 名称
            input_data: 输入数据
            result: 计算结果
        """
        key = self._generate_key(skill_name, input_data)
        self._cache[key] = result
        print(f"  💾 已缓存: {skill_name}")
    
    def clear(self) -> None:
        """清空缓存"""
        self._cache.clear()
        self._hit_count = 0
        self._miss_count = 0
    
    def get_stats(self) -> Dict[str, int]:
        """获取缓存统计信息"""
        return {
            "cache_size": len(self._cache),
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "hit_rate": self._hit_count / (self._hit_count + self._miss_count) if (self._hit_count + self._miss_count) > 0 else 0
        }


class PersistentCache:
    """
    持久化缓存类
    
    基于输入 JSON 文件路径作为唯一标识，将 LLM 调用结果缓存到本地文件。
    每次调用成功后自动保存，下次调用前先检查缓存。
    """
    
    # 默认缓存目录
    DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm_cache")
    
    def __init__(self, cache_dir: str = None):
        """
        初始化持久化缓存
        
        Args:
            cache_dir: 缓存目录路径，默认为当前目录下的 llm_cache 文件夹
        """
        self.cache_dir = cache_dir or self.DEFAULT_CACHE_DIR
        self._hit_count = 0
        self._miss_count = 0
        
        # 确保缓存目录存在
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            print(f"📁 已创建缓存目录: {self.cache_dir}")
    
    def _generate_cache_filename(self, json_file_path: str) -> str:
        """
        根据 JSON 文件路径生成唯一的缓存文件名
        
        Args:
            json_file_path: 输入 JSON 文件的路径
            
        Returns:
            缓存文件的完整路径
        """
        # 标准化路径
        normalized_path = os.path.normpath(os.path.abspath(json_file_path))
        
        # 使用 MD5 生成唯一标识
        path_hash = hashlib.md5(normalized_path.encode('utf-8')).hexdigest()[:12]
        
        # 提取原文件名作为前缀（方便识别）
        original_name = os.path.splitext(os.path.basename(json_file_path))[0]
        # 清理文件名中的非法字符
        safe_name = re.sub(r'[<>:"/\\|?*]', '_', original_name)[:50]
        
        cache_filename = f"{safe_name}_{path_hash}_cache.json"
        return os.path.join(self.cache_dir, cache_filename)
    
    def has_cache(self, json_file_path: str) -> bool:
        """
        检查是否存在缓存
        
        Args:
            json_file_path: 输入 JSON 文件的路径
            
        Returns:
            是否存在缓存
        """
        cache_path = self._generate_cache_filename(json_file_path)
        return os.path.exists(cache_path)
    
    def get(self, json_file_path: str) -> Optional[Dict[str, Any]]:
        """
        从缓存获取结果
        
        Args:
            json_file_path: 输入 JSON 文件的路径
            
        Returns:
            缓存的结果字典，如果不存在返回 None
        """
        cache_path = self._generate_cache_filename(json_file_path)
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                
                self._hit_count += 1
                print(f"  📦 持久化缓存命中: {os.path.basename(json_file_path)}")
                print(f"     缓存文件: {os.path.basename(cache_path)}")
                return cached_data
            except (json.JSONDecodeError, IOError) as e:
                print(f"  ⚠️ 读取缓存失败: {e}")
                return None
        
        self._miss_count += 1
        return None
    
    def set(self, json_file_path: str, result: Dict[str, Any]) -> bool:
        """
        将结果存入缓存
        
        Args:
            json_file_path: 输入 JSON 文件的路径
            result: 要缓存的结果字典
            
        Returns:
            是否保存成功
        """
        cache_path = self._generate_cache_filename(json_file_path)
        
        try:
            # 添加元数据
            cache_data = {
                "source_json": json_file_path,
                "cached_at": json.dumps({"timestamp": str(os.path.getmtime(json_file_path)) if os.path.exists(json_file_path) else "unknown"}),
                "result": result
            }
            
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            
            print(f"  💾 已持久化缓存: {os.path.basename(json_file_path)}")
            print(f"     缓存文件: {os.path.basename(cache_path)}")
            return True
        except IOError as e:
            print(f"  ⚠️ 保存缓存失败: {e}")
            return False
    
    def delete(self, json_file_path: str) -> bool:
        """
        删除指定的缓存
        
        Args:
            json_file_path: 输入 JSON 文件的路径
            
        Returns:
            是否删除成功
        """
        cache_path = self._generate_cache_filename(json_file_path)
        
        if os.path.exists(cache_path):
            try:
                os.remove(cache_path)
                print(f"  🗑️ 已删除缓存: {os.path.basename(cache_path)}")
                return True
            except IOError as e:
                print(f"  ⚠️ 删除缓存失败: {e}")
                return False
        return False
    
    # ============ 阶段级别缓存方法 ============
    
    def _generate_stage_cache_filename(self, json_file_path: str, stage_name: str) -> str:
        """
        根据 JSON 文件路径和阶段名称生成唯一的阶段缓存文件名
        
        Args:
            json_file_path: 输入 JSON 文件的路径
            stage_name: 阶段名称（如 DimensionPlanner, BatchDescriptor 等）
            
        Returns:
            阶段缓存文件的完整路径
        """
        # 标准化路径
        normalized_path = os.path.normpath(os.path.abspath(json_file_path))
        
        # 使用 MD5 生成唯一标识
        path_hash = hashlib.md5(normalized_path.encode('utf-8')).hexdigest()[:12]
        
        # 提取原文件名作为前缀（方便识别）
        original_name = os.path.splitext(os.path.basename(json_file_path))[0]
        # 清理文件名中的非法字符
        safe_name = re.sub(r'[<>:"/\\|?*]', '_', original_name)[:30]
        safe_stage = re.sub(r'[<>:"/\\|?*]', '_', stage_name)[:20]
        
        cache_filename = f"{safe_name}_{path_hash}_{safe_stage}_stage.json"
        return os.path.join(self.cache_dir, cache_filename)
    
    def get_stage(self, json_file_path: str, stage_name: str) -> Optional[Dict[str, Any]]:
        """
        从缓存获取指定阶段的结果
        
        Args:
            json_file_path: 输入 JSON 文件的路径
            stage_name: 阶段名称
            
        Returns:
            缓存的阶段结果，如果不存在返回 None
        """
        cache_path = self._generate_stage_cache_filename(json_file_path, stage_name)
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                
                self._hit_count += 1
                print(f"  📦 阶段缓存命中: {stage_name}")
                return cached_data.get("result")
            except (json.JSONDecodeError, IOError) as e:
                print(f"  ⚠️ 读取阶段缓存失败: {e}")
                return None
        
        self._miss_count += 1
        return None
    
    def set_stage(self, json_file_path: str, stage_name: str, result: Dict[str, Any]) -> bool:
        """
        将阶段结果存入缓存
        
        Args:
            json_file_path: 输入 JSON 文件的路径
            stage_name: 阶段名称
            result: 要缓存的阶段结果
            
        Returns:
            是否保存成功
        """
        cache_path = self._generate_stage_cache_filename(json_file_path, stage_name)
        
        try:
            # 添加元数据
            cache_data = {
                "source_json": json_file_path,
                "stage_name": stage_name,
                "cached_at": datetime.now().isoformat(),
                "result": result
            }
            
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            
            print(f"  💾 已持久化阶段缓存: {stage_name}")
            return True
        except IOError as e:
            print(f"  ⚠️ 保存阶段缓存失败: {e}")
            return False
    
    def has_stage_cache(self, json_file_path: str, stage_name: str) -> bool:
        """
        检查指定阶段是否存在缓存
        
        Args:
            json_file_path: 输入 JSON 文件的路径
            stage_name: 阶段名称
            
        Returns:
            是否存在阶段缓存
        """
        cache_path = self._generate_stage_cache_filename(json_file_path, stage_name)
        return os.path.exists(cache_path)
    
    def get_all_stages(self, json_file_path: str) -> Dict[str, Any]:
        """
        获取指定 JSON 文件的所有阶段缓存
        
        Args:
            json_file_path: 输入 JSON 文件的路径
            
        Returns:
            所有阶段缓存的字典 {stage_name: result}
        """
        results = {}
        normalized_path = os.path.normpath(os.path.abspath(json_file_path))
        path_hash = hashlib.md5(normalized_path.encode('utf-8')).hexdigest()[:12]
        
        if os.path.exists(self.cache_dir):
            for filename in os.listdir(self.cache_dir):
                if path_hash in filename and filename.endswith("_stage.json"):
                    try:
                        with open(os.path.join(self.cache_dir, filename), 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        stage_name = data.get("stage_name", "unknown")
                        results[stage_name] = data.get("result")
                    except (json.JSONDecodeError, IOError):
                        pass
        
        return results
    
    def delete_all_stages(self, json_file_path: str) -> int:
        """
        删除指定 JSON 文件的所有阶段缓存
        
        Args:
            json_file_path: 输入 JSON 文件的路径
            
        Returns:
            删除的阶段缓存数量
        """
        count = 0
        normalized_path = os.path.normpath(os.path.abspath(json_file_path))
        path_hash = hashlib.md5(normalized_path.encode('utf-8')).hexdigest()[:12]
        
        if os.path.exists(self.cache_dir):
            for filename in os.listdir(self.cache_dir):
                if path_hash in filename and filename.endswith("_stage.json"):
                    try:
                        os.remove(os.path.join(self.cache_dir, filename))
                        count += 1
                    except IOError:
                        pass
        
        if count > 0:
            print(f"  🗑️ 已删除 {count} 个阶段缓存")
        return count
    
    def clear_all(self) -> int:
        """
        清除所有缓存（包括完整结果缓存和阶段缓存）
        
        Returns:
            删除的缓存文件数量
        """
        count = 0
        if os.path.exists(self.cache_dir):
            for filename in os.listdir(self.cache_dir):
                # 同时删除完整缓存和阶段缓存
                if filename.endswith("_cache.json") or filename.endswith("_stage.json"):
                    try:
                        os.remove(os.path.join(self.cache_dir, filename))
                        count += 1
                    except IOError:
                        pass
        
        self._hit_count = 0
        self._miss_count = 0
        print(f"  🗑️ 已清除 {count} 个缓存文件")
        return count
    
    def list_cached_files(self) -> List[Dict[str, str]]:
        """
        列出所有缓存文件（包括完整结果缓存和阶段缓存）
        
        Returns:
            缓存文件信息列表
        """
        cached = []
        if os.path.exists(self.cache_dir):
            for filename in os.listdir(self.cache_dir):
                # 同时匹配完整缓存 (*_cache.json) 和阶段缓存 (*_stage.json)
                if filename.endswith("_cache.json") or filename.endswith("_stage.json"):
                    cache_path = os.path.join(self.cache_dir, filename)
                    try:
                        with open(cache_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        cached.append({
                            "cache_file": filename,
                            "source_json": data.get("source_json", "unknown"),
                            "size": os.path.getsize(cache_path)
                        })
                    except (json.JSONDecodeError, IOError):
                        cached.append({
                            "cache_file": filename,
                            "source_json": "无法读取",
                            "size": os.path.getsize(cache_path)
                        })
        return cached
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        cached_files = self.list_cached_files()
        total_size = sum(f["size"] for f in cached_files)
        
        return {
            "cache_dir": self.cache_dir,
            "cached_count": len(cached_files),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "hit_rate": self._hit_count / (self._hit_count + self._miss_count) if (self._hit_count + self._miss_count) > 0 else 0
        }


# 全局持久化缓存实例
_persistent_cache: Optional[PersistentCache] = None


def get_persistent_cache(cache_dir: str = None) -> PersistentCache:
    """
    获取全局持久化缓存实例
    
    Args:
        cache_dir: 缓存目录路径
        
    Returns:
        PersistentCache 实例
    """
    global _persistent_cache
    if _persistent_cache is None:
        _persistent_cache = PersistentCache(cache_dir)
    return _persistent_cache


class LLMInterface:
    """
    LLM 调用接口类
    
    提供统一的 call_llm 方法，支持 mock 模式和真实 API 模式。
    支持文本和图片的多模态输入。
    支持一次性发送所有图像进行批量处理。
    支持 Gemini 和 QWEN 两种大模型。
    """
    
    # 支持的模式
    SUPPORTED_MODES = ["mock", "api", "qwen"]
    
    def __init__(self, mode: str = "mock", api_key: Optional[str] = None, model: str = None, qwen_call_mode: str = None):
        """
        初始化 LLM 接口
        
        Args:
            mode: 运行模式，"mock"、"api"（Gemini）或 "qwen"
            api_key: API 密钥（api/qwen 模式下需要，也可通过环境变量设置）
            model: 模型名称
                - Gemini 模式默认为 gemini-3-flash-preview
                - QWEN 模式默认为 qwen-vl-max
            qwen_call_mode: QWEN 调用模式
                - "requests": 使用 requests 库直接调用 HTTP API（推荐用于自部署模型）
                - "openai_sdk": 使用 OpenAI SDK 兼容模式调用
                - 默认使用配置文件中的 QWEN_CALL_MODE 值
        """
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(f"不支持的模式: {mode}，支持的模式: {self.SUPPORTED_MODES}")
        
        self.mode = mode
        self.call_count = 0
        self.call_history = []
        self.client = None
        
        # Token 统计
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.token_history = []  # 每次调用的 token 记录
        
        # QWEN 调用模式（requests 或 openai_sdk）
        self.qwen_call_mode = qwen_call_mode or QWEN_CALL_MODE
        
        # 根据模式设置默认模型和 API Key
        if self.mode == "qwen":
            self.api_key = api_key or os.environ.get('QWEN_API_KEY')
            self.model =  QWEN_DEFAULT_MODEL
            self.base_url = QWEN_BASE_URL
        else:
            # Gemini 或 mock 模式
            self.api_key = api_key or os.environ.get('GEMINI_API_KEY')
            self.model = model or "gemini-3-flash-preview"
            self.base_url = None
        
        # 结果缓存
        self.result_cache = ResultCache()
        
        # 预加载的图像 Parts 缓存
        self._image_parts_cache: Dict[str, Any] = {}
        
        # 如果是 API 模式，初始化客户端
        if self.mode == "api":
            self._init_client()
        elif self.mode == "qwen":
            self._init_qwen_client()
    
    def preload_images(self, image_paths: List[str]) -> None:
        """
        预加载图像，将图像转换为 API 可接受的格式并缓存
        
        Args:
            image_paths: 图像文件路径列表
        """
        if self.mode != "api":
            return
        
        print(f"\n📷 预加载 {len(image_paths)} 张图像...")
        
        for idx, img_path in enumerate(image_paths, 1):
            if img_path in self._image_parts_cache:
                print(f"  ✅ 图片 {idx}/{len(image_paths)}: {os.path.basename(img_path)} (已缓存)")
                continue
            
            try:
                img_part = self._prepare_image_part(img_path)
                self._image_parts_cache[img_path] = img_part
                print(f"  ✅ 图片 {idx}/{len(image_paths)}: {os.path.basename(img_path)} (已加载)")
            except Exception as e:
                print(f"  ❌ 图片 {idx}/{len(image_paths)}: {img_path}，错误：{e}")
    
    def get_cached_image_part(self, img_path: str):
        """
        获取缓存的图像 Part
        
        Args:
            img_path: 图像文件路径
            
        Returns:
            缓存的图像 Part，如果不存在则返回 None
        """
        return self._image_parts_cache.get(img_path)
    
    def call_llm(self, prompt: str, expected_format: str = "json") -> str:
        """
        调用 LLM 的统一接口（纯文本）
        
        Args:
            prompt: 发送给 LLM 的提示词
            expected_format: 期望的输出格式，"json" 或 "text"
            
        Returns:
            str: LLM 的响应内容
        """
        self.call_count += 1
        self.call_history.append({
            "call_id": self.call_count,
            "prompt_preview": prompt[:200] + "..." if len(prompt) > 200 else prompt,
            "expected_format": expected_format,
            "has_images": False
        })
        
        if self.mode == "mock":
            return self._mock_response(prompt, expected_format)
        elif self.mode == "api":
            return self._api_call(prompt)
        elif self.mode == "qwen":
            return self._qwen_api_call(prompt)
        else:
            raise ValueError(f"未知的模式: {self.mode}")
    
    def call_llm_with_images(
        self, 
        prompt: str, 
        image_paths: List[str], 
        expected_format: str = "json"
    ) -> str:
        """
        调用 LLM（包含图片的多模态输入）
        
        Args:
            prompt: 发送给 LLM 的提示词
            image_paths: 图片文件路径列表
            expected_format: 期望的输出格式，"json" 或 "text"
            
        Returns:
            str: LLM 的响应内容
        """
        self.call_count += 1
        self.call_history.append({
            "call_id": self.call_count,
            "prompt_preview": prompt[:200] + "..." if len(prompt) > 200 else prompt,
            "expected_format": expected_format,
            "has_images": True,
            "image_count": len(image_paths)
        })
        
        if self.mode == "mock":
            return self._mock_response_with_images(prompt, image_paths, expected_format)
        elif self.mode == "api":
            return self._api_call_with_images(prompt, image_paths)
        elif self.mode == "qwen":
            return self._qwen_api_call_with_images(prompt, image_paths)
        else:
            raise ValueError(f"未知的模式: {self.mode}")
    
    def call_llm_with_all_images(
        self,
        prompt: str,
        query_image: str,
        candidate_images: Dict[str, str],
        expected_format: str = "json"
    ) -> str:
        """
        一次性发送所有图像（query + 所有 candidates）进行批量处理
        
        这个方法用于需要同时分析所有图像的场景，如批量描述或批量相似度判断。
        
        Args:
            prompt: 发送给 LLM 的提示词
            query_image: 查询图像路径
            candidate_images: 候选图像字典，格式为 {candidate_id: image_path}
            expected_format: 期望的输出格式
            
        Returns:
            str: LLM 的响应内容
        """
        # 构建图像列表：query 在前，candidates 在后
        all_image_paths = [query_image]
        image_order = ["query"]
        
        for cid, img_path in candidate_images.items():
            all_image_paths.append(img_path)
            image_order.append(cid)
        
        self.call_count += 1
        self.call_history.append({
            "call_id": self.call_count,
            "prompt_preview": prompt[:200] + "..." if len(prompt) > 200 else prompt,
            "expected_format": expected_format,
            "has_images": True,
            "image_count": len(all_image_paths),
            "batch_mode": True,
            "image_order": image_order
        })
        
        if self.mode == "mock":
            return self._mock_batch_response(prompt, query_image, candidate_images, expected_format)
        elif self.mode == "api":
            return self._api_call_with_all_images(prompt, all_image_paths, image_order)
        elif self.mode == "qwen":
            return self._qwen_api_call_with_all_images(prompt, all_image_paths, image_order)
        else:
            raise ValueError(f"未知的模式: {self.mode}")
    
    def _init_client(self):
        """
        初始化 Gemini API 客户端
        """
        if not GENAI_AVAILABLE:
            raise ImportError("google-genai 库未安装，请运行: pip install google-genai")
        
        if not self.api_key:
            raise ValueError("API 模式需要提供 api_key 或设置环境变量 GEMINI_API_KEY")
        
        # 配置 API Key

        
        # 创建客户端
        self.client = genai.Client(
                #   api_key="sk-qVSKal9KNzyij2DQHpKKkvRwEz3ASzzJJrmO6ceI8NOl4AQ0",
                #     http_options={
                #         "base_url": "https://api.yyds168.net",  # 替换为你要使用的网址
                #         "timeout": 999999  # 设置超时时间为 600 秒
                #     },
        )
        print(f"✅ Gemini API 客户端初始化成功，模型: {self.model}")
    
    def _init_qwen_client(self):
        """
        初始化 QWEN API 客户端
        支持两种模式：
        - requests: 使用 requests 库直接调用 HTTP API（推荐用于自部署模型）
        - openai_sdk: 使用 OpenAI SDK 兼容模式调用
        """
        if not self.api_key:
            raise ValueError("QWEN 模式需要提供 api_key 或设置环境变量 QWEN_API_KEY")
        
        if self.qwen_call_mode == "requests":
            # 使用 requests 直接调用
            # token 格式为 secret_id + "@1"
            self.qwen_token = self.api_key + "@1"
            self.qwen_headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.qwen_token}'
            }
            # 标记客户端已初始化（使用 requests 模式）
            self.client = "qwen_requests_mode"
            print(f"✅ QWEN API 客户端初始化成功（requests 模式），模型: {self.model}")
            print(f"   API URL: {self.base_url}")
        else:
            # 使用 OpenAI SDK 兼容模式
            if not OPENAI_AVAILABLE:
                raise ImportError("openai 库未安装，请运行: pip install openai")
            
            # 创建 OpenAI 兼容客户端
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            print(f"✅ QWEN API 客户端初始化成功（OpenAI SDK 模式），模型: {self.model}")
            print(f"   Base URL: {self.base_url}")
    
    def _get_mime_type(self, img_path: str) -> str:
        """
        根据图片路径获取对应的 MIME 类型
        
        Args:
            img_path: 图片文件路径
            
        Returns:
            str: MIME 类型
        """
        mime_type, _ = mimetypes.guess_type(img_path)
        if not mime_type:
            # 兜底处理常见格式
            ext = os.path.splitext(img_path)[1].lower()
            mime_map = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".bmp": "image/bmp",
                ".gif": "image/gif",
                ".webp": "image/webp"
            }
            mime_type = mime_map.get(ext, "image/jpeg")
        return mime_type
    
    def _prepare_image_part(self, img_path: str):
        """
        将图片文件转换为 Gemini API 可接受的 Part 格式
        
        Args:
            img_path: 图片文件路径
            
        Returns:
            types.Part: Gemini 图片 Part 对象
        """
        if not GENAI_AVAILABLE:
            raise ImportError("google-genai 库未安装")
        
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"图片文件不存在: {img_path}")
        
        # 读取图片字节数据
        with open(img_path, "rb") as f:
            img_bytes = f.read()
        
        # 获取 MIME 类型并构造 Part
        mime_type = self._get_mime_type(img_path)
        img_part = types.Part.from_bytes(
            data=img_bytes,
            mime_type=mime_type
        )
        
        return img_part
    
    def _prepare_image_parts(self, image_paths: List[str]) -> List:
        """
        批量将图片文件转换为 Gemini API 可接受的 Part 格式
        优先使用缓存的图像 Part
        
        Args:
            image_paths: 图片文件路径列表
            
        Returns:
            List: Gemini 图片 Part 对象列表
        """
        image_parts = []
        for idx, img_path in enumerate(image_paths, 1):
            try:
                # 优先使用缓存
                if img_path in self._image_parts_cache:
                    image_parts.append(self._image_parts_cache[img_path])
                    print(f"  ✅ 使用缓存图片 {idx}/{len(image_paths)}: {os.path.basename(img_path)}")
                else:
                    img_part = self._prepare_image_part(img_path)
                    self._image_parts_cache[img_path] = img_part  # 加入缓存
                    image_parts.append(img_part)
                    print(f"  ✅ 已准备图片 {idx}/{len(image_paths)}: {os.path.basename(img_path)}")
            except Exception as e:
                print(f"  ❌ 处理图片失败 {idx}/{len(image_paths)}: {img_path}，错误：{e}")
        
        return image_parts
    
    def _api_call(self, prompt: str) -> str:
        """
        真实 API 调用 - 使用 Google Gemini API（纯文本）
        
        Args:
            prompt: 提示词
            
        Returns:
            str: API 响应
        """
        if not GENAI_AVAILABLE:
            raise ImportError("google-genai 库未安装，请运行: pip install google-genai")
        
        if self.client is None:
            self._init_client()
        
        try:
            print(f"\n🚀 正在向 Gemini 发送请求...")
            print(f"   - 模型：{self.model}")
            print(f"   - Prompt 长度：{len(prompt)} 字符")
            
            # 发送请求
            response = self.client.models.generate_content(
                model=self.model,
                contents=[prompt],
                # config=types.GenerateContentConfig(
                #     temperature=0.1,  # 低随机性保证输出稳定
                #     max_output_tokens=8192  # 足够容纳结构化输出
                # )
            )
            
            # 检查响应是否有效
            if response.text:
                print("✅ Gemini 响应成功！")
                # 统计并打印 token 使用情况
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    usage = response.usage_metadata
                    input_tokens = getattr(usage, 'prompt_token_count', 0) or 0
                    output_tokens = getattr(usage, 'candidates_token_count', 0) or 0
                    total_tokens = getattr(usage, 'total_token_count', 0) or (input_tokens + output_tokens)
                    self.total_input_tokens += input_tokens
                    self.total_output_tokens += output_tokens
                    self.token_history.append({
                        "call_id": self.call_count,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": total_tokens
                    })
                    print(f"   - Token 使用: 输入 {input_tokens}, 输出 {output_tokens}, 总计 {total_tokens}")
                return response.text
            else:
                raise ValueError("Gemini 返回空响应")
        
        except Exception as e:
            print(f"❌ Gemini API 调用失败: {e}")
            raise RuntimeError(f"请求 Gemini 失败：{e}")
    
    def _api_call_with_images(self, prompt: str, image_paths: List[str]) -> str:
        """
        真实 API 调用 - 使用 Google Gemini API（包含图片的多模态输入）
        
        Args:
            prompt: 提示词
            image_paths: 图片文件路径列表
            
        Returns:
            str: API 响应
        """
        if not GENAI_AVAILABLE:
            raise ImportError("google-genai 库未安装，请运行: pip install google-genai")
        
        if self.client is None:
            self._init_client()
        
        try:
            print(f"\n🚀 正在向 Gemini 发送多模态请求...")
            print(f"   - 模型：{self.model}")
            print(f"   - Prompt 长度：{len(prompt)} 字符")
            print(f"   - 图片数量：{len(image_paths)}")
            
            # 准备图片 Parts
            print("📷 正在准备图片数据...")
            image_parts = self._prepare_image_parts(image_paths)
            
            if not image_parts:
                raise ValueError("没有成功加载任何图片")
            
            # 构造请求内容（文本 + 图片）
            contents = [prompt] + image_parts
            
            # # 计算 token 数量（可选）
            # try:
            #     total_tokens = self.client.models.count_tokens(
            #         model=self.model, 
            #         contents=contents
            #     )
            #     print(f"   - 预估 Token 数量: {total_tokens}")
            # except Exception as e:
            #     print(f"   - 无法计算 Token 数量: {e}")
            
            # 发送请求
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                # config=types.GenerateContentConfig(
                #     temperature=0.1,  # 低随机性保证输出稳定
                #     max_output_tokens=8192  # 足够容纳结构化输出
                # )
            )
            
            # 检查响应是否有效
            if response.text:
                print("✅ Gemini 多模态响应成功！")
                # 统计并打印 token 使用情况
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    usage = response.usage_metadata
                    input_tokens = getattr(usage, 'prompt_token_count', 0) or 0
                    output_tokens = getattr(usage, 'candidates_token_count', 0) or 0
                    total_tokens = getattr(usage, 'total_token_count', 0) or (input_tokens + output_tokens)
                    self.total_input_tokens += input_tokens
                    self.total_output_tokens += output_tokens
                    self.token_history.append({
                        "call_id": self.call_count,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": total_tokens
                    })
                    print(f"   - Token 使用: 输入 {input_tokens}, 输出 {output_tokens}, 总计 {total_tokens}")
                return response.text
            else:
                raise ValueError("Gemini 返回空响应")
        
        except Exception as e:
            print(f"❌ Gemini 多模态 API 调用失败: {e}")
            raise RuntimeError(f"请求 Gemini 失败：{e}")
    
    def _api_call_with_all_images(
        self, 
        prompt: str, 
        all_image_paths: List[str],
        image_order: List[str]
    ) -> str:
        """
        真实 API 调用 - 一次性发送所有图像
        
        Args:
            prompt: 提示词
            all_image_paths: 所有图像路径列表
            image_order: 图像顺序标识列表
            
        Returns:
            str: API 响应
        """
        if not GENAI_AVAILABLE:
            raise ImportError("google-genai 库未安装，请运行: pip install google-genai")
        
        if self.client is None:
            self._init_client()
        
        try:
            print(f"\n🚀 正在向 Gemini 发送批量多模态请求...")
            print(f"   - 模型：{self.model}")
            print(f"   - Prompt 长度：{len(prompt)} 字符")
            print(f"   - 总图片数量：{len(all_image_paths)}")
            print(f"   - 图片顺序：{image_order}")
            
            # 准备所有图片 Parts（使用缓存）
            print("📷 正在准备所有图片数据...")
            image_parts = self._prepare_image_parts(all_image_paths)
            
            if not image_parts:
                raise ValueError("没有成功加载任何图片")
            
            # 构造请求内容（文本 + 所有图片）
            contents = [prompt] + image_parts
            
            # 计算 token 数量（可选）
            # try:
            #     total_tokens = self.client.models.count_tokens(
            #         model=self.model, 
            #         contents=contents
            #     )
            #     print(f"   - 预估 Token 数量: {total_tokens}")
            # except Exception as e:
            #     print(f"   - 无法计算 Token 数量: {e}")
            
            # 发送请求
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                # config=types.GenerateContentConfig(
                #     temperature=0.1,  # 低随机性保证输出稳定
                #     max_output_tokens=16384  # 批量处理需要更大的输出空间
                # )
            )
            
            # 检查响应是否有效
            if response.text:
                print("✅ Gemini 批量多模态响应成功！")
                # 统计并打印 token 使用情况
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    usage = response.usage_metadata
                    input_tokens = getattr(usage, 'prompt_token_count', 0) or 0
                    output_tokens = getattr(usage, 'candidates_token_count', 0) or 0
                    total_tokens = getattr(usage, 'total_token_count', 0) or (input_tokens + output_tokens)
                    self.total_input_tokens += input_tokens
                    self.total_output_tokens += output_tokens
                    self.token_history.append({
                        "call_id": self.call_count,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": total_tokens
                    })
                    print(f"   - Token 使用: 输入 {input_tokens}, 输出 {output_tokens}, 总计 {total_tokens}")
                return response.text
            else:
                raise ValueError("Gemini 返回空响应")
        
        except Exception as e:
            print(f"❌ Gemini 批量多模态 API 调用失败: {e}")
            raise RuntimeError(f"请求 Gemini 失败：{e}")
    
    # ============ QWEN API 调用方法 ============
    
    def _encode_image_to_base64(self, img_path: str) -> str:
        """
        将图片编码为 base64 字符串
        
        Args:
            img_path: 图片文件路径
            
        Returns:
            str: base64 编码的图片字符串
        """
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"图片文件不存在: {img_path}")
        
        with open(img_path, "rb") as f:
            img_bytes = f.read()
        
        return base64.b64encode(img_bytes).decode('utf-8')
    
    def _get_image_url_for_qwen(self, img_path: str) -> str:
        """
        将图片转换为 QWEN API 可接受的格式（data URL）
        
        Args:
            img_path: 图片文件路径
            
        Returns:
            str: data URL 格式的图片
        """
        mime_type = self._get_mime_type(img_path)
        base64_data = self._encode_image_to_base64(img_path)
        return f"data:{mime_type};base64,{base64_data}"
    
    def _qwen_api_call(self, prompt: str) -> str:
        """
        QWEN API 调用（纯文本）
        支持 requests 直接调用和 OpenAI SDK 两种模式
        
        Args:
            prompt: 提示词
            
        Returns:
            str: API 响应
        """
        if self.client is None:
            self._init_qwen_client()
        
        if self.qwen_call_mode == "requests":
            return self._qwen_api_call_requests(prompt)
        else:
            return self._qwen_api_call_openai_sdk(prompt)
    
    def _qwen_api_call_requests(self, prompt: str) -> str:
        """
        QWEN API 调用（纯文本）- 使用 requests 直接调用
        
        Args:
            prompt: 提示词
            
        Returns:
            str: API 响应
        """
        try:
            print(f"\n🚀 正在向 QWEN 发送请求（requests 模式）...")
            print(f"   - 模型：{self.model}")
            print(f"   - Prompt 长度：{len(prompt)} 字符")
            
            # 构建请求 payload
            payload = {
                'model': self.model,
                'messages': [
                    {'role': 'user', 'content': prompt}
                ],
                'temperature': 0.1,  # 低随机性保证输出稳定
                'max_tokens': 8192
            }
            
            # 发送请求
            response = requests.post(
                self.base_url,
                headers=self.qwen_headers,
                data=json.dumps(payload),
                timeout=300  # 5分钟超时
            )
            
            # 检查 HTTP 状态码
            if response.status_code != 200:
                error_msg = response.json() if response.text else f"HTTP {response.status_code}"
                raise RuntimeError(f"QWEN API 返回错误: {error_msg}")
            
            # 解析响应
            resp_json = response.json()
            
            # 检查响应是否有效
            if 'choices' in resp_json and resp_json['choices'] and resp_json['choices'][0].get('message', {}).get('content'):
                result = resp_json['choices'][0]['message']['content']
                print("✅ QWEN 响应成功！")
                # 统计并打印 token 使用情况
                if 'usage' in resp_json and resp_json['usage']:
                    usage = resp_json['usage']
                    input_tokens = usage.get('prompt_tokens', 0) or 0
                    output_tokens = usage.get('completion_tokens', 0) or 0
                    total_tokens = usage.get('total_tokens', 0) or (input_tokens + output_tokens)
                    self.total_input_tokens += input_tokens
                    self.total_output_tokens += output_tokens
                    self.token_history.append({
                        "call_id": self.call_count,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": total_tokens
                    })
                    print(f"   - Token 使用: 输入 {input_tokens}, 输出 {output_tokens}, 总计 {total_tokens}")
                return result
            else:
                raise ValueError(f"QWEN 返回空响应或格式错误: {resp_json}")
        
        except requests.exceptions.Timeout:
            print(f"❌ QWEN API 调用超时")
            raise RuntimeError("请求 QWEN 超时")
        except requests.exceptions.RequestException as e:
            print(f"❌ QWEN API 请求异常: {e}")
            raise RuntimeError(f"请求 QWEN 失败：{e}")
        except Exception as e:
            print(f"❌ QWEN API 调用失败: {e}")
            raise RuntimeError(f"请求 QWEN 失败：{e}")
    
    def _qwen_api_call_openai_sdk(self, prompt: str) -> str:
        """
        QWEN API 调用（纯文本）- 使用 OpenAI SDK 兼容模式
        
        Args:
            prompt: 提示词
            
        Returns:
            str: API 响应
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("openai 库未安装，请运行: pip install openai")
        
        try:
            print(f"\n🚀 正在向 QWEN 发送请求（OpenAI SDK 模式）...")
            print(f"   - 模型：{self.model}")
            print(f"   - Prompt 长度：{len(prompt)} 字符")
            
            # 发送请求
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # 低随机性保证输出稳定
                max_tokens=8192,
                extra_body={"enable_thinking": True}
            )
            
            # 检查响应是否有效
            if response.choices and response.choices[0].message.content:
                result = response.choices[0].message.content
                print("✅ QWEN 响应成功！")
                # 统计并打印 token 使用情况
                if hasattr(response, 'usage') and response.usage:
                    input_tokens = response.usage.prompt_tokens or 0
                    output_tokens = response.usage.completion_tokens or 0
                    total_tokens = response.usage.total_tokens or (input_tokens + output_tokens)
                    self.total_input_tokens += input_tokens
                    self.total_output_tokens += output_tokens
                    self.token_history.append({
                        "call_id": self.call_count,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": total_tokens
                    })
                    print(f"   - Token 使用: 输入 {input_tokens}, 输出 {output_tokens}, 总计 {total_tokens}")
                return result
            else:
                raise ValueError("QWEN 返回空响应")
        
        except Exception as e:
            print(f"❌ QWEN API 调用失败: {e}")
            raise RuntimeError(f"请求 QWEN 失败：{e}")
    
    def _qwen_api_call_with_images(self, prompt: str, image_paths: List[str]) -> str:
        """
        QWEN API 调用（包含图片的多模态输入）
        支持 requests 直接调用和 OpenAI SDK 两种模式
        
        Args:
            prompt: 提示词
            image_paths: 图片文件路径列表
            
        Returns:
            str: API 响应
        """
        if self.client is None:
            self._init_qwen_client()
        
        if self.qwen_call_mode == "requests":
            return self._qwen_api_call_with_images_requests(prompt, image_paths)
        else:
            return self._qwen_api_call_with_images_openai_sdk(prompt, image_paths)
    
    def _qwen_api_call_with_images_requests(self, prompt: str, image_paths: List[str]) -> str:
        """
        QWEN API 调用（包含图片的多模态输入）- 使用 requests 直接调用
        
        Args:
            prompt: 提示词
            image_paths: 图片文件路径列表
            
        Returns:
            str: API 响应
        """
        try:
            print(f"\n🚀 正在向 QWEN 发送多模态请求（requests 模式）...")
            print(f"   - 模型：{self.model}")
            print(f"   - Prompt 长度：{len(prompt)} 字符")
            print(f"   - 图片数量：{len(image_paths)}")
            
            # 构建消息内容
            content = []
            
            # 添加文本
            content.append({"type": "text", "text": prompt})
            
            # 添加图片
            print("📷 正在准备图片数据...")
            for idx, img_path in enumerate(image_paths, 1):
                try:
                    image_url = self._get_image_url_for_qwen(img_path)
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": image_url}
                    })
                    print(f"  ✅ 已准备图片 {idx}/{len(image_paths)}: {os.path.basename(img_path)}")
                except Exception as e:
                    print(f"  ❌ 处理图片失败 {idx}/{len(image_paths)}: {img_path}，错误：{e}")
            
            # 构建请求 payload
            payload = {
                'model': self.model,
                'messages': [
                    {'role': 'user', 'content': content}
                ],
                'temperature': 0.1,
                'max_tokens': 8192
            }
            
            # 发送请求
            response = requests.post(
                self.base_url,
                headers=self.qwen_headers,
                data=json.dumps(payload),
                timeout=300  # 5分钟超时
            )
            
            # 检查 HTTP 状态码
            if response.status_code != 200:
                error_msg = response.json() if response.text else f"HTTP {response.status_code}"
                raise RuntimeError(f"QWEN API 返回错误: {error_msg}")
            
            # 解析响应
            resp_json = response.json()
            
            # 检查响应是否有效
            if 'choices' in resp_json and resp_json['choices'] and resp_json['choices'][0].get('message', {}).get('content'):
                result = resp_json['choices'][0]['message']['content']
                print("✅ QWEN 多模态响应成功！")
                # 统计并打印 token 使用情况
                if 'usage' in resp_json and resp_json['usage']:
                    usage = resp_json['usage']
                    input_tokens = usage.get('prompt_tokens', 0) or 0
                    output_tokens = usage.get('completion_tokens', 0) or 0
                    total_tokens = usage.get('total_tokens', 0) or (input_tokens + output_tokens)
                    self.total_input_tokens += input_tokens
                    self.total_output_tokens += output_tokens
                    self.token_history.append({
                        "call_id": self.call_count,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": total_tokens
                    })
                    print(f"   - Token 使用: 输入 {input_tokens}, 输出 {output_tokens}, 总计 {total_tokens}")
                return result
            else:
                raise ValueError(f"QWEN 返回空响应或格式错误: {resp_json}")
        
        except requests.exceptions.Timeout:
            print(f"❌ QWEN 多模态 API 调用超时")
            raise RuntimeError("请求 QWEN 超时")
        except requests.exceptions.RequestException as e:
            print(f"❌ QWEN 多模态 API 请求异常: {e}")
            raise RuntimeError(f"请求 QWEN 失败：{e}")
        except Exception as e:
            print(f"❌ QWEN 多模态 API 调用失败: {e}")
            raise RuntimeError(f"请求 QWEN 失败：{e}")
    
    def _qwen_api_call_with_images_openai_sdk(self, prompt: str, image_paths: List[str]) -> str:
        """
        QWEN API 调用（包含图片的多模态输入）- 使用 OpenAI SDK 兼容模式
        
        Args:
            prompt: 提示词
            image_paths: 图片文件路径列表
            
        Returns:
            str: API 响应
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("openai 库未安装，请运行: pip install openai")
        
        try:
            print(f"\n🚀 正在向 QWEN 发送多模态请求（OpenAI SDK 模式）...")
            print(f"   - 模型：{self.model}")
            print(f"   - Prompt 长度：{len(prompt)} 字符")
            print(f"   - 图片数量：{len(image_paths)}")
            
            # 构建消息内容
            content = []
            
            # 添加文本
            content.append({"type": "text", "text": prompt})
            
            # 添加图片
            print("📷 正在准备图片数据...")
            for idx, img_path in enumerate(image_paths, 1):
                try:
                    image_url = self._get_image_url_for_qwen(img_path)
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": image_url}
                    })
                    print(f"  ✅ 已准备图片 {idx}/{len(image_paths)}: {os.path.basename(img_path)}")
                except Exception as e:
                    print(f"  ❌ 处理图片失败 {idx}/{len(image_paths)}: {img_path}，错误：{e}")
            
            # 发送请求
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": content}
                ],
                temperature=0.1,
                max_tokens=8192
            )
            
            # 检查响应是否有效
            if response.choices and response.choices[0].message.content:
                result = response.choices[0].message.content
                print("✅ QWEN 多模态响应成功！")
                # 统计并打印 token 使用情况
                if hasattr(response, 'usage') and response.usage:
                    input_tokens = response.usage.prompt_tokens or 0
                    output_tokens = response.usage.completion_tokens or 0
                    total_tokens = response.usage.total_tokens or (input_tokens + output_tokens)
                    self.total_input_tokens += input_tokens
                    self.total_output_tokens += output_tokens
                    self.token_history.append({
                        "call_id": self.call_count,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": total_tokens
                    })
                    print(f"   - Token 使用: 输入 {input_tokens}, 输出 {output_tokens}, 总计 {total_tokens}")
                return result
            else:
                raise ValueError("QWEN 返回空响应")
        
        except Exception as e:
            print(f"❌ QWEN 多模态 API 调用失败: {e}")
            raise RuntimeError(f"请求 QWEN 失败：{e}")
    
    def _qwen_api_call_with_all_images(
        self, 
        prompt: str, 
        all_image_paths: List[str],
        image_order: List[str]
    ) -> str:
        """
        QWEN API 调用 - 一次性发送所有图像
        支持 requests 直接调用和 OpenAI SDK 两种模式
        
        Args:
            prompt: 提示词
            all_image_paths: 所有图像路径列表
            image_order: 图像顺序标识列表
            
        Returns:
            str: API 响应
        """
        if self.client is None:
            self._init_qwen_client()
        
        if self.qwen_call_mode == "requests":
            return self._qwen_api_call_with_all_images_requests(prompt, all_image_paths, image_order)
        else:
            return self._qwen_api_call_with_all_images_openai_sdk(prompt, all_image_paths, image_order)
    
    def _qwen_api_call_with_all_images_requests(
        self, 
        prompt: str, 
        all_image_paths: List[str],
        image_order: List[str]
    ) -> str:
        """
        QWEN API 调用 - 一次性发送所有图像 - 使用 requests 直接调用
        
        Args:
            prompt: 提示词
            all_image_paths: 所有图像路径列表
            image_order: 图像顺序标识列表
            
        Returns:
            str: API 响应
        """
        try:
            print(f"\n🚀 正在向 QWEN 发送批量多模态请求（requests 模式）...")
            print(f"   - 模型：{self.model}")
            print(f"   - Prompt 长度：{len(prompt)} 字符")
            print(f"   - 总图片数量：{len(all_image_paths)}")
            print(f"   - 图片顺序：{image_order}")
            
            # 构建消息内容
            content = []
            
            # 添加文本
            content.append({"type": "text", "text": prompt})
            
            # 添加所有图片
            print("📷 正在准备所有图片数据...")
            for idx, img_path in enumerate(all_image_paths, 1):
                try:
                    image_url = self._get_image_url_for_qwen(img_path)
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": image_url}
                    })
                    print(f"  ✅ 已准备图片 {idx}/{len(all_image_paths)}: {os.path.basename(img_path)}")
                except Exception as e:
                    print(f"  ❌ 处理图片失败 {idx}/{len(all_image_paths)}: {img_path}，错误：{e}")
            
            # 构建请求 payload
            payload = {
                'model': self.model,
                'messages': [
                    {'role': 'user', 'content': content}
                ],
                'temperature': 0.1,
                'max_tokens': 16384  # 批量处理需要更大的输出空间
            }
            
            # 发送请求
            response = requests.post(
                self.base_url,
                headers=self.qwen_headers,
                data=json.dumps(payload),
                timeout=600  # 10分钟超时（批量处理需要更长时间）
            )
            
            # 检查 HTTP 状态码
            if response.status_code != 200:
                error_msg = response.json() if response.text else f"HTTP {response.status_code}"
                raise RuntimeError(f"QWEN API 返回错误: {error_msg}")
            
            # 解析响应
            resp_json = response.json()
            
            # 检查响应是否有效
            if 'choices' in resp_json and resp_json['choices'] and resp_json['choices'][0].get('message', {}).get('content'):
                result = resp_json['choices'][0]['message']['content']
                print("✅ QWEN 批量多模态响应成功！")
                # 统计并打印 token 使用情况
                if 'usage' in resp_json and resp_json['usage']:
                    usage = resp_json['usage']
                    input_tokens = usage.get('prompt_tokens', 0) or 0
                    output_tokens = usage.get('completion_tokens', 0) or 0
                    total_tokens = usage.get('total_tokens', 0) or (input_tokens + output_tokens)
                    self.total_input_tokens += input_tokens
                    self.total_output_tokens += output_tokens
                    self.token_history.append({
                        "call_id": self.call_count,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": total_tokens
                    })
                    print(f"   - Token 使用: 输入 {input_tokens}, 输出 {output_tokens}, 总计 {total_tokens}")
                return result
            else:
                raise ValueError(f"QWEN 返回空响应或格式错误: {resp_json}")
        
        except requests.exceptions.Timeout:
            print(f"❌ QWEN 批量多模态 API 调用超时")
            raise RuntimeError("请求 QWEN 超时")
        except requests.exceptions.RequestException as e:
            print(f"❌ QWEN 批量多模态 API 请求异常: {e}")
            raise RuntimeError(f"请求 QWEN 失败：{e}")
        except Exception as e:
            print(f"❌ QWEN 批量多模态 API 调用失败: {e}")
            raise RuntimeError(f"请求 QWEN 失败：{e}")
    
    def _qwen_api_call_with_all_images_openai_sdk(
        self, 
        prompt: str, 
        all_image_paths: List[str],
        image_order: List[str]
    ) -> str:
        """
        QWEN API 调用 - 一次性发送所有图像 - 使用 OpenAI SDK 兼容模式
        
        Args:
            prompt: 提示词
            all_image_paths: 所有图像路径列表
            image_order: 图像顺序标识列表
            
        Returns:
            str: API 响应
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("openai 库未安装，请运行: pip install openai")
        
        try:
            print(f"\n🚀 正在向 QWEN 发送批量多模态请求（OpenAI SDK 模式）...")
            print(f"   - 模型：{self.model}")
            print(f"   - Prompt 长度：{len(prompt)} 字符")
            print(f"   - 总图片数量：{len(all_image_paths)}")
            print(f"   - 图片顺序：{image_order}")
            
            # 构建消息内容
            content = []
            
            # 添加文本
            content.append({"type": "text", "text": prompt})
            
            # 添加所有图片
            print("📷 正在准备所有图片数据...")
            for idx, img_path in enumerate(all_image_paths, 1):
                try:
                    image_url = self._get_image_url_for_qwen(img_path)
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": image_url}
                    })
                    print(f"  ✅ 已准备图片 {idx}/{len(all_image_paths)}: {os.path.basename(img_path)}")
                except Exception as e:
                    print(f"  ❌ 处理图片失败 {idx}/{len(all_image_paths)}: {img_path}，错误：{e}")
            
            # 发送请求
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": content}
                ],
                temperature=0.1,
                max_tokens=16384  # 批量处理需要更大的输出空间
            )
            
            # 检查响应是否有效
            if response.choices and response.choices[0].message.content:
                result = response.choices[0].message.content
                print("✅ QWEN 批量多模态响应成功！")
                # 统计并打印 token 使用情况
                if hasattr(response, 'usage') and response.usage:
                    input_tokens = response.usage.prompt_tokens or 0
                    output_tokens = response.usage.completion_tokens or 0
                    total_tokens = response.usage.total_tokens or (input_tokens + output_tokens)
                    self.total_input_tokens += input_tokens
                    self.total_output_tokens += output_tokens
                    self.token_history.append({
                        "call_id": self.call_count,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": total_tokens
                    })
                    print(f"   - Token 使用: 输入 {input_tokens}, 输出 {output_tokens}, 总计 {total_tokens}")
                return result
            else:
                raise ValueError("QWEN 返回空响应")
        
        except Exception as e:
            print(f"❌ QWEN 批量多模态 API 调用失败: {e}")
            raise RuntimeError(f"请求 QWEN 失败：{e}")
    
    def _mock_response(self, prompt: str, expected_format: str) -> str:
        """
        生成 mock 响应（纯文本）
        
        根据 prompt 内容智能生成模拟的 LLM 响应。
        
        Args:
            prompt: 提示词
            expected_format: 期望格式
            
        Returns:
            str: 模拟响应
        """
        prompt_lower = prompt.lower()
        
        # 根据不同的 skill 类型生成对应的 mock 响应
        if "维度规划" in prompt or "dimensionplanner" in prompt_lower or "dimension planning" in prompt_lower or "维度银行" in prompt:
            return self._mock_dimension_planner_response(prompt)
        elif "相似度评判" in prompt or "相似度判断" in prompt or "judgeskill" in prompt_lower or ("评估" in prompt and "相似度" in prompt):
            return self._mock_judge_response(prompt)
        elif "排序验证" in prompt or "一致性校验" in prompt or "validateskill" in prompt_lower or "加权总分" in prompt:
            return self._mock_validate_response(prompt)
        elif "物品描述专家" in prompt or "候选描述" in prompt or "descriptorskill" in prompt_lower or ("描述" in prompt and "候选物品标识" in prompt):
            return self._mock_descriptor_response(prompt)
        else:
            # 默认返回一个通用的 JSON 响应
            return json.dumps({"status": "success", "message": "Mock response generated"}, ensure_ascii=False)
    
    def _mock_response_with_images(
        self, 
        prompt: str, 
        image_paths: List[str], 
        expected_format: str
    ) -> str:
        """
        生成 mock 响应（包含图片）
        
        Args:
            prompt: 提示词
            image_paths: 图片文件路径列表
            expected_format: 期望格式
            
        Returns:
            str: 模拟响应
        """
        prompt_lower = prompt.lower()
        
        # 根据图片生成不同的 mock 响应
        if "描述" in prompt or "descriptor" in prompt_lower:
            return self._mock_image_descriptor_response(prompt, image_paths)
        elif "相似度" in prompt or "judge" in prompt_lower or "评判" in prompt:
            return self._mock_image_judge_response(prompt, image_paths)
        else:
            # 默认返回图片描述
            return self._mock_image_descriptor_response(prompt, image_paths)
    
    def _mock_batch_response(
        self,
        prompt: str,
        query_image: str,
        candidate_images: Dict[str, str],
        expected_format: str
    ) -> str:
        """
        生成批量处理的 mock 响应
        
        Args:
            prompt: 提示词
            query_image: 查询图像路径
            candidate_images: 候选图像字典
            expected_format: 期望格式
            
        Returns:
            str: 模拟响应
        """
        prompt_lower = prompt.lower()
        
        # 根据 prompt 内容判断任务类型
        if "描述" in prompt or "descriptor" in prompt_lower:
            # 批量描述任务
            return self._mock_batch_descriptor_response(prompt, query_image, candidate_images)
        elif "相似度" in prompt or "judge" in prompt_lower or "评判" in prompt:
            # 批量相似度判断任务
            return self._mock_batch_judge_response(prompt, query_image, candidate_images)
        else:
            return self._mock_batch_descriptor_response(prompt, query_image, candidate_images)
    
    def _mock_batch_descriptor_response(
        self,
        prompt: str,
        query_image: str,
        candidate_images: Dict[str, str]
    ) -> str:
        """生成批量描述的 mock 响应"""
        # 提取维度名称
        dimension_names = re.findall(r'\*\*([a-zA-Z_][a-zA-Z0-9_]*)\*\*', prompt)
        dimension_names = [d for d in dimension_names if '_' in d]
        
        if not dimension_names:
            dimension_names = [
                "overall_shape", "primary_function", "usage_scene_match",
                "material_texture_quality", "user_interaction_ergonomics"
            ]
        
        # 去重
        dimension_names = list(dict.fromkeys(dimension_names))
        
        # 为每个候选生成描述
        all_descriptions = {}
        for candidate_id, img_path in candidate_images.items():
            image_filename = os.path.basename(img_path)
            descriptions = {}
            for dim_name in dimension_names:
                descriptions[dim_name] = f"基于图片 {image_filename} 分析：该物体在 {dim_name} 维度上展现出独特的特征，整体表现良好。"
            all_descriptions[candidate_id] = {
                "candidate_id": candidate_id,
                "descriptions": descriptions
            }
        
        response = {
            "batch_descriptions": all_descriptions
        }
        return json.dumps(response, ensure_ascii=False, indent=2)
    
    def _mock_batch_judge_response(
        self,
        prompt: str,
        query_image: str,
        candidate_images: Dict[str, str]
    ) -> str:
        """生成批量相似度判断的 mock 响应"""
        # 提取维度名称
        dimension_names = re.findall(r'\*\*([a-zA-Z_][a-zA-Z0-9_]*)\*\*', prompt)
        dimension_names = [d for d in dimension_names if '_' in d]
        
        if not dimension_names:
            dimension_names = [
                "overall_shape", "primary_function", "usage_scene_match",
                "material_texture_quality", "user_interaction_ergonomics"
            ]
        
        # 去重
        dimension_names = list(dict.fromkeys(dimension_names))
        
        # 为每个候选生成评分
        all_scores = {}
        score_variations = [0.05, 0.10, -0.05, 0.0, -0.08, 0.03, -0.02, 0.07]
        
        for idx, (candidate_id, img_path) in enumerate(candidate_images.items()):
            base_score = hash(img_path) % 100 / 100.0
            base_score = max(0.3, min(0.95, base_score + 0.3))
            
            scores = {}
            for i, dim_name in enumerate(dimension_names):
                variation = score_variations[(i + idx) % len(score_variations)]
                score = round(min(1.0, max(0.0, base_score + variation)), 2)
                scores[dim_name] = {
                    "score": score,
                    "reason": f"基于图片视觉分析，候选物品 {candidate_id} 在 {dim_name} 维度上与查询目标具有{self._get_similarity_desc(score)}相似性。"
                }
            
            all_scores[candidate_id] = {
                "candidate_id": candidate_id,
                "scores": scores
            }
        
        response = {
            "batch_scores": all_scores
        }
        return json.dumps(response, ensure_ascii=False, indent=2)
    
    def _mock_image_descriptor_response(self, prompt: str, image_paths: List[str]) -> str:
        """生成基于图片的候选描述 mock 响应"""
        # 从 prompt 中提取 candidate_id
        candidate_id = "unknown"
        match = re.search(r'候选物品 ID[：:]?\s*([^\s\n]+)', prompt)
        if match:
            candidate_id = match.group(1).strip()
        
        # 从图片路径生成描述
        image_filename = os.path.basename(image_paths[0]) if image_paths else "unknown.jpg"
        
        response = {
            "candidate_id": candidate_id,
            "descriptions": {
                "overall_shape": f"基于图片 {image_filename} 分析：该物体呈现规则的几何形状，整体轮廓清晰，比例协调。外形设计具有良好的对称性。",
                "primary_function": f"基于图片 {image_filename} 分析：根据物体的结构特征，其主要功能是提供特定的实用价值，设计符合其功能定位。",
                "usage_scene_match": f"基于图片 {image_filename} 分析：该物品适合在室内日常环境中使用，外观风格与一般家用/办公场景相匹配。",
                "material_texture_quality": f"基于图片 {image_filename} 分析：从图像可见材质表面处理精细，具有良好的质感，材质看起来耐用可靠。",
                "user_interaction_ergonomics": f"基于图片 {image_filename} 分析：物体的设计考虑了人体工程学因素，便于用户进行操作和交互。"
            }
        }
        return json.dumps(response, ensure_ascii=False, indent=2)
    
    def _mock_image_judge_response(self, prompt: str, image_paths: List[str]) -> str:
        """生成基于图片的相似度判断 mock 响应"""
        candidate_id = "unknown"
        match = re.search(r'候选物品 ID[：:]?\s*([^\s\n]+)', prompt)
        if match:
            candidate_id = match.group(1).strip()
        
        # 根据图片文件名生成差异化分数
        if image_paths:
            base_score = hash(image_paths[0]) % 100 / 100.0
            base_score = max(0.3, min(0.95, base_score + 0.3))
        else:
            base_score = 0.6
        
        # 从 prompt 中提取维度名称
        dimension_names = re.findall(r'\*\*([a-zA-Z_][a-zA-Z0-9_]*)\*\*', prompt)
        dimension_names = [d for d in dimension_names if '_' in d]
        
        if not dimension_names:
            dimension_names = [
                "overall_shape", "primary_function", "usage_scene_match",
                "material_texture_quality", "user_interaction_ergonomics"
            ]
        
        # 去重
        dimension_names = list(dict.fromkeys(dimension_names))
        
        scores = {}
        score_variations = [0.05, 0.10, -0.05, 0.0, -0.08, 0.03, -0.02, 0.07]
        
        for i, dim_name in enumerate(dimension_names):
            variation = score_variations[i % len(score_variations)]
            score = round(min(1.0, max(0.0, base_score + variation)), 2)
            scores[dim_name] = {
                "score": score,
                "reason": f"基于图片视觉分析，候选物品在 {dim_name} 维度上与查询目标具有{self._get_similarity_desc(score)}相似性。"
            }
        
        response = {
            "candidate_id": candidate_id,
            "scores": scores
        }
        return json.dumps(response, ensure_ascii=False, indent=2)
    
    def _mock_dimension_planner_response(self, prompt: str) -> str:
        """生成维度规划的 mock 响应"""
        # 尝试从 prompt 中提取 query 信息
        query_context = "通用查询"
        if "query" in prompt.lower():
            # 简单提取一些上下文
            query_context = "特定场景查询"
        
        response = {
            "inferred_scenario": "物品替代与相似性匹配",
            "scenario_reasoning": "根据查询内容分析，用户希望找到与目标物品相似的替代品，主要关注功能匹配和视觉相似性。",
            "dimensions": [
                {
                    "name": "overall_shape",
                    "description": "物体的整体几何外形特征，包括轮廓、比例、尺寸等视觉特征",
                    "weight": 0.25,
                    "scoring_criteria": "评估候选物品与查询目标在整体形状上的相似程度。完全匹配得1分，形状相近得0.7-0.9分，部分相似得0.4-0.6分，差异明显得0.1-0.3分",
                    "source": "bank"
                },
                {
                    "name": "primary_function",
                    "description": "物体的主要功能与核心用途，决定其基本使用价值",
                    "weight": 0.30,
                    "scoring_criteria": "评估候选物品是否能满足查询目标的主要功能需求。功能完全匹配得1分，功能基本等效得0.7-0.9分，功能部分重叠得0.4-0.6分，功能差异大得0.1-0.3分",
                    "source": "bank"
                },
                {
                    "name": "usage_scene_match",
                    "description": "物体是否适合目标使用场景，包括环境适应性和使用情境匹配度",
                    "weight": 0.20,
                    "scoring_criteria": "评估候选物品在目标使用场景下的适用性。完全适用得1分，大部分场景适用得0.7-0.9分，部分场景适用得0.4-0.6分，场景适配性差得0.1-0.3分",
                    "source": "bank"
                },
                {
                    "name": "material_texture_quality",
                    "description": "物体的材质特征、表面质感和整体品质感",
                    "weight": 0.15,
                    "scoring_criteria": "评估候选物品与查询目标在材质和品质上的相似度。材质完全一致得1分，材质相近得0.7-0.9分，材质有差异但不影响使用得0.4-0.6分，材质差异明显得0.1-0.3分",
                    "source": "bank"
                },
                {
                    "name": "user_interaction_ergonomics",
                    "description": "用户与物品交互时的人体工程学特征，包括握持感、操作便利性等",
                    "weight": 0.10,
                    "scoring_criteria": "评估候选物品在用户交互体验上与查询目标的相似度。交互体验完全一致得1分，交互体验相近得0.7-0.9分，交互体验有差异得0.4-0.6分，交互体验差异大得0.1-0.3分",
                    "source": "invented"
                }
            ]
        }
        return json.dumps(response, ensure_ascii=False, indent=2)
    
    def _mock_descriptor_response(self, prompt: str) -> str:
        """生成候选描述的 mock 响应"""
        # 尝试从 prompt 中提取 candidate_id
        candidate_id = "unknown"
        if "candidate_id" in prompt:
            # 简单提取
            match = re.search(r'candidate_id["\s:]+([^\s",]+)', prompt)
            if match:
                candidate_id = match.group(1).strip('"\'')
        
        response = {
            "candidate_id": candidate_id,
            "descriptions": {
                "overall_shape": f"候选物品 {candidate_id} 呈现规则的几何形状，整体轮廓清晰，比例协调，具有对称性特征。尺寸适中，便于单手握持和操作。",
                "primary_function": f"候选物品 {candidate_id} 的主要功能是提供特定的实用价值，能够有效完成其设计目标任务。功能实现方式直观，操作简便。",
                "usage_scene_match": f"候选物品 {candidate_id} 适合在室内环境中使用，能够适应日常使用场景的需求。环境适应性良好，使用限制较少。",
                "material_texture_quality": f"候选物品 {candidate_id} 采用优质材料制成，表面处理精细，质感良好。材质耐用，具有适当的重量感。",
                "user_interaction_ergonomics": f"候选物品 {candidate_id} 的设计符合人体工程学原则，握持舒适，操作时手感良好。用户交互体验流畅自然。"
            }
        }
        return json.dumps(response, ensure_ascii=False, indent=2)
    
    def _mock_judge_response(self, prompt: str) -> str:
        """生成相似度判断的 mock 响应"""
        # 从 prompt 中提取 candidate_id 以生成差异化的分数
        candidate_id = "unknown"
        if "candidate_id" in prompt:
            match = re.search(r'候选物品 ID[：:]?\s*(\S+)', prompt)
            if match:
                candidate_id = match.group(1).strip('"\'')
            else:
                match = re.search(r'candidate_id["\s:]+([^\s",]+)', prompt)
                if match:
                    candidate_id = match.group(1).strip('"\'')
        
        # 从 prompt 中提取维度名称
        dimension_names = []
        # 匹配 **维度名称**（权重 格式
        dim_matches = re.findall(r'\*\*([a-zA-Z_][a-zA-Z0-9_]*)\*\*[（(]', prompt)
        if not dim_matches:
            # 尝试匹配 **维度名称** 格式（更宽松）
            dim_matches = re.findall(r'\*\*([a-zA-Z_][a-zA-Z0-9_]*)\*\*', prompt)
            if dim_matches:
                # 过滤出合法的维度名称（包含下划线的通常是维度名）
                dim_matches = [d for d in dim_matches if '_' in d]
        
        if dim_matches:
            dimension_names = dim_matches
        
        # 如果没有找到维度名称，使用默认的
        if not dimension_names:
            dimension_names = [
                "overall_shape",
                "primary_function",
                "usage_scene_match",
                "material_texture_quality",
                "user_interaction_ergonomics"
            ]
        
        # 去重并保持顺序
        seen = set()
        unique_dims = []
        for d in dimension_names:
            if d not in seen:
                seen.add(d)
                unique_dims.append(d)
        dimension_names = unique_dims
        
        # 根据 candidate_id 生成不同的分数（模拟真实场景）
        base_score = hash(candidate_id) % 100 / 100.0
        base_score = max(0.3, min(0.95, base_score + 0.3))  # 确保分数在合理范围内
        
        # 为每个维度生成评分
        scores = {}
        score_variations = [0.05, 0.10, -0.05, 0.0, -0.08, 0.03, -0.02, 0.07]
        
        for i, dim_name in enumerate(dimension_names):
            variation = score_variations[i % len(score_variations)]
            score = round(min(1.0, max(0.0, base_score + variation)), 2)
            scores[dim_name] = {
                "score": score,
                "reason": f"候选物品 {candidate_id} 在 {dim_name} 维度上与查询目标具有{self._get_similarity_desc(score)}相似性。"
            }
        
        response = {
            "candidate_id": candidate_id,
            "scores": scores
        }
        return json.dumps(response, ensure_ascii=False, indent=2)
    
    def _get_similarity_desc(self, score: float) -> str:
        """根据分数返回相似度描述"""
        if score >= 0.8:
            return "较高的"
        elif score >= 0.6:
            return "中等的"
        elif score >= 0.4:
            return "一定的"
        else:
            return "较低的"
    
    def _mock_validate_response(self, prompt: str) -> str:
        """生成验证排序的 mock 响应"""
        # 尝试从 prompt 中提取候选 ID 列表
        candidate_ids = []
        # 简单匹配提取候选 ID
        matches = re.findall(r'candidate_[a-zA-Z0-9_]+', prompt)
        if matches:
            candidate_ids = list(set(matches))
        
        if not candidate_ids:
            candidate_ids = ["candidate_001", "candidate_002", "candidate_003"]
        
        # 按照一定规则排序（模拟验证后的排序）
        candidate_ids.sort()
        
        response = {
            "initial_ranking": candidate_ids.copy(),
            "validation_checks": [
                {
                    "check_type": "monotonicity",
                    "status": "passed",
                    "description": "相似度分数整体呈递减趋势，排序符合预期"
                },
                {
                    "check_type": "semantic_consistency",
                    "status": "passed",
                    "description": "相邻候选物品之间不存在明显的语义反转"
                },
                {
                    "check_type": "score_gap_analysis",
                    "status": "passed",
                    "description": "相邻排名之间的分数差距合理，无异常跳跃"
                }
            ],
            "adjustments_made": [],
            "final_ranking": candidate_ids,
            "validation_notes": "经过一致性校验，初始排序结果合理有效，未发现需要调整的问题。所有候选物品的排序符合相似度递减的预期，语义一致性良好。",
            "confidence_score": 0.92
        }
        return json.dumps(response, ensure_ascii=False, indent=2)
    
    def get_call_statistics(self) -> dict:
        """
        获取调用统计信息
        
        Returns:
            dict: 调用统计
        """
        cache_stats = self.result_cache.get_stats()
        return {
            "total_calls": self.call_count,
            "mode": self.mode,
            "history_count": len(self.call_history),
            "cache_stats": cache_stats,
            "preloaded_images": len(self._image_parts_cache),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "token_history": self.token_history
        }
    
    def get_token_statistics(self) -> dict:
        """
        获取 Token 使用统计信息
        
        Returns:
            dict: Token 统计信息
        """
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "call_count": len(self.token_history),
            "avg_input_tokens": self.total_input_tokens / len(self.token_history) if self.token_history else 0,
            "avg_output_tokens": self.total_output_tokens / len(self.token_history) if self.token_history else 0,
            "token_history": self.token_history
        }
    
    def reset_statistics(self):
        """重置统计信息"""
        self.call_count = 0
        self.call_history = []
        self.result_cache.clear()
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.token_history = []
    
    def clear_image_cache(self):
        """清空图像缓存"""
        self._image_parts_cache.clear()


# 全局 LLM 接口实例
_llm_instance = None


def get_llm_interface(
    mode: str = "mock", 
    api_key: Optional[str] = None,
    model: str = None,
    force_new: bool = False,
    qwen_call_mode: str = None
) -> LLMInterface:
    """
    获取 LLM 接口单例
    
    Args:
        mode: 运行模式，"mock"、"api"（Gemini）或 "qwen"
        api_key: API 密钥（也可通过环境变量设置）
            - Gemini: 环境变量 GEMINI_API_KEY
            - QWEN: 环境变量 QWEN_API_KEY
        model: 模型名称（可选，不指定则使用各模式的默认模型）
            - Gemini: 默认 gemini-3-flash-preview
            - QWEN: 默认 qwen-vl-max
        force_new: 是否强制创建新实例（用于切换模式）
        qwen_call_mode: QWEN 调用模式（仅在 mode="qwen" 时有效）
            - "requests": 使用 requests 库直接调用 HTTP API（推荐用于自部署模型）
            - "openai_sdk": 使用 OpenAI SDK 兼容模式调用
            - 默认使用配置文件中的 QWEN_CALL_MODE 值
        
    Returns:
        LLMInterface: LLM 接口实例
    """
    global _llm_instance
    if _llm_instance is None or force_new:
        _llm_instance = LLMInterface(mode=mode, api_key=api_key, model=model, qwen_call_mode=qwen_call_mode)
    return _llm_instance


def get_result_cache() -> ResultCache:
    """
    获取结果缓存实例
    
    Returns:
        ResultCache: 结果缓存实例
    """
    llm = get_llm_interface()
    return llm.result_cache


def call_llm(prompt: str, expected_format: str = "json") -> str:
    """
    便捷的 LLM 调用函数（纯文本）
    
    Args:
        prompt: 提示词
        expected_format: 期望格式
        
    Returns:
        str: LLM 响应
    """
    llm = get_llm_interface()
    return llm.call_llm(prompt, expected_format)


def call_llm_with_images(
    prompt: str, 
    image_paths: List[str], 
    expected_format: str = "json"
) -> str:
    """
    便捷的 LLM 调用函数（包含图片）
    
    Args:
        prompt: 提示词
        image_paths: 图片文件路径列表
        expected_format: 期望格式
        
    Returns:
        str: LLM 响应
    """
    llm = get_llm_interface()
    return llm.call_llm_with_images(prompt, image_paths, expected_format)


def call_llm_with_all_images(
    prompt: str,
    query_image: str,
    candidate_images: Dict[str, str],
    expected_format: str = "json"
) -> str:
    """
    便捷的 LLM 调用函数（一次性发送所有图片）
    
    Args:
        prompt: 提示词
        query_image: 查询图像路径
        candidate_images: 候选图像字典
        expected_format: 期望格式
        
    Returns:
        str: LLM 响应
    """
    llm = get_llm_interface()
    return llm.call_llm_with_all_images(prompt, query_image, candidate_images, expected_format)
