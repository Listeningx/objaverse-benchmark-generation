"""
直接执行可视化脚本 - 基于 image_ranking_result.json
"""

import os
import sys
import glob
from datetime import datetime

# 添加当前目录到路径
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from visualize_ranking import visualize_from_json, generate_unique_filename

def run_visualization(json_file: str = None):
    """
    执行可视化
    
    Args:
        json_file: 指定要可视化的 JSON 文件路径（可选，默认使用最新的排序结果）
    """
    print("=" * 60)
    print("排序结果可视化")
    print("=" * 60)
    
    # 查找 JSON 文件
    if json_file is None:
        # 查找所有排序结果文件
        pattern = os.path.join(script_dir, "image_ranking_result*.json")
        json_files = glob.glob(pattern)
        
        if not json_files:
            print(f"❌ 未找到排序结果文件 (image_ranking_result*.json)")
            return
        
        # 按修改时间排序，选择最新的
        json_files.sort(key=os.path.getmtime, reverse=True)
        json_file = json_files[0]
        
        print(f"📂 找到 {len(json_files)} 个排序结果文件")
        print(f"📄 使用最新文件: {os.path.basename(json_file)}")
    
    json_path = json_file
    if not os.path.isabs(json_path):
        json_path = os.path.join(script_dir, json_file)
    
    if not os.path.exists(json_path):
        print(f"❌ 文件不存在: {json_path}")
        return
    
    # 生成唯一的输出文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"ranking_visualization_{timestamp}.png"
    output_path = os.path.join(script_dir, output_filename)
    
    # 执行可视化
    output_path = visualize_from_json(
        json_path=json_path,
        output_path=output_path,
        image_size=(250, 250),  # 稍大一点的图像
        cols=4,  # 每行4张图
        use_timestamp=False  # 已经手动指定了带时间戳的路径
    )
    
    print(f"\n✅ 可视化完成！")
    print(f"📍 输出文件: {output_path}")
    
    return output_path


def list_ranking_results():
    """列出所有排序结果文件"""
    pattern = os.path.join(script_dir, "image_ranking_result*.json")
    json_files = glob.glob(pattern)
    
    if not json_files:
        print("未找到排序结果文件")
        return []
    
    # 按修改时间排序
    json_files.sort(key=os.path.getmtime, reverse=True)
    
    print("\n📂 可用的排序结果文件:")
    print("-" * 60)
    for i, f in enumerate(json_files, 1):
        mtime = os.path.getmtime(f)
        mtime_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
        print(f"  {i}. {os.path.basename(f)} ({mtime_str})")
    
    return json_files


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="排序结果可视化")
    parser.add_argument(
        "--json", "-j",
        type=str,
        default=None,
        help="指定要可视化的 JSON 文件路径（默认使用最新的）"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="列出所有可用的排序结果文件"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_ranking_results()
    else:
        run_visualization(args.json)
