#!/usr/bin/env python
"""
便捷运行脚本：可视化 openshape_clustering_output 目录下的排序结果

使用方法:
    python run_visualize_ranking.py
"""

import os
import sys

# 确保模块路径正确
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from visualize_ranking_results import (
    visualize_result_file, 
    visualize_directory,
    load_ranking_result
)


def main():
    # 默认路径配置
    INPUT_DIR = "/apdcephfs/share_303565425/DCC3/serenasnliu/GAR/Uni3D/data/test_datasets/objaverse/openshape_clustering_output"
    OUTPUT_DIR = "/apdcephfs/share_303565425/DCC3/serenasnliu/GAR/Uni3D/data/test_datasets/objaverse/openshape_clustering_output/visualizations"
    
    print("=" * 70)
    print("🎨 排序结果可视化工具")
    print("=" * 70)
    
    print("""
请选择操作模式：

1. 可视化最新的结果文件（推荐）
2. 可视化指定的结果文件
3. 批量可视化所有结果文件
4. 查看结果文件列表

请输入选项 (1-4): """, end="")
    
    try:
        choice = input().strip()
    except:
        choice = "1"
    
    if choice == "1":
        # 找到最新的文件
        import glob
        json_files = glob.glob(os.path.join(INPUT_DIR, "ranking_cases*.json"))
        if not json_files:
            print(f"❌ 未找到结果文件: {INPUT_DIR}")
            return
        
        # 按修改时间排序，取最新的
        json_files.sort(key=os.path.getmtime, reverse=True)
        latest_file = json_files[0]
        
        print(f"\n📄 最新文件: {os.path.basename(latest_file)}")
        print(f"   修改时间: {os.path.getmtime(latest_file)}")
        
        # 询问可视化选项
        print("\n选择可视化模式：")
        print("  1. 网格模式（紧凑，推荐）")
        print("  2. 详细模式（展示更多信息）")
        print("请输入 (1-2, 默认1): ", end="")
        mode_choice = input().strip()
        grid_mode = mode_choice != "2"
        
        print("\n最大可视化 case 数量（直接回车处理全部）: ", end="")
        max_cases_input = input().strip()
        max_cases = int(max_cases_input) if max_cases_input else None
        
        visualize_result_file(
            json_path=latest_file,
            output_dir=OUTPUT_DIR,
            max_cases=max_cases,
            grid_mode=grid_mode,
            dpi=200
        )
    
    elif choice == "2":
        # 指定文件
        import glob
        json_files = glob.glob(os.path.join(INPUT_DIR, "ranking_cases*.json"))
        
        print("\n可用的结果文件：")
        for i, f in enumerate(json_files, 1):
            size_kb = os.path.getsize(f) / 1024
            mtime = os.path.getmtime(f)
            from datetime import datetime
            mtime_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
            print(f"  {i}. {os.path.basename(f)} ({size_kb:.1f} KB, {mtime_str})")
        
        print("\n请输入文件编号或完整路径: ", end="")
        file_input = input().strip()
        
        if file_input.isdigit():
            idx = int(file_input) - 1
            if 0 <= idx < len(json_files):
                selected_file = json_files[idx]
            else:
                print("❌ 无效编号")
                return
        else:
            selected_file = file_input
        
        if not os.path.exists(selected_file):
            print(f"❌ 文件不存在: {selected_file}")
            return
        
        visualize_result_file(
            json_path=selected_file,
            output_dir=OUTPUT_DIR,
            max_cases=None,
            grid_mode=True,
            dpi=200
        )
    
    elif choice == "3":
        # 批量处理
        print("\n⚠️ 批量处理可能需要较长时间，是否继续？(y/n): ", end="")
        confirm = input().strip().lower()
        if confirm != 'y':
            print("已取消")
            return
        
        print("\n每个文件最大可视化 case 数量（直接回车处理全部）: ", end="")
        max_cases_input = input().strip()
        max_cases = int(max_cases_input) if max_cases_input else None
        
        visualize_directory(
            input_dir=INPUT_DIR,
            output_dir=OUTPUT_DIR,
            max_cases_per_file=max_cases,
            grid_mode=True,
            dpi=200
        )
    
    elif choice == "4":
        # 查看文件列表
        import glob
        json_files = glob.glob(os.path.join(INPUT_DIR, "ranking_cases*.json"))
        
        print(f"\n📂 目录: {INPUT_DIR}")
        print(f"📊 共找到 {len(json_files)} 个结果文件\n")
        
        for i, f in enumerate(json_files, 1):
            size_kb = os.path.getsize(f) / 1024
            from datetime import datetime
            mtime = datetime.fromtimestamp(os.path.getmtime(f)).strftime("%Y-%m-%d %H:%M:%S")
            
            # 读取文件获取 case 数量
            try:
                data = load_ranking_result(f)
                n_cases = len(data.get('cases', []))
                metadata = data.get('metadata', {})
                use_agent = "✅" if metadata.get('use_agent_ranking', False) else "❌"
            except:
                n_cases = "?"
                use_agent = "?"
            
            print(f"  {i}. {os.path.basename(f)}")
            print(f"     大小: {size_kb:.1f} KB | 时间: {mtime} | Cases: {n_cases} | Agent: {use_agent}")
    
    else:
        print("无效选项")


if __name__ == "__main__":
    main()
