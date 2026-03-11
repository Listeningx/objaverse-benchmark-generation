"""
排序结果可视化模块

将排序结果中的 query 和 candidate 图像按顺序渲染到一张大图中，方便查看。
"""

import json
import os
from typing import Dict, List, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont
import math
from datetime import datetime


def load_ranking_result(json_path: str) -> dict:
    """
    加载排序结果 JSON 文件
    
    Args:
        json_path: JSON 文件路径
        
    Returns:
        dict: 排序结果字典
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_image(image_path: str, target_size: Tuple[int, int] = (200, 200)) -> Optional[Image.Image]:
    """
    加载并调整图像大小
    
    Args:
        image_path: 图像路径
        target_size: 目标尺寸 (width, height)
        
    Returns:
        PIL.Image: 调整后的图像，如果加载失败返回 None
    """
    try:
        if not os.path.exists(image_path):
            print(f"警告: 图像不存在 - {image_path}")
            return None
        
        img = Image.open(image_path)
        img = img.convert('RGB')
        
        # 保持宽高比缩放
        img.thumbnail(target_size, Image.Resampling.LANCZOS)
        
        # 创建目标大小的白色背景
        background = Image.new('RGB', target_size, (255, 255, 255))
        
        # 将图像居中放置
        offset_x = (target_size[0] - img.width) // 2
        offset_y = (target_size[1] - img.height) // 2
        background.paste(img, (offset_x, offset_y))
        
        return background
    except Exception as e:
        print(f"加载图像失败: {image_path}, 错误: {e}")
        return None


def get_font(size: int = 14):
    """获取字体，尝试加载系统字体"""
    font_paths = [
        "C:/Windows/Fonts/msyh.ttc",  # 微软雅黑
        "C:/Windows/Fonts/simhei.ttf",  # 黑体
        "C:/Windows/Fonts/simsun.ttc",  # 宋体
        "C:/Windows/Fonts/arial.ttf",  # Arial
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
    ]
    
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, size)
            except:
                continue
    
    # 使用默认字体
    return ImageFont.load_default()


def create_ranking_visualization(
    ranking_result: dict,
    output_path: str,
    image_size: Tuple[int, int] = (200, 200),
    cols: int = 4,
    padding: int = 20,
    show_scores: bool = True
) -> str:
    """
    创建排序结果可视化大图
    
    Args:
        ranking_result: 排序结果字典
        output_path: 输出图像路径
        image_size: 单张图像大小
        cols: 每行显示的图像数量
        padding: 图像间距
        show_scores: 是否显示分数
        
    Returns:
        str: 输出图像路径
    """
    # 提取信息
    query_image_path = ranking_result.get("query_image", "")
    candidate_reports = ranking_result.get("candidate_reports", [])
    final_ranking = ranking_result.get("final_ranking", [])
    
    # 总图像数 = 1 (query) + N (candidates)
    total_images = 1 + len(candidate_reports)
    
    # 计算布局
    rows = math.ceil(total_images / cols)
    
    # 文字区域高度
    text_height = 60 if show_scores else 40
    
    # 计算大图尺寸
    canvas_width = cols * image_size[0] + (cols + 1) * padding
    canvas_height = rows * (image_size[1] + text_height) + (rows + 1) * padding + 80  # 额外80为标题区域
    
    # 创建画布
    canvas = Image.new('RGB', (canvas_width, canvas_height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    
    # 获取字体
    title_font = get_font(20)
    label_font = get_font(14)
    small_font = get_font(11)
    
    # 绘制标题
    title = "排序结果可视化"
    scenario = ranking_result.get("inferred_scenario", "")
    
    draw.text((padding, padding), title, fill=(0, 0, 0), font=title_font)
    if scenario:
        draw.text((padding, padding + 30), f"场景: {scenario[:50]}...", fill=(100, 100, 100), font=small_font)
    
    # 当前位置
    start_y = 80 + padding
    
    # ============ 绘制 Query 图像 ============
    query_img = load_image(query_image_path, image_size)
    if query_img:
        x = padding
        y = start_y
        
        # 绘制边框（蓝色表示 Query）
        border_rect = [x - 3, y - 3, x + image_size[0] + 3, y + image_size[1] + 3]
        draw.rectangle(border_rect, outline=(0, 100, 255), width=3)
        
        # 粘贴图像
        canvas.paste(query_img, (x, y))
        
        # 绘制标签
        label_y = y + image_size[1] + 5
        draw.text((x, label_y), "Query (查询图像)", fill=(0, 100, 255), font=label_font)
        
        # 显示查询分析
        query_analysis = ranking_result.get("query_object_analysis", "")
        if query_analysis and show_scores:
            short_analysis = query_analysis[:30] + "..." if len(query_analysis) > 30 else query_analysis
            draw.text((x, label_y + 18), short_analysis, fill=(100, 100, 100), font=small_font)
    
    # ============ 绘制 Candidate 图像 ============
    for idx, report in enumerate(candidate_reports):
        # 计算位置（从第二个格子开始）
        grid_idx = idx + 1  # Query 占了第一个位置
        row = grid_idx // cols
        col = grid_idx % cols
        
        x = col * (image_size[0] + padding) + padding
        y = start_y + row * (image_size[1] + text_height + padding)
        
        # 加载图像
        candidate_image_path = report.get("candidate_image", "")
        candidate_id = report.get("candidate_id", f"candidate_{idx}")
        rank = report.get("rank", idx + 1)
        score = report.get("weighted_total_score", 0)
        
        candidate_img = load_image(candidate_image_path, image_size)
        
        if candidate_img:
            # 根据排名设置边框颜色
            if rank == 1:
                border_color = (255, 215, 0)  # 金色
            elif rank == 2:
                border_color = (192, 192, 192)  # 银色
            elif rank == 3:
                border_color = (205, 127, 50)  # 铜色
            else:
                border_color = (200, 200, 200)  # 灰色
            
            # 绘制边框
            border_rect = [x - 3, y - 3, x + image_size[0] + 3, y + image_size[1] + 3]
            draw.rectangle(border_rect, outline=border_color, width=3)
            
            # 粘贴图像
            canvas.paste(candidate_img, (x, y))
            
            # 绘制排名标签
            rank_text = f"#{rank}"
            rank_bbox = draw.textbbox((0, 0), rank_text, font=label_font)
            rank_width = rank_bbox[2] - rank_bbox[0]
            rank_height = rank_bbox[3] - rank_bbox[1]
            
            # 排名标签背景
            rank_bg_rect = [x, y, x + rank_width + 10, y + rank_height + 6]
            draw.rectangle(rank_bg_rect, fill=border_color)
            draw.text((x + 5, y + 2), rank_text, fill=(0, 0, 0), font=label_font)
            
            # 绘制标签
            label_y = y + image_size[1] + 5
            
            # 显示 ID（截断）
            short_id = candidate_id[:20] + "..." if len(candidate_id) > 20 else candidate_id
            draw.text((x, label_y), short_id, fill=(50, 50, 50), font=small_font)
            
            # 显示分数
            if show_scores:
                score_text = f"Score: {score:.4f}"
                draw.text((x, label_y + 15), score_text, fill=(100, 100, 100), font=small_font)
        else:
            # 如果图像加载失败，绘制占位符
            placeholder_rect = [x, y, x + image_size[0], y + image_size[1]]
            draw.rectangle(placeholder_rect, fill=(240, 240, 240), outline=(200, 200, 200))
            draw.text((x + 10, y + image_size[1] // 2), f"#{rank} 图像缺失", fill=(150, 150, 150), font=label_font)
    
    # 绘制图例
    legend_y = canvas_height - 30
    draw.text((padding, legend_y), "图例: ", fill=(0, 0, 0), font=small_font)
    
    # Query 图例
    draw.rectangle([padding + 40, legend_y, padding + 55, legend_y + 15], outline=(0, 100, 255), width=2)
    draw.text((padding + 60, legend_y), "Query", fill=(0, 100, 255), font=small_font)
    
    # 排名图例
    legend_colors = [
        ((255, 215, 0), "#1"),
        ((192, 192, 192), "#2"),
        ((205, 127, 50), "#3"),
    ]
    
    legend_x = padding + 120
    for color, label in legend_colors:
        draw.rectangle([legend_x, legend_y, legend_x + 15, legend_y + 15], fill=color)
        draw.text((legend_x + 20, legend_y), label, fill=(50, 50, 50), font=small_font)
        legend_x += 50
    
    # 保存图像
    canvas.save(output_path, quality=95)
    print(f"✅ 可视化图像已保存: {output_path}")
    
    return output_path


def generate_unique_filename(base_name: str, extension: str = ".png", directory: str = None) -> str:
    """
    生成唯一的文件名（带时间戳）
    
    Args:
        base_name: 基础文件名
        extension: 文件扩展名
        directory: 输出目录
        
    Returns:
        str: 完整的文件路径
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{base_name}_{timestamp}{extension}"
    
    if directory:
        return os.path.join(directory, filename)
    return filename


def visualize_from_json(
    json_path: str,
    output_path: Optional[str] = None,
    image_size: Tuple[int, int] = (200, 200),
    cols: int = 4,
    use_timestamp: bool = True
) -> str:
    """
    从 JSON 文件创建可视化
    
    Args:
        json_path: 排序结果 JSON 文件路径
        output_path: 输出图像路径（可选，默认在同目录生成）
        image_size: 单张图像大小
        cols: 每行图像数量
        use_timestamp: 是否使用时间戳生成唯一文件名（默认 True）
        
    Returns:
        str: 输出图像路径
    """
    # 加载结果
    result = load_ranking_result(json_path)
    
    # 生成输出路径
    if output_path is None:
        json_dir = os.path.dirname(json_path)
        json_name = os.path.splitext(os.path.basename(json_path))[0]
        
        if use_timestamp:
            # 使用时间戳生成唯一文件名
            output_path = generate_unique_filename(
                base_name=f"{json_name}_visualization",
                extension=".png",
                directory=json_dir
            )
        else:
            output_path = os.path.join(json_dir, f"{json_name}_visualization.png")
    
    # 创建可视化
    return create_ranking_visualization(
        ranking_result=result,
        output_path=output_path,
        image_size=image_size,
        cols=cols
    )


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="排序结果可视化工具")
    parser.add_argument(
        "--json", "-j",
        type=str,
        default="image_ranking_result.json",
        help="排序结果 JSON 文件路径"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="输出图像路径"
    )
    parser.add_argument(
        "--size", "-s",
        type=int,
        default=200,
        help="单张图像大小（正方形）"
    )
    parser.add_argument(
        "--cols", "-c",
        type=int,
        default=4,
        help="每行图像数量"
    )
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    json_path = args.json
    if not os.path.isabs(json_path):
        # 如果是相对路径，尝试在当前目录查找
        script_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(script_dir, args.json)
    
    if not os.path.exists(json_path):
        print(f"❌ 文件不存在: {json_path}")
        return
    
    # 执行可视化
    output_path = visualize_from_json(
        json_path=json_path,
        output_path=args.output,
        image_size=(args.size, args.size),
        cols=args.cols
    )
    
    print(f"\n可视化完成！图像已保存到: {output_path}")


if __name__ == "__main__":
    main()
