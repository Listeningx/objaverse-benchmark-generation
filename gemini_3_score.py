from google import genai
from google.genai import types
import os
import glob
import mimetypes

# ===================== 配置项 =====================
# 1. 批量处理配置
# 支持多个输入目录，所有结果输出到同一个目录
INPUT_DIRS = [
    r"D:\3d-object-数据集\评测\GSO_resample\image_lists_extra",
    r"D:\3d-object-数据集\评测\GSO_resample\image_lists_specific",  # 可以添加更多目录
    # r"D:\3d-object-数据集\评测\其他数据集\image_lists",
]
OUTPUT_DIR = r"D:\code\score_benchmark\gemini3flash_x.x"  # 输出目录：保存打分结果
# 2. Gemini 配置（需先配置 API Key，参考下方说明）
GENAI_API_KEY = ""  # 替换为你的 Gemini API Key
os.environ['GEMINI_API_KEY']=''

MODEL_NAME = "gemini-3-flash-preview"
# 3. 评分指令（可根据你的需求修改）
SCORING_PROMPT = """
Given 50 pictures, Image 1 is the query image, and Figures 2 to 50 are the candidate images. Your task is to evaluate the similarity between each candidate image and the queried image, and assign the final similarity score according to the following rules.
Overview of the scoring scheme
Evaluate similarity using a hierarchical scoring scheme:
First, based on strict semantic definitions, each candidate image is assigned to one of the five major similarities (1-5).
Then, for fractions 1 to 4, use decimal fractions to further refine the similarity within the same level. The fraction 5 is reserved for the same object and cannot be further subdivided.
Five main similarities and the definition of detail similarity within each level:
Score = 5
Candidate images describe instances of physical objects that are the same as the query image. The differences are limited to viewpoints, poses, lighting or rendering. Do not use decimal fractions at this level.
The score is 4.x
The candidate image describes a different instance of an object, but its function is the same as that of the query image, and its overall geometry, materials, and colors are also similar. Score the details of objects based on the similarity of geometric structure, material and color.
The score is 3.x
The candidate image has the same function as the query image, but: the geometric structure is significantly different, or the color and material are significantly different. Score the details of objects based on the similarity of geometric structure, material and color.
The score is 2.x
The candidate images belong to the same semantic category, but their functions are obviously different. Fine-grained scoring is given to objects at this level based on the similarity of function, color and geometric structure.
The score is 1.x
The candidate images belong to different categories, with only the overall visual style being similar. Fine-grained scoring is given to objects at this level based on the similarity of semantic categories.
Anti-collapse constraint (mandatory)
To ensure fine-grained and meaningful evaluations, you must follow the following rules:
Use all five major similarity levels (1-5).
Each major level must contain multiple candidate images.
For levels 1 to 4, multiple different decimal fractions are used in each level.
Do not assign the same decimal fraction to all images in one level.
Do not rate most images to a single score or a single major grade.
When multiple images look similar, you still have to sort them relatively based on subtle differences.
When scoring, please ignore the differences in perspective and focus on the 3D objects described in the view.
These constraints take precedence over simplicity or convenience.
The output format must strictly follow: each line should only contain "serial_number score", separated by Spaces, for example,
1 5.0
2 5.0
...
50 1.0
Only output the content in the above format. Do not add any additional text, explanations or punctuation. Make sure there are 50 rows.

"""

# --------------- 新增：解析并排序分数 ---------------
def parse_and_sort_scores(response_text):
    """
    解析Gemini返回的分数文本，按相似度分数降序排序
    返回：排序后的列表，每个元素为 (序号, 图片路径, 分数)
    """
    score_list = []
    lines = response_text.strip().split("\n")
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # 拆分每行（按空格拆分，处理路径中含空格的情况）
        parts = line.split()
        try:
            # 序号是第一个元素，分数是最后一个元素，中间是路径
            idx = int(parts[0])
            score = float(parts[-1])
            # img_path = " ".join(parts[1:-1])  # 拼接路径（处理路径含空格）
            score_list.append((idx, score))
        except (ValueError, IndexError):
            print(f"警告：无效行，跳过 → {line}")
            continue
    
    # 按分数降序排序（核心：key=lambda x: x[2] 取分数，reverse=True 降序）
    sorted_score_list = sorted(score_list, key=lambda x: x[1], reverse=True)
    return sorted_score_list

# ===================== 核心函数 =====================
def load_image_paths(file_path):
    """读取图片路径文件，验证路径有效性并返回有效路径列表"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"图片列表文件不存在：{file_path}")
    
    valid_paths = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            img_path = line.strip()
            if not img_path:
                continue  # 跳过空行
            
            # 验证图片文件是否存在
            if not os.path.exists(img_path):
                print(f"警告：第 {line_num} 行路径无效，跳过 → {img_path}")
                continue
            
            valid_paths.append(img_path)
    
    print(f"✅ 成功加载 {len(valid_paths)} 个有效图片路径（总计读取 {line_num} 行）")
    return valid_paths

def get_mime_type(img_path):
    """根据图片路径获取对应的 MIME 类型（自动适配jpg/png等格式）"""
    mime_type, _ = mimetypes.guess_type(img_path)
    if not mime_type:
        # 兜底处理常见格式
        ext = os.path.splitext(img_path)[1].lower()
        mime_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".bmp": "image/bmp",
            ".gif": "image/gif"
        }
        mime_type = mime_map.get(ext, "image/jpeg")
    return mime_type

def prepare_image_parts(img_paths):
    """将图片路径转换为 Gemini 可接受的 Part 格式（批量处理）"""
    image_parts = []
    for idx, img_path in enumerate(img_paths, 1):
        try:
            # 读取图片字节数据
            with open(img_path, "rb") as f:
                img_bytes = f.read()
            
            # 获取 MIME 类型并构造 Part
            mime_type = get_mime_type(img_path)
            img_part = types.Part.from_bytes(
                data=img_bytes,
                mime_type=mime_type
            )
            image_parts.append(img_part)
            print(f"✅ 已准备图片 {idx}/{len(img_paths)} → {img_path}")
        
        except Exception as e:
            print(f"❌ 处理图片失败 {idx}/{len(img_paths)} → {img_path}，错误：{e}")
    
    return image_parts

def send_gemini_request(prompt, image_parts):
    """向 Gemini 3 Pro 发送包含文本指令和多图片的请求"""
    # 配置 API Key
    genai.configure(api_key=GENAI_API_KEY)
    client = genai.Client(
        # api_key="",
        # http_options={
        #     "base_url": "https://api.yyds168.net",  # 替换为你要使用的网址
        # },
    )
    
    # 构造请求内容（文本指令 + 所有图片）
    contents = [prompt] + image_parts
    
    print("\n🚀 正在向 Gemini 3 Pro 发送请求（包含文本指令 + 图片）...")
    print(f"   - 图片数量：{len(image_parts)}")
    print(f"   - 模型：{MODEL_NAME}")
    
    # 发送请求（添加超时和重试机制，适配大请求）
    try:
        total_tokens = client.models.count_tokens(
            model=MODEL_NAME, contents=contents
        )
        print("total_tokens: ", total_tokens)
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            # generation_config=types.GenerationConfig(
            #     temperature=0.1,  # 低随机性保证打分稳定
            #     max_output_tokens=8192  # 足够容纳70张图片的打分结果
            # )
        )
        # 检查响应是否有效
        if response.text:
            print("✅ Gemini 响应成功！")
            print(response.usage_metadata)
            print(response.text)

            # 解析并排序分数
            sorted_scores = parse_and_sort_scores(response.text)
            
            # 输出排序结果
            print("\n✅ 按相似度分数降序排序结果：")
            print("排名 | 序号 |  相似度分数")
            print("-" * 80)
            for rank, (idx,score) in enumerate(sorted_scores, 1):
                print(f"{rank:4d} | {idx:3d} | {score:.1f}")
            
            return sorted_scores
            # return response.text
        else:
            raise ValueError("Gemini 返回空响应")
    
    except Exception as e:
        raise RuntimeError(f"请求 Gemini 失败：{e}")


def save_result(sorted_scores, save_path):
    """保存打分结果到文件"""
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("排名 | 原始序号 | 相似度分数\n")
        f.write("-" * 80 + "\n")
        for rank, (idx, score) in enumerate(sorted_scores, 1):
            f.write(f"{rank} | {idx} | {score:.1f}\n")
    print(f"📄 打分结果已保存到：{save_path}")

def process_single_file(input_file, output_file):
    """处理单个txt文件"""
    try:
        print(f"\n{'='*80}")
        print(f"📝 处理文件: {os.path.basename(input_file)}")
        
        # 1. 加载并验证图片路径
        img_paths = load_image_paths(input_file)
        if not img_paths:
            print(f"⚠️  无有效图片路径，跳过该文件")
            return False
        
        # 2. 准备图片数据（转换为 Gemini 可接受的格式）
        image_parts = prepare_image_parts(img_paths)
        if len(image_parts) == 0:
            print(f"⚠️  无可用的图片数据，跳过该文件")
            return False
        
        # 3. 发送请求获取打分结果
        scoring_result = send_gemini_request(SCORING_PROMPT, image_parts)
        
        # 4. 保存结果
        save_result(scoring_result, output_file)
        print(f"✅ 文件处理成功: {os.path.basename(input_file)}")
        return True
    
    except Exception as e:
        print(f"❌ 文件处理失败: {os.path.basename(input_file)} → {e}")
        return False
# ===================== 主程序 =====================
if __name__ == "__main__":
    print("🎯 批量打分处理开始")
    print(f"输入目录数量: {len(INPUT_DIRS)}")
    for i, dir_path in enumerate(INPUT_DIRS, 1):
        print(f"  [{i}] {dir_path}")
    print(f"输出目录: {OUTPUT_DIR}")
    
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 从所有输入目录中收集txt文件，并记录文件来源
    all_input_files = []  # 存储 (文件路径, 目录索引, 目录名称) 的元组
    
    for dir_idx, input_dir in enumerate(INPUT_DIRS, 1):
        if not os.path.exists(input_dir):
            print(f"⚠️  输入目录不存在，跳过: {input_dir}")
            continue
        
        # 提取目录名称作为类别标识
        dir_name = os.path.basename(input_dir)
        
        # 查找当前目录下的所有txt文件
        txt_files = glob.glob(os.path.join(input_dir, "*.txt"))
        
        # 为每个文件添加来源信息
        for txt_file in txt_files:
            all_input_files.append((txt_file, dir_idx, dir_name))
        
        print(f"  ✅ 从 {dir_name} 找到 {len(txt_files)} 个txt文件")
    
    # 按文件路径排序
    all_input_files.sort(key=lambda x: x[0])
    
    print(f"\n📊 总计找到 {len(all_input_files)} 个txt文件")
    
    if not all_input_files:
        print("❌ 未找到任何txt文件，程序退出")
        exit(1)
    
    # 统计处理结果
    success_count = 0
    fail_count = 0
    skip_count = 0
    
    # 逐个处理文件
    for input_file, dir_idx, dir_name in all_input_files:
        # 从输入文件名提取基础名称（不含扩展名）
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        
        # 构造输出文件名：类别名_序号_原文件名.txt
        output_filename = f"{dir_name}_{dir_idx:02d}_{base_name}.txt"
        output_file = os.path.join(OUTPUT_DIR, output_filename)
        
        # 检查输出文件是否已存在（可选：跳过已处理的文件）
        if os.path.exists(output_file):
            print(f"\n⚠️  输出文件已存在，跳过: {output_filename}")
            skip_count += 1
            continue
        
        # 处理单个文件
        print(f"\n📝 处理: {os.path.basename(input_file)} → {output_filename}")
        if process_single_file(input_file, output_file):
            success_count += 1
        else:
            fail_count += 1
    
    # 输出总体统计
    print(f"\n{'='*80}")
    print("\n🎉 批量处理完成！")
    print(f"✅ 成功: {success_count} 个文件")
    print(f"❌ 失败: {fail_count} 个文件")
    print(f"⏭️  跳过: {skip_count} 个文件（已存在）")
    print(f"📁 结果保存目录: {OUTPUT_DIR}")