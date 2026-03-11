from openai import OpenAI
import os
import base64

# ===================== 配置项 =====================
# 1. 文件路径配置
IMAGE_LIST_FILE = r"D:\3d-object-数据集\评测\GSO_resample\image_lists_50\Toys_01.txt"  # 图片路径列表文件
RESULT_SAVE_FILE = r"D:\code\gemini_benchmark\test_qwen_50_longp\Toys_01.txt"  # 打分结果保存文件

# 2. Qwen 配置
API_KEY = ''
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen3-vl-235b-a22b-thinking"

# 3. 评分指令（与 Gemini 版本相同）
SCORING_PROMPT ="""
Given:
- One query image of a 3D object.
- A list of candidate images rendered from different 3D objects and viewpoints.

For each candidate image, compute a similarity score with respect to the query image.
Similarity must be evaluated under a strict priority order.
Higher-priority criteria MUST dominate lower-priority ones and should not be overridden by them.
1. Same object instance across different viewpoints (Highest priority)
Images from the same 3D object must be ranked highest, regardless of viewpoint or pose.
2. Same function with similar geometry and color
Objects designed for the same function, while also exhibiting similar geometric structures and comparable color appearance, should be ranked higher than those sharing only functional similarity.
3. Same function
Objects serving the same functional purpose, even if they differ in geometry, structure, or color, should be ranked next.
4. Same object category
Objects belonging to the same semantic category, regardless of functional variation or visual differences, should be ranked lower than function-level similarity.
5. Same style or aesthetic attributes (Lowest priority)
Objects sharing similar design style or aesthetic attributes (e.g., modern, vintage, minimalistic) should be considered only when higher-level criteria are not satisfied.
Take the first image as the query and score the 50 images according to the above requirements.
Objects sharing similar design style or aesthetic attributes (e.g., modern, vintage, minimalistic) should be considered only when higher-level criteria are not satisfied.

IMPORTANT SCORING RULES
Exact same image as the query must receive the maximum possible score.
Different views of the same 3D object instance must receive a score very close to the maximum, regardless of viewpoint or pose differences.
Different objects with the same function, exhibiting highly similar geometric structure and color appearance, should receive a high but clearly lower score than the same-instance cases.
Objects with the same function as the query, but with noticeable differences in geometry or color, should receive a moderate score.
Objects belonging to the same semantic category but serving different functions or with substantially different geometry, should receive a low score.
Objects sharing only stylistic or aesthetic similarity (e.g., modern, vintage, minimalistic), without functional or categorical alignment, must receive very low scores.
Completely unrelated objects must receive the lowest possible scores.

Finally output a "Total score of Comprehensive Similarity" for each picture (0 to 10 points, with a higher score indicating a higher overall similarity).
The output format must strictly follow: each line should only contain "serial_number score", separated by Spaces, for example:
1 9.5
2 8.2
...
50 7.8
Only output the content in the above format. Do not add any additional text, explanations or punctuation. Make sure there are 50 rows.
"""

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

def encode_image_to_base64(img_path):
    """将图片编码为 base64 格式"""
    with open(img_path, "rb") as f:
        img_bytes = f.read()
    return base64.b64encode(img_bytes).decode('utf-8')

def prepare_qwen_messages(img_paths, prompt):
    """准备 Qwen API 的消息格式"""
    content = []
    
    # 添加所有图片
    for idx, img_path in enumerate(img_paths, 1):
        try:
            # 方式1：使用本地文件路径（如果支持）
            # 方式2：使用 base64 编码
            base64_image = encode_image_to_base64(img_path)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })
            print(f"✅ 已准备图片 {idx}/{len(img_paths)} → {img_path}")
        except Exception as e:
            print(f"❌ 处理图片失败 {idx}/{len(img_paths)} → {img_path}，错误：{e}")
    
    # 添加文本指令
    content.append({"type": "text", "text": prompt})
    
    return [{"role": "user", "content": content}]

def parse_and_sort_scores(response_text):
    """
    解析返回的分数文本，按相似度分数降序排序
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
    
    # 按分数降序排序
    sorted_score_list = sorted(score_list, key=lambda x: x[1], reverse=True)
    return sorted_score_list

def send_qwen_request(messages):
    """向 Qwen API 发送请求"""
    # 初始化OpenAI客户端
    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL
    )
    
    print("\n🚀 正在向 Qwen API 发送请求...")
    print(f"   - 模型：{MODEL_NAME}")
    
    try:
        # 创建聊天完成请求（非流式）
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            extra_body={
                'enable_thinking': True,
            },
            stream=False,
            temperature=0.1,  # 低随机性保证打分稳定
            max_tokens=8192  # 足够容纳70张图片的打分结果
        )
        
        # 获取响应内容
        response_text = completion.choices[0].message.content
        
        if response_text:
            print("✅ Qwen 响应成功！")
            print(f"   - Token 使用量：{completion.usage}")
            print("\n=================== Qwen 原始响应 ===================")
            print(response_text)
            
            # 解析并排序分数
            sorted_scores = parse_and_sort_scores(response_text)
            
            # 输出排序结果
            print("\n✅ 按相似度分数降序排序结果：")
            print("排名 | 序号 |  相似度分数")
            print("-" * 80)
            for rank, (idx, score) in enumerate(sorted_scores, 1):
                print(f"{rank:4d} | {idx:3d} | {score:.1f}")
            
            return sorted_scores
        else:
            raise ValueError("Qwen 返回空响应")
    
    except Exception as e:
        raise RuntimeError(f"请求 Qwen 失败：{e}")

def save_result(sorted_scores, save_path):
    """保存打分结果到文件"""
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("排名 | 原始序号 | 图片路径 | 相似度分数\n")
        f.write("-" * 80 + "\n")
        for rank, (idx, score) in enumerate(sorted_scores, 1):
            f.write(f"{rank} | {idx}| {score:.1f}\n")
    print(f"\n📄 打分结果已保存到：{save_path}")

# ===================== 主程序 =====================
if __name__ == "__main__":
    try:
        # 1. 加载并验证图片路径
        img_paths = load_image_paths(IMAGE_LIST_FILE)
        if not img_paths:
            raise ValueError("无有效图片路径，程序终止")
        
        # 2. 准备消息格式
        messages = prepare_qwen_messages(img_paths, SCORING_PROMPT)
        
        # 3. 发送请求获取打分结果
        sorted_scores = send_qwen_request(messages)
        
        # 4. 保存结果
        save_result(sorted_scores, RESULT_SAVE_FILE)
        
        print("\n✅ 程序执行完成！")
    
    except Exception as e:
        print(f"\n❌ 程序执行失败：{e}")