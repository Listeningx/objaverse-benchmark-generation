# objaverse_benchmark_generation

## 数据预处理：把golen资产按类分好并分好组

openshape_clustering_pipeline.py分类生成gt
run_all_categories_pipeline.py所有类都生成gt
使用示例
```python
python openshape_clustering_pipeline.py \
--input ./objaverse_golden_character_groups.json \
--category Character \
--cache_base_dir ./test_datasets/objaverse \
--output_dir ./character \
--categorized_json ./categorized_objaverse_golden.json \
--num_cases 100 \
--llm_mode qwen \
--seed 20260301
```
## OpenShape 聚类精排 Pipeline —— 基准数据生成过程
本代码的目标是：从 Objaverse 3D 模型库中，自动生成用于3D 物体检索精排的评测基准集（benchmark）。每个评测 case 包含 1 个查询物体 + 49 个候选物体 = 50 个物体，候选物体按不同难度梯度组织，用于评估排序算法区分相似物体的能力。

### 第一步：加载输入数据
代码读取一个预先整理好的 JSON 文件（如 objaverse_golden_character_groups.json），其中按"组"（group）组织了 Objaverse 中的 3D 物体。每个物体包含：
- object_id：唯一标识
- mesh_path：3D 模型的相对路径
- image_path：渲染图片路径
- description：文本描述
- category：所属大类（如 Character、Animal、Vehicle 等）
同时，还会加载一个分类物体 JSON（categorized_objaverse_golden.json），它包含所有大类的物体数据，用于后续采样"其他类别"的干扰项。
此外，还会自动扫描所有类别的 embedding 缓存目录（如 openshape_cache/、openshape_cache_animal/ 等），将已有的 embedding 预加载到内存中，避免重复编码。

### 第二步：提取点云并用 OpenShape 编码为 Embedding
对每个 3D 物体：
1. 从 GLB 文件提取点云：用 trimesh 加载 .glb 3D 模型文件，采样 10000 个点，每个点包含 3D 坐标（xyz）和颜色（rgb），得到 [10000, 6] 的点云数组。
2. 用 OpenShape 模型编码：将点云送入 OpenShape（一个基于稀疏卷积 MinkowskiEngine 的 3D 预训练模型），输出一个归一化的 embedding 向量。这个向量捕捉了物体的 3D 形状和外观特征。
编码结果会缓存到磁盘（.npz 文件），下次运行时直接加载，无需重新编码。代码还有智能的增量编码逻辑——只对缺少 embedding 的物体进行编码。

### 第三步：KMeans 聚类 + 合并小聚类
将所有物体的 embedding 做 L2 归一化后，执行 KMeans 聚类：
1. 自动确定聚类数：如果未指定，按 总物体数 / (最小聚类大小 × 2) 估算，确保每个聚类平均有足够多的物体。
2. 合并过小的聚类：聚类完成后，将物体数少于 30 的小聚类合并到离其质心最近的大聚类中，保证每个聚类至少有 30 个物体。
3. 计算邻居关系：对每个聚类，按质心间的余弦相似度排序所有其他聚类，形成一个从"最近邻"到"最远邻"的邻居列表。这个列表在后续 case 生成中，用来区分"相邻聚类"（近似物体）和"远距聚类"（差异较大的物体）。
聚类结果也会缓存到磁盘（.pkl 文件）。

### 第四步：生成精排 Case（核心步骤）
对每个 case，流程如下：
#### 4.1 选取 Query
从有效聚类（物体数 ≥ 30）中随机选择一个 cluster，再从中随机选一个物体作为 query（查询物体）。
#### 4.2 组建候选物体（按难度梯度）
候选物体分为四个梯度：
| 梯度 | 来源 | 数量 | 说明 | 是否参与 Agent 精排 |
|------|------|------|------|----------------|
| 最相似 | 同 Cluster 中与 query 余弦相似度最高的物体 | 10 个 | 形状最接近的物体，区分难度最高 | ✅ |
| 相邻 Cluster | 距离最近的几个邻居 cluster 中随机选取 | 10 个 | 形状较接近但有差异的物体 | ✅ |
| 较远 Cluster | 排在邻居列表靠后的 cluster 中随机选取 | 20 个 | 形状差异较大，相对容易区分 | ❌（仅按余弦相似度排序） |
| 其他类别 | 完全不同的大类（如 query 是 Character，则从 Animal、Vehicle 等类别取） | 9 个 | 类别完全不同，最容易区分 | ❌（随机排列） |
#### 4.3 Agent 精排（可选）
将 query + 20 个候选物体（最相似 10 个 + 相邻 cluster 10 个）送入大语言模型（LLM）驱动的 Agent 精排系统：
- 支持 QWEN（阿里云）、Gemini（Google）或 Mock 模式
- Agent 会从多个维度（如形状相似度、语义相似度等）给每个候选物体打分
- 输出加权总分和各维度分数，并据此重新排序这 20 个物体
- 如果不使用 Agent 精排（--no_agent），则直接按余弦相似度排序。
#### 4.4 组装最终排序
最终每个 case 的 50 个物体排列为：
```
[1 个 Query] + [20 个精排物体] + [20 个较远 Cluster 物体] + [9 个其他类别物体] = 50 个物体
```
同时记录：
- 精排前后的余弦相似度分数（用于对比 Agent 精排效果）
- Agent 给出的加权分数和各维度得分
- 每个物体的详细信息（图片路径、描述、类别等）

### 第五步：保存结果
最终输出一个 JSON 文件，结构如下：
```json
{
  "metadata": { 配置信息、排序结构说明 },
  "cluster_statistics": { 聚类统计 },
  "other_category_statistics": { 其他类别统计 },
  "cases": [
    {
      "case_id": "...",
      "query_object_id": "...",
      "final_ranking": [50个物体ID],
      "ranking_details": {
        "reranked_objects": [精排后的20个],
        "top_similar_objects": [同cluster最相似的10个],
        "neighbor_random_objects": [相邻cluster的10个],
        "distant_ranking": [较远cluster的20个],
        "other_category_objects": [其他类别的9个]
      },
      "weighted_scores": { Agent精排分数 },
      "pre_rerank_cosine_scores": { 精排前余弦相似度 },
      "objects": { 每个物体的详细信息 }
    },
    ...
  ],
  "failed_cases": [失败的case信息]
}
```
每个中间 case 也会单独保存到中间缓存目录，支持断点续跑——如果某些 case 的 Agent 精排失败（如 LLM 调用超时），可以通过 --resume 参数只重跑失败的 case，已成功的保持不变。

## 设计思路总结
这个 benchmark 的核心设计理念是分梯度的难度控制：
```
最难区分 ←————————————————→ 最易区分
同Cluster最相似  相邻Cluster  较远Cluster  完全不同类别
    (10个)        (10个)       (20个)       (9个)
```
通过这种结构，可以全面评估一个 3D 检索排序算法在不同难度级别下的表现——既能衡量它区分"非常相似的物体"的精细能力，也能测试它排除"明显不同物体"的基础能力。
# other benchmark generation
run_qwen_strategy1.py 一步打分，对应strategy1
