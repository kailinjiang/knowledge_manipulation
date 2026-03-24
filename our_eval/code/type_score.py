import json
from collections import defaultdict, Counter

# 定义函数：统计 type 的数量和对应的 score 均值
def analyze_jsonl_scores(file_path, output_txt):
    type_scores = defaultdict(list)  # 存储每种 type 的 score 列表

    # 读取 JSONL 文件
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)  # 解析每一行
            type_key = entry.get('type', None)  # 提取 type 字段
            score = entry.get('score', 0)  # 提取 score 字段，默认为 0
            if type_key:  # 如果 type 存在
                type_scores[type_key].append(score)

    # 统计每种 type 的数量和 score 均值
    type_stats = []
    for t, scores in type_scores.items():
        avg_score = sum(scores) / len(scores) if scores else 0
        type_stats.append((t, len(scores), avg_score))

    # 按 type 数量从大到小排序
    type_stats_sorted = sorted(type_stats, key=lambda x: x[1], reverse=True)

    # 打印数量最多的前10种 type 及其 score 均值
    print("数量最多的前10种 type 及其 score 均值：")
    for t, count, avg_score in type_stats_sorted[:10]:
        print(f"Type: {t}, Count: {count}, Average Score: {avg_score:.4f}")

    # 将所有 type 的统计结果写入 txt 文件
    with open(output_txt, 'w', encoding='utf-8') as out_f:
        out_f.write("Type\tCount\tAverage_Score\n")
        for t, count, avg_score in type_stats_sorted:
            out_f.write(f"{t}\t{count}\t{avg_score:.4f}\n")

    print(f"\n所有统计结果已写入文件：{output_txt}")

# 文件路径
input_file = "/home/jiangkailin/project/LLaVA/playground/data/eval/evalwiki/answers/12_15_eval_wikidata_vqa/1_wikidata-llava-v1.5-7b-task-full-ft/merge_eval.jsonl"  # 请替换为您的 JSONL 文件路径
output_file = "/home/jiangkailin/project/LLaVA/playground/data/eval/evalwiki/answers/12_15_eval_wikidata_vqa/1_wikidata-llava-v1.5-7b-task-full-ft/type_score_statistics.txt"  # 输出 txt 文件路径

# 执行函数
analyze_jsonl_scores(input_file, output_file)
