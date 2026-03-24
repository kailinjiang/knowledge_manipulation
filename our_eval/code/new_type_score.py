import json

# 定义函数来统计每个type的平均score
def calculate_average_score_by_type(file_path, output_file):
    # 初始化数据结构
    type_scores = {}
    type_counts = {}

    # 读取JSONL文件
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            type_key = data.get("type")
            score = data.get("score", 0)

            # 收集每种type的score
            if type_key in type_scores:
                type_scores[type_key] += score
                type_counts[type_key] += 1
            else:
                type_scores[type_key] = score
                type_counts[type_key] = 1

    # 计算每种type的平均score
    type_avg_scores = {key: type_scores[key] / type_counts[key] for key in type_scores}

    # 按type数量从大到小排序
    sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)

    # 将结果写入TXT文件
    with open(output_file, 'w', encoding='utf-8') as output:
        output.write("Type\tCount\tAverage_Score\n")
        for type_key, count in sorted_types:
            avg_score = type_avg_scores[type_key]
            output.write(f"{type_key}\t{count}\t{avg_score:.6f}\n")

# 使用函数
input_file = "/home/jiangkailin/project/LLaVA/playground/data/eval/evalwiki/answers/12_15_eval_wikidata_vqa/merge_2_wikidata-llava-v1.5-7b-task-lora/merge_eval_type.jsonl"  # 替换为您的JSONL文件路径
output_file = "/home/jiangkailin/project/LLaVA/playground/data/eval/evalwiki/answers/12_15_eval_wikidata_vqa/merge_2_wikidata-llava-v1.5-7b-task-lora/type_score.txt"  # 替换为您希望生成的TXT文件路径
calculate_average_score_by_type(input_file, output_file)
