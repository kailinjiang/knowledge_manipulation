import json
from collections import defaultdict
import os


def read_jsonl(file_path):
    """读取JSONL文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def write_to_txt(file_path, content):
    """将内容写入TXT文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)


def build_label_dict(label_file_path):
    """读取label文件，构建question_id -> label映射"""
    label_data = read_jsonl(label_file_path)
    label_dict = {}
    for item in label_data:
        qid = item.get('question_id', item.get('id'))
        if qid is not None and 'label' in item:
            label_dict[qid] = item['label']
    return label_dict


def process_files(file_a_path, file_b_path, label_file_path, output_txt_path):
    data_a = read_jsonl(file_a_path)
    data_b = read_jsonl(file_b_path)
    label_dict = build_label_dict(label_file_path)

    b_data_dict = {item['question_id']: item for item in data_b}

    wiki_data = []
    cnn_data = []
    label_stats = defaultdict(list)

    for item in data_a:
        question_id = item.get('question_id', item.get('id'))
        if question_id in b_data_dict:
            merged_data = {**item, **b_data_dict[question_id]}

            type_value = item.get('cnn_wiki_type', item.get('type'))  # 优先取cnn_wiki_type，没有则取type
            if type_value == 'wiki':
                wiki_data.append(merged_data)
            elif type_value == 'news':
                cnn_data.append(merged_data)

            label_value = label_dict.get(question_id)
            if label_value in (1, 2, "1", "2"):
                label_stats[str(label_value)].append(merged_data)

    def calculate_average(data_list):
        if not data_list:
            return 0, 0
        total_score = sum(item['score'] for item in data_list)
        total_f1_score = sum(item['f1_score'] for item in data_list)
        return total_score / len(data_list), total_f1_score / len(data_list)

    wiki_avg_score, wiki_avg_f1_score = calculate_average(wiki_data)
    cnn_avg_score, cnn_avg_f1_score = calculate_average(cnn_data)

    def calculate_type_statistics(data_list):
        type_stats = defaultdict(list)
        for item in data_list:
            type_stats[item['type']].append(item)
        stats_result = []
        type_count = len(type_stats)
        for type_key, items in sorted(type_stats.items(), key=lambda x: len(x[1]), reverse=True):
            avg_score, avg_f1_score = calculate_average(items)
            stats_result.append(f"{type_key} {len(items)} {avg_score:.4f} {avg_f1_score:.4f}")
        print(f"所有类型的总数: {type_count}")
        return "\n".join(stats_result)

    def calculate_label_statistics(label_data):
        if not label_data:
            return "无匹配的label记录"
        stats_lines = []
        for label_key in sorted(label_data.keys()):
            items = label_data[label_key]
            avg_score, avg_f1_score = calculate_average(items)
            stats_lines.append(
                f"Label {label_key} Count: {len(items)} Score(ACC): {avg_score:.4f} F1: {avg_f1_score:.4f}"
            )
        return "\n".join(stats_lines)

    wiki_type_stats = calculate_type_statistics(wiki_data)
    cnn_type_stats = calculate_type_statistics(cnn_data)
    label_stats_output = calculate_label_statistics(label_stats)

    output_content = (
        f"Wiki平均分:\nScore: {wiki_avg_score:.4f}, F1_Score: {wiki_avg_f1_score:.4f}\n\n"
        f"CNN平均分:\nScore: {cnn_avg_score:.4f}, F1_Score: {cnn_avg_f1_score:.4f}\n\n"
        f"Wiki类型统计:\n{wiki_type_stats}\n\n"
        f"CNN类型统计:\n{cnn_type_stats}\n\n"
        f"Label统计(ACC/F1):\n{label_stats_output}\n"
    )
    write_to_txt(output_txt_path, output_content)
    print("Label统计(ACC/F1):")
    print(label_stats_output)
    print(f"结果已写入 {output_txt_path}")


file_a_path = "/home/jiangkailin/project/New_Knowledge/full_training_data/1_1_data/eval_vqa.jsonl"
label_file_path = "/home/jiangkailin/project/LLaVA/our_eval/new/code/news_check_labeled.jsonl"

# file_a_path = "/home/jiangkailin/project/LLaVA/our_eval/data/clean_eval_vqa.jsonl"
# file_a_path = "/home/jiangkailin/project/LLaVA/1_1_new_eval_result/rebuttal/filter/answer_3_hop.jsonl"
# file_a_path = "/home/jiangkailin/project/LLaVA/1_1_new_eval_result/rebuttal/filter/answer_counterfact.jsonl"

file_b_path = "/home/jiangkailin/project/LLaVA/1_1_new_eval_result/IAG/Perplexity_AI/all/merge_eval_acc_f1.jsonl"
output_file_name = "rebuttal_type_score_output.txt"

base_name = os.path.basename(file_b_path)
output_file_name = base_name.split('.')[0] + "_" + output_file_name
output_txt_path = os.path.join(os.path.dirname(file_b_path), output_file_name)

process_files(file_a_path, file_b_path, label_file_path, output_txt_path)