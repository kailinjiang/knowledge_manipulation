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

def process_files(file_a_path, file_b_path, output_txt_path):
    data_a = read_jsonl(file_a_path)
    data_b = read_jsonl(file_b_path)

    b_data_dict = {item['question_id']: item for item in data_b}

    wiki_data = []
    cnn_data = []

    for item in data_a:
        question_id = item['question_id']
        if question_id in b_data_dict:
            merged_data = {**item, **b_data_dict[question_id]}
            if item['cnn_wiki_type'] == 'wiki':
                wiki_data.append(merged_data)
            elif item['cnn_wiki_type'] == 'news':
                cnn_data.append(merged_data)

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
        for type_key, items in sorted(type_stats.items(), key=lambda x: len(x[1]), reverse=True):
            avg_score, avg_f1_score = calculate_average(items)
            stats_result.append(f"{type_key} {len(items)} {avg_score:.4f} {avg_f1_score:.4f}")
        return "\n".join(stats_result)

    wiki_type_stats = calculate_type_statistics(wiki_data)
    cnn_type_stats = calculate_type_statistics(cnn_data)

    output_content = (
        f"Wiki平均分:\nScore: {wiki_avg_score:.4f}, F1_Score: {wiki_avg_f1_score:.4f}\n\n"
        f"CNN平均分:\nScore: {cnn_avg_score:.4f}, F1_Score: {cnn_avg_f1_score:.4f}\n\n"
        f"Wiki类型统计:\n{wiki_type_stats}\n\n"
        f"CNN类型统计:\n{cnn_type_stats}\n"
    )
    write_to_txt(output_txt_path, output_content)
    print(f"结果已写入 {output_txt_path}")

def process_directory(file_a_path, b_dir_path):
    # Traverse all subdirectories of the given directory
    for root, dirs, files in os.walk(b_dir_path):
        # Look for the merge_eval_acc_f1.jsonl file in each subdirectory
        if 'merge_eval_acc_f1.jsonl' in files:
            b_file_path = os.path.join(root, 'merge_eval_acc_f1.jsonl')
            output_file_name = "type_score_output.txt"
            output_txt_path = os.path.join(root, output_file_name)
            
            print(f"Processing: {b_file_path}")
            process_files(file_a_path, b_file_path, output_txt_path)

# Set the paths for the evaluation data and the directory containing 'merge_eval_acc_f1.jsonl' files
# file_a_path = "/home/jiangkailin/project/New_Knowledge/full_training_data/1_1_data/eval_vqa.jsonl"

file_a_path = "/home/jiangkailin/project/LLaVA/our_eval/data/clean_eval_vqa.jsonl"
b_dir_path = "/home/jiangkailin/project/LLaVA/1_1_new_eval_result/clean_answer_results/llava"

# Process the directory and evaluate all 'merge_eval_acc_f1.jsonl' files in its subdirectories
process_directory(file_a_path, b_dir_path)
