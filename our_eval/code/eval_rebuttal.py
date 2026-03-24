import json
from tools import VQAEval
import re
import string
import collections
import os

eval_tool = VQAEval()

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()

def get_f1_score(a_pred, a_gold):
    """Calculate F1 score between predicted and gold answers."""
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def evaluate_vqa(a_file, b_file, output_file, filter_file, ignore_case=False):
    # 读取filter_file，获取所有需要保留的id
    with open(filter_file, 'r', encoding='utf-8') as ff:
        filter_ids = set()
        for line in ff:
            entry = json.loads(line)
            filter_ids.add(entry['id'])

    total_score = 0
    total_f1_score = 0
    num_entries = 0

    with open(a_file, 'r', encoding='utf-8') as af:
        a_entries = [json.loads(line) for line in af]
        a_data = {entry.get('question_id', entry.get('id')): entry['answer'] for entry in a_entries}
        a_data_type = {entry.get('question_id', entry.get('id')): entry['type'] for entry in a_entries}

    updated_b_data = []

    with open(b_file, 'r', encoding='utf-8') as bf:
        for line in bf:
            b_entry = json.loads(line)
            qid = b_entry['question_id']
            # 只处理在filter_ids中的数据
            if qid not in filter_ids:
                continue
            pred_answer = b_entry['text']
            gt_answer = a_data.get(qid, "")  
            type = a_data_type.get(qid, "")  

            # 判断是否需要忽略大小写
            if ignore_case:
                is_match = pred_answer.strip().lower() == gt_answer.strip().lower()
            else:
                is_match = pred_answer.strip() == gt_answer.strip()

            score = eval_tool.evaluate(pred_answer, [gt_answer])
            f1_score = get_f1_score(pred_answer, gt_answer)
            
            b_entry['score'] = score
            b_entry['f1_score'] = f1_score  # Add F1 score
            b_entry['gt_answer'] = gt_answer  
            b_entry['type'] = type
            b_entry['case_insensitive_match'] = is_match  # 新增字段，标记是否大小写无关匹配
            
            total_score += score
            total_f1_score += f1_score
            num_entries += 1

            updated_b_data.append(b_entry)

    avg_score = total_score / num_entries if num_entries > 0 else 0
    avg_f1_score = total_f1_score / num_entries if num_entries > 0 else 0

    with open(output_file, 'w', encoding='utf-8') as out_f:
        for entry in updated_b_data:
            out_f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"\nTotal Score: {total_score}, Total F1-Score: {total_f1_score:.4f}")
    print(f"Average Score: {avg_score:.4f}, Average F1-Score: {avg_f1_score:.4f}")
    print(f"Matched/Filtered entries: {num_entries}")
    return avg_score, avg_f1_score

# 路径配置
# a_file_path = "/home/jiangkailin/project/LLaVA/1_1_new_eval_result/rebuttal/filter/answer_3_hop.jsonl"  
# b_file_path = "/home/jiangkailin/project/LLaVA/1_1_new_eval_result/rebuttal/results/3_hop/lora/lora_3hop.jsonl"
# 路径配置



# a_file_path = "/home/jiangkailin/project/LLaVA/1_1_new_eval_result/rebuttal/filter/answer_counterfact.jsonl"  
# b_file_path = "/home/jiangkailin/project/LLaVA/1_1_new_eval_result/rebuttal/results/counterfact/lora/lora_counterfact.jsonl"



a_file_path = "/home/jiangkailin/project/LLaVA/1_1_new_eval_result/rebuttal/filter/eval_vqa.jsonl"  
b_file_path = "/home/jiangkailin/project/LLaVA/1_1_new_eval_result/rebuttal/results/ICL/res_ground_truth_1.jsonl"

output_file_name = "merge_eval_acc_f1.jsonl"  
output_file_path = os.path.join(os.path.dirname(b_file_path), output_file_name)

# 新增：filter_file路径
filter_file_path = "/home/jiangkailin/project/LLaVA/1_1_new_eval_result/rebuttal/filter/matched_ids.jsonl"  # 这里替换成你的filter文件路径

evaluate_vqa(a_file_path, b_file_path, output_file_path, filter_file_path)