import json

def find_matching_and_calculate(a_file, b_file):
    # 读取 A 文件数据并建立字典，以 question_id 为键
    with open(a_file, 'r', encoding='utf-8') as af:
        a_data = {json.loads(line)['question_id']: json.loads(line) for line in af}

    # 初始化统计变量
    matching_count = 0
    total_score = 0
    total_f1_score = 0

    # 遍历 B 文件，匹配 question_id
    with open(b_file, 'r', encoding='utf-8') as bf:
        for line in bf:
            b_entry = json.loads(line)
            qid = b_entry['question_id']

            # 检查是否在 A 文件中
            if qid in a_data:
                matching_count += 1
                total_score += b_entry.get('score', 0)
                total_f1_score += b_entry.get('f1_score', 0)

    # 计算平均值
    avg_score = total_score / matching_count if matching_count > 0 else 0
    avg_f1_score = total_f1_score / matching_count if matching_count > 0 else 0

    # 打印结果
    print(f"Number of matching question_id: {matching_count}")
    print(f"Average Score: {avg_score:.4f}")
    print(f"Average F1-Score: {avg_f1_score:.4f}")

'''
5
/home/jiangkailin/project/LLaVA/playground/data/eval/eval_new_knowledge/answers/eval_vqa/DSP5_eval_vqa/DSP5_epoch_6_llava_7b_full_ft/checkpoint-40/merge_eval_acc_f1.jsonl
20
/home/jiangkailin/project/LLaVA/playground/data/eval/eval_new_knowledge/answers/eval_vqa/DSP20_eval_vqa/DSP20_epoch_6_llava_7b_full_ft/checkpoint-150/merge_eval_acc_f1.jsonl
40
/home/jiangkailin/project/LLaVA/playground/data/eval/eval_new_knowledge/answers/eval_vqa/DSP40_eval_vqa/DSP40_epoch_5_llava_7b_full_ft/merge_eval_acc_f1.jsonl
60
/home/jiangkailin/project/LLaVA/playground/data/eval/eval_new_knowledge/answers/eval_vqa/DSP60_eval_vqa/DSP60_epoch_6_llava_7b_full_ft/checkpoint-445/merge_eval_acc_f1.jsonl
80
/home/jiangkailin/project/LLaVA/playground/data/eval/eval_new_knowledge/answers/eval_vqa/DSP80_eval_vqa/DSP80_epoch_5_llava_7b_full_ft/merge_eval_acc_f1.jsonl
100
/home/jiangkailin/project/LLaVA/playground/data/eval/eval_new_knowledge/answers/eval_vqa/epoch_10_llava_7b_full_ft/checkpoint-495/merge_eval_acc_f1.jsonl

base
/home/jiangkailin/project/LLaVA/playground/data/eval/eval_new_knowledge/answers/eval_vqa/llava-v1.5-7b-base-model/merge_eval_acc_f1.jsonl
'''
a_file_path = "/home/jiangkailin/project/LLaVA/playground/data/eval/eval_new_knowledge/answers/eval_vqa/DSP5_eval_vqa/merge_DSP5_epoch_7_llava_7b_lora/merge_eval_acc_f1.jsonl"  # 替换为你的A文件路径
b_file_path = "/home/jiangkailin/project/LLaVA/playground/data/eval/eval_new_knowledge/answers/eval_vqa/DSP40_eval_vqa/merge_DSP40_epoch_7_llava_7b_lora/merge_eval_acc_f1.jsonl"  # 替换为你的B文件路径
# 调用函数
find_matching_and_calculate(a_file_path, b_file_path)
#lora
'''
5
/home/jiangkailin/project/LLaVA/playground/data/eval/eval_new_knowledge/answers/eval_vqa/DSP5_eval_vqa/merge_DSP5_epoch_7_llava_7b_lora/merge_eval_acc_f1.jsonl
20
/home/jiangkailin/project/LLaVA/playground/data/eval/eval_new_knowledge/answers/eval_vqa/DSP20_eval_vqa/merge_DSP20_epoch_7_llava_7b_lora/merge_eval_acc_f1.jsonl
40
/home/jiangkailin/project/LLaVA/playground/data/eval/eval_new_knowledge/answers/eval_vqa/DSP40_eval_vqa/merge_DSP40_epoch_7_llava_7b_lora/merge_eval_acc_f1.jsonl
60
/home/jiangkailin/project/LLaVA/playground/data/eval/eval_new_knowledge/answers/eval_vqa/DSP60_eval_vqa/merge_DSP60_epoch_7_llava_7b_lora/merge_eval_acc_f1.jsonl
80
/home/jiangkailin/project/LLaVA/playground/data/eval/eval_new_knowledge/answers/eval_vqa/DSP80_eval_vqa/merge_DSP80_epoch_7_llava_7b_lora/merge_eval_acc_f1.jsonl
100
/home/jiangkailin/project/LLaVA/playground/data/eval/eval_new_knowledge/answers/eval_vqa/merge_epoch_7_llava_7b_lora/merge_eval_acc_f1.jsonl
base
/home/jiangkailin/project/LLaVA/playground/data/eval/eval_new_knowledge/answers/eval_vqa/merge_llava_7b_lora_base_model/merge_eval_acc_f1.jsonl
'''