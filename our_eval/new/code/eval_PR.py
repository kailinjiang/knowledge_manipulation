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
1
/home/jiangkailin/project/New_Knowledge/full_training_data/phase_data/Phase1_eval_vqa.jsonl
/home/jiangkailin/project/LLaVA/1_1_new_eval_result/new_knowledge/PR/full-ft/llava_ckpt_PR1_full_ft_epoch_6/merge_eval_acc_f1.jsonl
2
/home/jiangkailin/project/New_Knowledge/full_training_data/phase_data/Phase2_eval_vqa.jsonl
/home/jiangkailin/project/LLaVA/1_1_new_eval_result/new_knowledge/PR/full-ft/llava_ckpt_PR2_full_ft_epoch_6/merge_eval_acc_f1.jsonl
3
/home/jiangkailin/project/New_Knowledge/full_training_data/phase_data/Phase3_eval_vqa.jsonl
/home/jiangkailin/project/LLaVA/1_1_new_eval_result/new_knowledge/PR/full-ft/llava_ckpt_PR3_full_ft_epoch_6/merge_eval_acc_f1.jsonl
4
/home/jiangkailin/project/New_Knowledge/full_training_data/phase_data/Phase4_eval_vqa.jsonl
/home/jiangkailin/project/LLaVA/1_1_new_eval_result/new_knowledge/PR/full-ft/llava_ckpt_PR4_full_ft_epoch_6/merge_eval_acc_f1.jsonl

100
/home/jiangkailin/project/LLaVA/1_1_new_eval_result/new_knowledge/epoch/llava_full_ft/llava_ckpt_full_ft_epoch_6/merge_eval_acc_f1.jsonl

base model
/home/jiangkailin/project/LLaVA/playground2/data/eval/eval_new_knowledge/answers/eval_vqa/llava-v1.5-7b-base-model/merge_eval_acc_f1.jsonl
'''

# 示例文件路径（请替换为实际路径）
# a_file_path = "/home/jiangkailin/project/New_Knowledge/full_training_data/1_1_data/phase_data/p12/p12_Phase12_eval_vqa.jsonl" 

# a_file_path = "/home/jiangkailin/project/New_Knowledge/full_training_data/1_1_data/phase_data/p8/p8_Phase8_eval_vqa.jsonl" 


# a_file_path = "/home/jiangkailin/project/New_Knowledge/full_training_data/1_1_data/phase_data/p4/Phase1_eval_vqa.jsonl"
# a_file_path = "/home/jiangkailin/project/New_Knowledge/full_training_data/1_1_data/phase_data/p4/all_Phase2_eval_vqa.jsonl"  
# a_file_path = "/home/jiangkailin/project/New_Knowledge/full_training_data/1_1_data/phase_data/p4/all_Phase3_eval_vqa.jsonl"  
# a_file_path = "/home/jiangkailin/project/New_Knowledge/full_training_data/1_1_data/phase_data/p4/all_Phase4_eval_vqa.jsonl" 

# a_file_path = "/home/jiangkailin/project/New_Knowledge/full_training_data/1_1_data/phase_data/p8/all_p8_Phase1_eval_vqa.jsonl"
# a_file_path = "/home/jiangkailin/project/New_Knowledge/full_training_data/1_1_data/phase_data/p8/all_p8_Phase2_eval_vqa.jsonl"
# a_file_path = "/home/jiangkailin/project/New_Knowledge/full_training_data/1_1_data/phase_data/p8/all_p8_Phase3_eval_vqa.jsonl"
# a_file_path = "/home/jiangkailin/project/New_Knowledge/full_training_data/1_1_data/phase_data/p8/all_p8_Phase4_eval_vqa.jsonl"
# a_file_path = "/home/jiangkailin/project/New_Knowledge/full_training_data/1_1_data/phase_data/p8/all_p8_Phase5_eval_vqa.jsonl"
# a_file_path = "/home/jiangkailin/project/New_Knowledge/full_training_data/1_1_data/phase_data/p8/all_p8_Phase6_eval_vqa.jsonl"
# a_file_path = "/home/jiangkailin/project/New_Knowledge/full_training_data/1_1_data/phase_data/p8/all_p8_Phase7_eval_vqa.jsonl"
# a_file_path = "/home/jiangkailin/project/New_Knowledge/full_training_data/1_1_data/phase_data/p8/all_p8_Phase8_eval_vqa.jsonl"

# a_file_path = "/home/jiangkailin/project/New_Knowledge/full_training_data/1_1_data/phase_data/p12/all_p12_Phase1_eval_vqa.jsonl"
# a_file_path = "/home/jiangkailin/project/New_Knowledge/full_training_data/1_1_data/phase_data/p12/all_p12_Phase2_eval_vqa.jsonl"
# a_file_path = "/home/jiangkailin/project/New_Knowledge/full_training_data/1_1_data/phase_data/p12/all_p12_Phase3_eval_vqa.jsonl"
# a_file_path = "/home/jiangkailin/project/New_Knowledge/full_training_data/1_1_data/phase_data/p12/all_p12_Phase4_eval_vqa.jsonl"
# a_file_path = "/home/jiangkailin/project/New_Knowledge/full_training_data/1_1_data/phase_data/p12/all_p12_Phase5_eval_vqa.jsonl"
# a_file_path = "/home/jiangkailin/project/New_Knowledge/full_training_data/1_1_data/phase_data/p12/all_p12_Phase6_eval_vqa.jsonl"
# a_file_path = "/home/jiangkailin/project/New_Knowledge/full_training_data/1_1_data/phase_data/p12/all_p12_Phase7_eval_vqa.jsonl"
# a_file_path = "/home/jiangkailin/project/New_Knowledge/full_training_data/1_1_data/phase_data/p12/all_p12_Phase8_eval_vqa.jsonl"
# a_file_path = "/home/jiangkailin/project/New_Knowledge/full_training_data/1_1_data/phase_data/p12/all_p12_Phase9_eval_vqa.jsonl"
# a_file_path = "/home/jiangkailin/project/New_Knowledge/full_training_data/1_1_data/phase_data/p12/all_p12_Phase10_eval_vqa.jsonl"
# a_file_path = "/home/jiangkailin/project/New_Knowledge/full_training_data/1_1_data/phase_data/p12/all_p12_Phase11_eval_vqa.jsonl"
a_file_path = "/home/jiangkailin/project/New_Knowledge/full_training_data/1_1_data/phase_data/p12/all_p12_Phase12_eval_vqa.jsonl"


# b_file_path = "/home/jiangkailin/project/LLaVA/playground2/data/eval/eval_new_knowledge/answers/eval_vqa/llava-v1.5-7b-base-model/merge_eval_acc_f1.jsonl"  
b_file_path = "/home/jiangkailin/project/LLaVA/1_1_new_eval_result/new_knowledge/epoch/llava_lora/llava_ckpt_lora_epoch_8/merge_eval_acc_f1.jsonl"  
# b_file_path = "/home/jiangkailin/project/LLaVA/1_1_new_eval_result/new_knowledge/epoch/llava_full_ft/llava_ckpt_full_ft_epoch_6/merge_eval_acc_f1.jsonl"  


find_matching_and_calculate(a_file_path, b_file_path)

'''
1
/home/jiangkailin/project/New_Knowledge/full_training_data/phase_data/Phase1_eval_vqa.jsonl
/home/jiangkailin/project/LLaVA/1_1_new_eval_result/new_knowledge/PR/lora/llava_ckpt_PR1_lora_epoch_8/merge_eval_acc_f1.jsonl
2
/home/jiangkailin/project/New_Knowledge/full_training_data/phase_data/Phase2_eval_vqa.jsonl
/home/jiangkailin/project/LLaVA/1_1_new_eval_result/new_knowledge/PR/lora/llava_ckpt_PR2_lora_epoch_8/merge_eval_acc_f1.jsonl
3
/home/jiangkailin/project/New_Knowledge/full_training_data/phase_data/Phase3_eval_vqa.jsonl
/home/jiangkailin/project/LLaVA/1_1_new_eval_result/new_knowledge/PR/lora/llava_ckpt_PR3_lora_epoch_8/merge_eval_acc_f1.jsonl
4
/home/jiangkailin/project/New_Knowledge/full_training_data/phase_data/Phase4_eval_vqa.jsonl
/home/jiangkailin/project/LLaVA/1_1_new_eval_result/new_knowledge/PR/lora/llava_ckpt_PR4_lora_epoch_8/merge_eval_acc_f1.jsonl
100
/home/jiangkailin/project/LLaVA/1_1_new_eval_result/new_knowledge/epoch/llava_lora/llava_ckpt_lora_epoch_8/merge_eval_acc_f1.jsonl
base model
/home/jiangkailin/project/LLaVA/playground2/data/eval/eval_new_knowledge/answers/eval_vqa/llava-v1.5-7b-base-model/merge_eval_acc_f1.jsonl
'''

