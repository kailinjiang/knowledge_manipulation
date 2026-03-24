import json
from tools import VQAEval

eval_tool = VQAEval()

def evaluate_vqa(a_file, b_file, output_file):

    total_score = 0
    num_entries = 0

    with open(a_file, 'r', encoding='utf-8') as af:

        a_entries = [json.loads(line) for line in af]
        
    
        a_data = {entry['question_id']: entry['answer'] for entry in a_entries}
        a_data_type = {entry['question_id']: entry['type'] for entry in a_entries}


    updated_b_data = []

    with open(b_file, 'r', encoding='utf-8') as bf:
        for line in bf:
            b_entry = json.loads(line)
            qid = b_entry['question_id']
            pred_answer = b_entry['text']
            gt_answer = a_data.get(qid, "")  
            type = a_data_type.get(qid, "")  

            score = eval_tool.evaluate(pred_answer, [gt_answer])
            b_entry['score'] = score
            b_entry['gt_answer'] = gt_answer  
            b_entry['type'] = type
            
            total_score += score
            num_entries += 1

            updated_b_data.append(b_entry)

    avg_score = total_score / num_entries if num_entries > 0 else 0

    with open(output_file, 'w', encoding='utf-8') as out_f:
        for entry in updated_b_data:
            out_f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"\n sum_score: {total_score}, len_data: {num_entries}")
    print(f"average_score: {avg_score:.4f}")
    return avg_score


a_file_path = "/home/jiangkailin/project/New_Knowledge/full_training_data/1_1_data/eval_vqa.jsonl"  
b_file_path = "/home/jiangkailin/project/LLaVA/playground/data/eval/eval_new_knowledge/answers/eval_vqa/checkpoint-2475/merge.jsonl"  
output_file_path = "/home/jiangkailin/project/LLaVA/playground/data/eval/eval_new_knowledge/answers/eval_vqa/checkpoint-2475/merge_eval.jsonl"  


evaluate_vqa(a_file_path, b_file_path, output_file_path)
