import json

def copy_type_to_b(file_a, file_b, output_file):
    # 加载A文件数据到字典中（基于question_id索引）
    type_mapping = {}
    with open(file_a, 'r', encoding='utf-8') as file_a_data:
        for line in file_a_data:
            data = json.loads(line)
            question_id = data.get("question_id")
            type_value = data.get("type")
            if question_id is not None and type_value is not None:
                type_mapping[question_id] = type_value

    # 加载B文件数据并更新type字段
    updated_b_data = []
    with open(file_b, 'r', encoding='utf-8') as file_b_data:
        for line in file_b_data:
            data = json.loads(line)
            question_id = data.get("question_id")
            if question_id in type_mapping:
                # 如果question_id匹配，将A文件的type复制到B文件
                data["type"] = type_mapping[question_id]
            updated_b_data.append(data)

    # 保存更新后的B文件
    with open(output_file, 'w', encoding='utf-8') as output_file_data:
        for data in updated_b_data:
            output_file_data.write(json.dumps(data, ensure_ascii=False) + '\n')

# 使用函数
file_a = "/home/jiangkailin/project/LLaVA/playground/data/eval/evalwiki/answers/12_15_eval_wikidata_vqa/merge_2_epoch_10_wikidata-llava-v1.5-7b-task-lora/merge_eval.jsonl"  # 替换为A文件的路径
file_b = "/home/jiangkailin/project/LLaVA/playground/data/eval/evalwiki/answers/12_15_eval_wikidata_vqa/merge_2_wikidata-llava-v1.5-7b-task-lora/merge_eval.jsonl"  # 替换为B文件的路径
output_file = "/home/jiangkailin/project/LLaVA/playground/data/eval/evalwiki/answers/12_15_eval_wikidata_vqa/merge_2_wikidata-llava-v1.5-7b-task-lora/merge_eval_type.jsonl"  # 替换为输出B文件的路径

copy_type_to_b(file_a, file_b, output_file)
