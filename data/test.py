import json

def generate_repeated_json(count, output_file="output.json"):
    # 原始单条数据内容
    single_item = {
        "messages": [
            {
                "content": "<image>Who are they?",
                "role": "user"
            },
            {
                "content": "They're Kane and Gretzka from Bayern Munich.",
                "role": "assistant"
            }
        ],
        "images": [
            "mllm_demo_data/1.jpg"
        ],
        "rats": "They're Kane and Gretzka from Bayern Munich."
    }

    # 使用列表推导式复制指定次数
    # 注意：这里使用了深拷贝的逻辑（通过json转换或直接生成），确保每条数据独立
    result_list = [single_item for _ in range(count)]

    # 导出为带缩进的 JSON 文件
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            # indent=4 实现缩进，ensure_ascii=False 保证中文/特殊字符显示正常
            json.dump(result_list, f, indent=4, ensure_ascii=False)
        print(f"成功导出！已复制 {count} 遍数据至: {output_file}")
    except Exception as e:
        print(f"导出失败: {e}")

if __name__ == "__main__":
    # --- 你可以在这里修改参数 ---
    repeat_times = 100  # 想要复制的遍数
    file_name = "/data/jiangkl/project/LlamaFactory-main/data/mllm_demo_test2.json" # 输出文件名
    
    generate_repeated_json(repeat_times, file_name)