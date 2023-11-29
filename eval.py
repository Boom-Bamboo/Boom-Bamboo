import os
import torch
import platform
import subprocess
import json
from tqdm import tqdm
import re
from base import *
from sat import get_args

from chat_model import ChatModel

from accelerate import Accelerator


# accelerator = Accelerator()


def init_model():
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
    args=argparse.Namespace(fp16=True, skip_init=True, use_gpu_initialization=True, device='cuda')
    model = ChatModel(args)
    # model = accelerator.prepare(model)
    model = model.eval()
    return model, tokenizer

def main():
    model, tokenizer = init_model()
    
    with open('CMB-test-choice-question-merge.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    results = []
    for i in tqdm(range(len(data))):
        test_id = data[i]['id']
        exam_type = data[i]['exam_type']
        exam_class = data[i]['exam_class']
        question_type = data[i]['question_type']
        question = data[i]['question']
        option_str = 'A. ' + data[i]['option']['A'] + '\nB. ' + data[i]['option']['B']+ '\nC. ' + data[i]['option']['C']+ '\nD. ' + data[i]['option']['D']
        
        prompt = f'以下是中国{exam_type}中{exam_class}考试的一道{question_type}，不需要做任何分析和解释，直接输出答案选项。\n{question}\n{option_str}'
        
        response, history = model.chat(tokenizer, prompt, history=[])
        
        #print(response)
        matches = re.findall("[ABCDE]", response)
        final_result = "".join(matches)
        
        info = {
                "id": test_id,
                "model_answer": final_result
            }
        results.append(info)

    with open('output.json', 'w', encoding="utf-8") as f1:
        json.dump(results, f1, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
