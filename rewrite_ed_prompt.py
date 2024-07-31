import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cuda" # the device to load the model onto

from vllm import LLM,SamplingParams
import json
import random
import re

def load_file(load_path):
    with open(load_path, 'r', encoding='utf-8') as f1:
        data = json.load(f1)
        print(data[0])
    return data

def save_file(data, save_path):
    with open(save_path, 'w', encoding='utf-8') as f1:
        f1.write(json.dumps(data))

prompt_template = """
Original Context: {context}
Original Response: {response}

You are an empathetic psychologist with powerful sensibility and rationality. Please generate at least three rounds of completely new emotional context and corresponding empathetic response. The newly generated multi-round context should satisfy semantic continuity. The response should be empathetic and should not exceed 30 words.

Round:
Context:
Response:
"""

load_path = "/data/train_results.json"
test_data = load_file(load_path)

model_path="/data2/sft/InternData/slz/Meta-Llama-3-70B-Instruct"
model_name = model_path.split('/')[-1]

llm = LLM(model=model_path, tensor_parallel_size=8)
tokenizer = llm.get_tokenizer()

def zero_shot_eval(test_data, save_response_path, model_path):
    
    prompt = []
    with open(save_response_path, 'w', encoding='utf-8') as f1:
        
        # for i in range(0, 100):
        for i in range(0, len(test_data)):
            prompt_i = prompt_template.format(
                context=test_data[i]["context"], 
                response=test_data[i]["response"]
            )
            tmp = tokenizer.apply_chat_template(
                [{'role': 'user', 'content': prompt_i}],
                tokenize=False,
            )
            prompt.append(tmp) 
        print(prompt[0])   
        print(prompt[1])
        print(prompt[2])
        print('** len: ', len(prompt))


        sampling_params = SamplingParams(temperature=0.8, top_p=0.9, max_tokens=300, stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")])
        outputs = llm.generate(prompt, sampling_params)
        
        # for output in outputs:
        res_data = []
        for j in range(0, len(outputs)):
            output = outputs[j]
            prompt = output.prompt
            response = output.outputs[0].text
            tmp = {}
            tmp["tmp"] = str(response)
            res_data.append(tmp)
            
            # try:
                # pattern = r"Round (\d+):\s*Context: (.*?)\s*Response: (.*?)(?=\nRound |\Z)"
                # matches = re.findall(pattern, response, re.DOTALL)
                # data = [{"Round": match[0].strip(), "Context": match[1].strip(), "Response": match[2].strip()} for match in matches]
            #     res_data.append(data)
            # except Exception as e:
            #     print("miss")

        save_file(res_data, save_response_path)

if __name__ == '__main__':
   
    for i in range(1,6):
        save_response_path = f"/data_generation/aaa-generated_responses_round_race3_{i}_sensibilityRationality.json"
        print("save_response_path: ", save_response_path)
        zero_shot_eval(test_data, save_response_path, model_path)


    r1 = load_file("/data_generation/generated_responses_round_race3_1_sensibilityRationality.json")
    r2 = load_file("/data_generation/generated_responses_round_race3_2_sensibilityRationality.json")
    r3 = load_file("/data_generation/generated_responses_round_race3_3_sensibilityRationality.json")
    r4 = load_file("/data_generation/generated_responses_round_race3_4_sensibilityRationality.json")
    r5 = load_file("/data_generation/generated_responses_round_race3_5_sensibilityRationality.json")

    data = r1+r2+r3+r4+r5
    print(len(r1))
    print(len(r2))
    print(len(r3))
    print(len(r4))
    print(len(r5))
    print(len(data))

    fin_data = []
    for i in range(0, len(data)):
        pattern = r"Context: (.+?)\nResponse: (.+?)(?=(\n|$))"
        matches = re.findall(pattern, data[i]['tmp'], re.DOTALL)
        mat = [{"Context": match[0].strip(), "Response": match[1].strip()} for match in matches]
        if(len(mat) <= 0):
            # print(mat)
            print(data[i]['tmp'])
            print('=' * 10)
        else:
            fin_data.append(mat)
    
    print(len(fin_data))
    random.shuffle(fin_data)

    save_file(fin_data, "/data/processed_contexts_responses_70b_race3.json")