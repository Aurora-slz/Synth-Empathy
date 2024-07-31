
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
device = "cuda" # the device to load the model onto

from vllm import LLM,SamplingParams
import json
import random

def load_file(load_path):
    with open(load_path, 'r', encoding='utf-8') as f1:
        data = json.load(f1)
        # print(data[0])
    return data

def save_file(data, save_path):
    with open(save_path, 'w', encoding='utf-8') as f1:
        f1.write(json.dumps(data))



model_path="/llama3_sft_base_origin_checkpoint-3200"
model_name = model_path.split('/')[-1]
save_path = "/select_data/apaper_"+model_name

llm = LLM(model=model_path, tensor_parallel_size=8)
tokenizer = llm.get_tokenizer()




def zero_shot_eval_model(test_data, model_path):
    # llm = LLM(model=model_path, tensor_parallel_size=8)
    # tokenizer = llm.get_tokenizer()

    prompt = []

    # for i in range(0, 100):
    for i in range(0, len(test_data)):
        
        prompt_i = [{"role":"user", "content":f"""{test_data[i]["instruction"]}"""}]
        input = [{"role":"system", "content":"You are a helpful assistant."}] + prompt_i
        tmp = tokenizer.apply_chat_template(input, tokenize=False, add_generation_prompt=True)
        prompt.append(tmp) 
        # print(tmp)
    print(prompt[0])   
    print(prompt[1])
    print(prompt[2])
    print('** len: ', len(prompt))
    
    sampling_params = SamplingParams(temperature=0.8, top_p=0.9, max_tokens=128, stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")])
    outputs = llm.generate(prompt, sampling_params)
    
    # for output in outputs:
    for j in range(0, len(outputs)):
        output = outputs[j]
        prompt = output.prompt
        response = output.outputs[0].text
        # if(response[0] == "\"" and response[-1] == "\""):
        #     response = response[1:-1]
        
        # print(response)
        # print('='*10)
        test_data[j]["llama_response"] =response
    
    return test_data

def zero_shot_eval_embedscore(test_data):

    model = SentenceTransformer("/gte-Qwen2-7B-instruct", trust_remote_code=True)
    model.max_seq_length = 8192
    
    
    for i in range(0, len(test_data)):
    # for i in range(0, 10):
        conversations = []
        conversations.append(test_data[i]['output'])
        conversations.append(test_data[i]['llama_response']) 

        conversation_embeddings = model.encode(conversations, prompt_name="query")
        scores = (conversation_embeddings[0] @ conversation_embeddings[1].T) * 100
        test_data[i]['embed_score'] = scores

        if(i % 1000 == 0):
            print(i)
        # print('='*10)
        # print(conversations)
        # print('score: ', scores)
    return test_data


if __name__ == '__main__':

    
    load_path = "/select_data/expand_turn_data_70b_race3.json"
    test_data = load_file(load_path)
    print('** len: ', len(test_data))
    print("save_response_path: ", save_path)
    model_data = zero_shot_eval_model(test_data, model_path)
    save_file(model_data, save_path + "_score_expandOriginData_70b_race3.json")
    print("** done model_data")


    # close vllm to reduce GPU memory
    load_path = "/select_data/llama3_sft_base_origin_checkpoint-3200_score_expandOriginData_70b_race3.json"
    test_data = load_file(load_path)
    embedscore_data = zero_shot_eval_embedscore(test_data)
    save_file(embedscore_data, "/select_data/llama3_sft_base_origin_checkpoint-3200_score_expandOriginData_70b_race23.json")


