
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



# model_path="/data2/sft/InternData/slz/LLaMA-Factory/models/llama3_sft_base_origin_checkpoint-3200"
model_path="/data2/sft/InternData/slz/Meta-Llama-3-70B-Instruct"
model_name = model_path.split('/')[-1]
save_path = "/data_generation/select_data/apaper_"+model_name

llm = LLM(model=model_path, tensor_parallel_size=8)
tokenizer = llm.get_tokenizer()


def zero_shot_eval_Naturalness(test_data, model_path):
    # llm = LLM(model=model_path, tensor_parallel_size=8)
    # tokenizer = llm.get_tokenizer()

    prompt = []
    
        
    # for i in range(0, 100):
    for i in range(0, len(test_data)):

        prompt_i_ = f"""You will be given a conversation between two individuals. You will then be given one potential response for the next turn in the conversation. \nYour task is to rate the responses on one metric. \nPlease make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.\n\nEvaluation Crieteria:\nNaturalness (1-3) Is the response naturally written?\n- A score of 1 (bad) means that theresponse is unnatural.\n- A score of 2 (ok) means the response is strange, but not entirely unnatural.\n- A score of 3 (good) means that the response is natural.\n\nEvaluation Steps:\n1. Read the conversation between the two individuals.\n2. Read the potential response for the next turn in the conversation.\n3. Evaluate the response based on its naturalness, using the provided criteria.\n4. Assign a rating score of 1, 2, or 3 based on the evaluation."""
        prompt_i = [{"role":"user", "content":prompt_i_}, {"role":"user", "content":f"""Conversation History:\n{test_data[i]["instruction"]}\n\nResponse:\n{test_data[i]["output"]}\n\nEvaluation Form (scores ONLY):\n- Naturalness:"""}]
        
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

        test_data[j]["Naturalness"] =response
    return test_data

def zero_shot_eval_Coherence(test_data, model_path):
    # llm = LLM(model=model_path, tensor_parallel_size=8)
    # tokenizer = llm.get_tokenizer()

    prompt = []

    
    # for i in range(0, 100):
    for i in range(0, len(test_data)):
        prompt_i_ = f"""You will be given a conversation between two individuals. You will then be given one potential response for the next turn in the conversation. \nYour task is to rate the responses on one metric. \nPlease make sure you read and understand these instructions carefully. Please keep this document open while reviewing,and refer to it as needed.\n\nEvaluation Crieteria:\nCoherence (1-3) Does the response serve as a valid continuation of the conversation history?\n- A score of 1 (no) means that the response drastically changes topic or ignores the conversation history.\n- A score of 2 (somewhat) means the response refers to the conversation history in a limited capacity (e.g., in a generic way) and shifts the conversation topic.\n- A score of 3 (yes) means the response is on topic and strongly acknowledges the conversation history.\n\nEvaluation Steps:\n1. Read the conversation history.\n2. Read the potential response.\n3. Evaluate the coherence of the response based on the conversation history.\n4. Assign a score of 1, 2, or 3 for coherence."""
        prompt_i = [{"role":"user", "content":prompt_i_}, {"role":"user", "content":f"""Conversation History:\n{test_data[i]["instruction"]}\n\nResponse:\n{test_data[i]["output"]}\n\nEvaluation Form (scores ONLY):\n- Coherence:"""}]

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

        test_data[j]["Coherence"] =response
    return test_data



def zero_shot_eval_Empathy(test_data, model_path):
    # llm = LLM(model=model_path, tensor_parallel_size=8)
    # tokenizer = llm.get_tokenizer()

    prompt = []

    # for i in range(0, 100):
    for i in range(0, len(test_data)):
        prompt_i_ = f"""You will be given a conversation between two individuals. You will then be given one potential response for the next turn in the conversation. \nYour task is to rate the responses on one metric. \nPlease make sure you read and understand these instructions carefully. Please keep this document open while reviewing,and refer to it as needed.\n\nEvaluation Crieteria:\nEmpathy (1-3) Does the response demonstrate empathy towards the user's feelings or situation?\n- A score of 1 (bad) means the response lacks empathy and does not acknowledge the user's feelings or situation.\n- A score of 2 (ok) means the response shows some understanding of the user's feelings or situation but could be more empathetic.\n- A score of 3 (good) means the response demonstrates a strong sense of empathy towards the user's feelings or situation and acknowledges them effectively.\n\nEvaluation Steps:\n1. Read the conversation history.\n2. Read the potential response.\n3. Evaluate the empathy of the response based on the conversation history.\n4. Assign a score of 1, 2, or 3 for empathy."""
        prompt_i = [{"role":"user", "content":prompt_i_}, {"role":"user", "content":f"""Conversation History:\n{test_data[i]["instruction"]}\n\nResponse:\n{test_data[i]["output"]}\n\nEvaluation Form (scores ONLY):\n- Empathy:"""}]
        
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

        test_data[j]["Empathy"] =response
    return test_data


if __name__ == '__main__':

    load_path = "/data_generation/filter_kcentergreeedy_embedscore60_llama3_sft_base_origin_checkpoint-3200_score_expandOriginData.json"
    test_data = load_file(load_path)
    print('** len: ', len(test_data))
    
    print("save_response_path: ", save_path)

    Naturalness_data = zero_shot_eval_Naturalness(test_data, model_path)
    save_file(Naturalness_data, save_path + "_Naturalness.json")
    print("** done Continuity_Naturalness")

    Coherence_data = zero_shot_eval_Coherence(Naturalness_data, model_path)
    save_file(Coherence_data, save_path + "_Naturalness_Coherence.json")
    print("** done Continuity_Naturalness_Coherence!")

    Empathy_data = zero_shot_eval_Empathy(Coherence_data, model_path)
    save_file(Empathy_data, save_path + "_Naturalness_Coherence_Empathy.json")
    print("** done!")


