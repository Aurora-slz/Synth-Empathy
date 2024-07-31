
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cuda" # the device to load the model onto

# from vllm import LLM,SamplingParams
import json
import random
import math 

def load_file(load_path):
    with open(load_path, 'r', encoding='utf-8') as f1:
        data = json.load(f1)
        print(data[0])
    return data

def save_file(data, save_path):
    with open(save_path, 'w', encoding='utf-8') as f1:
        f1.write(json.dumps(data))

def get_conversation_expand(context):
    new_con = ""
    for i in range(0, len(context)):
        
        # new_con += "Speaker: {}\n".format(context[i]["content"])
        new_con += "Speaker: {}\n".format(context[i]["Context"])
    
        # new_con += "Listener: {}\n".format(context[i]["content"])
        new_con += "Listener: {}\n".format(context[i]["Response"])
    # prompt = "<s>[INST] <<SYS>>\nYou are a professional psychologist. Please give a reply from the perspective of a friend based on the patient's situation. Be careful to avoid a large number of professional vocabulary in your reply, and the expression should be natural.\nThe following is a set of historical conversations between you and your best friend. Please respond according to the context and pay attention to fully consider the emotional information in the historical conversations. Remember, response should not exceed 30 words.\n<</SYS>>\n\n{}[/INST]".format(new_con)
    # prompt = "{}".format(new_con)
    return new_con

import random
import matplotlib.pyplot as plt

def view(random_list):

    
    plt.hist(random_list, bins=7, edgecolor='black')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Random Integers')
    plt.xticks([1, 2, 3,4,5,6,7])  

    plt.savefig('/select_data/dialog_turn_distribution_new_70b_race23.png')


def sample_turn(data):
    view_turn = []
    ori_data = []
    for i in range(0, len(data)):

        random_integer = min(random.randint(1, 3), len(data[i]) - 1)
        view_turn.append(random_integer)
        # print('** data[i]: ', len(data[i]))
        # print('** random_integer: ', random_integer)
        context = get_conversation_expand(data[i][:random_integer])
        context += "Speaker: {}\n".format(data[i][random_integer]["Context"])
        context += "Listener: "
        response = data[i][-1]["Response"]

        tmp = {}
        tmp["instruction"] = context
        tmp["input"] = ''
        tmp["output"] = response
        ori_data.append(tmp)
    
    view(view_turn)


    return ori_data



if __name__ == '__main__':
    load_path = "/data/processed_contexts_responses_70b_race3.json"
    test_data = load_file(load_path)
    
    sample_data = sample_turn(test_data)
    random.shuffle(sample_data)
    print(len(sample_data))
    save_file(sample_data, "/select_data/expand_turn_data_70b_race3.json")

    