

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cuda" # the device to load the model onto

# from vllm import LLM,SamplingParams
import json
import random
import matplotlib.pyplot as plt
from collections import Counter


def load_file(load_path):
    with open(load_path, 'r', encoding='utf-8') as f1:
        data = json.load(f1)
        print(data[0])
    return data


def save_file(data, save_path):
    with open(save_path, 'w', encoding='utf-8') as f1:
        f1.write(json.dumps(data))

def filter_embedscore(data, YUZHI):

    score = []
    fin_data =  []
    for i in range(0, len(data)):
        s = int(data[i]['embed_score'])
        score.append(int(s / 10))
        if(s >= YUZHI):
            fin_data.append(data[i])
    
    x = range(0, 11, 1)
    
    plt.hist(score, bins=range(0,11), edgecolor='black', align='left')
    # plt.bar(x, score, width=8, align='center')

    plt.title('Data Distribution')
    plt.xlabel('Range (0-100, step 10)')
    plt.ylabel('Values')
    plt.savefig("select_data/embed_socre_70b_race32.png")

    return fin_data





if __name__ == '__main__':
    load_path = "select_data/llama3_sft_base_origin_checkpoint-3200_score_expandOriginData_70b_race3.json"
    test_data = load_file(load_path)
    fin_data = filter_embedscore(test_data, 50)
    print('** len-fin: ', len(fin_data))
    save_file(fin_data, "select_data/filter_embedscore50_llama3_sft_base_origin_checkpoint-3200_score_expandOriginData_70b_race3.json")
    
    