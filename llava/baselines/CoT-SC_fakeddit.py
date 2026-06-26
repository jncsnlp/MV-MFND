from cProfile import label
from doctest import OutputChecker
from fastapi import types
from numpy import real
import pandas as pd 
import sys
import os
import csv
import json
from sympy import im
from torch import ne, tensor
import json
import os
import gc
import torch
import string
import requests
import csv
import ast
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '/home/jncsnlp4/tb/LLaVA-main')))
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

model_path = "/home/jncsnlp4/SSD2/model/llava-v1.5-7b"
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path),
    device_map = 'cuda'
)

with open("/home/jncsnlp4/tb/prompt-llava/FAKEDDIT.json",'r',encoding='utf-8') as file:
    data = json.load(file)
#twitter
# with open("/home/jncsnlp4/tb/LEMMA-main/data/twitter/twitter.json",'r',encoding='utf-8') as file:
#     data = json.load(file)    

def remove_punctuation_manual(text):
    return ''.join(char for char in text if char != '"'and char != "," and char != ":")

def get_label(final_output):
    output_text = final_output.split()
    for i, item in enumerate(output_text):
        if "label" in item:
            index = i
            output = remove_punctuation_manual(output_text[index+1])
            break
        output = "fake"    
    return output

def get_label2(final_output):
    output_text = final_output.split()
    output_text = output_text[-2] + output_text[-1]
    print(output_text)
    output = "real" if "real" in output_text.lower() else "fake"
    return output

def get_certain(final_output):
    output_text = final_output.split()
    for i, item in enumerate(output_text):
        if "certain" in item:
            index = i
            break
    output = remove_punctuation_manual(output_text[index+1])
    return output


def genai_chat_completion_response(image_file,prompt,types):
    try:
        if types == "mm":
            args = type('Args', (), {
                    "model_path": model_path,
                    "model_base": None,
                    "model_name": get_model_name_from_path(model_path),
                    "query": prompt,
                    "conv_mode": None,
                    "image_file": image_file,
                    "sep": ",",
                    "temperature": 1.0,
                    "top_p": 0.95,
                    "num_beams": 1,
                    "do_sample": True, 
                    "max_new_tokens": 512 #512
                })()
        else:
            args = type('Args', (), {
                    "model_path": model_path,
                    "model_base": None,
                    "model_name": get_model_name_from_path(model_path),
                    "query": prompt,
                    "conv_mode": None,
                    "image_file": "none",
                    "sep": ",",
                    "temperature": 1.0,
                    "top_p": 0.95,
                    "num_beams": 1,
                    "do_sample": True, 
                    "max_new_tokens": 512 #512
                })()
        model_name = get_model_name_from_path(args.model_path)
        response,real_prob,fake_prob=eval_model(args,model_name,tokenizer, model, image_processor)
        # print("--------------------------------")
        # print("response",response)   
        
        return response,real_prob,fake_prob
    except Exception as e:
        print(f"An error occurred: {e}")
        print('————————————prompt again————————————')
        return "wrong",0,0

y = -1
fake_num = 0
real_num = 0
tp = 0
tn = 0
fp = 0
fn = 0
two_step_num = 0
one_step_true_num = 0
one_step_false_num = 0
t_f_num = 0
f_t_num = 0
cf_t_num = 0
ct_f_num = 0
true_knowledeg_list = []
begin_time = datetime.now()
real_logits = []
fake_logits = []
labels = [] 
for x in range(len(data)):
    image_path = "/home/jncsnlp4/tb/prompt-llava/picture/post" + str(x) + ".jpg" #fakeddit
    
    if os.path.exists(image_path):
        y = y + 1
        number = x
        text = data[x]['original_post']
        label = data[x]['label']
        if label == 1:
            label = "fake"
            fake_num = fake_num + 1
        else:
            label = "real"
            real_num = real_num + 1
        label_list = []
        for i in range(5):
            role_response_prompt = ("You are given a piece of **Input Text** and an image. Your task is to predict whether misinformation is present in them. \
            The text and the image come from the same post (or the same news report), where the text serves as the content, and the image complements or provides evidence for the text, \
            Please predict whether this is a post containing misinformation by verifying the consistency of text and image and detecting whether text and image violate common sense. \
            You will be punished if your answer is wrong. Please follow the Rules below:\n"
                        "Rules:\n"
                        "Generate a JSON object with two properties: 'label', 'explanation'.\n" 
                        "The return value of 'label' property should be selected from [\"real\", \"fake\"].\n"
                        "real indicates that no misinformation is detected. \n"
                        "fake indicates that misinformation is detected. \n"
                        "The return value of 'explanation' property should be a detailed reasoning for the given 'label'. \n"
                        "Note that 'label' and 'explanation' should be consistent in their judgment of whether or not the news contains misinformation.\n"
                        "Note that your response will be passed to the python interpreter, SO NO OTHER WORDS! Just only output the json object. Do not add ```json in front of json object or behind it! \n"
                        "Example output (JSON):\n"
                        "{{\n"
                        "\"label\": ,\n"
                        "\"explanation\":\n"
                        "}}\n"
                        "Input Text:\n"
                        "{}\n"
                        "Let's think step by step."
                        "Your Response:\n").format(text)
            mm_output1,mm_real_logits,mm_fake_logits = genai_chat_completion_response(image_path,role_response_prompt,types="mm")
            print(mm_output1)
            mm_label1 = get_label(mm_output1)
            label_list.append(mm_label1)
            # print(2222,mm_output1)
        real_ = 0
        fake_ = 0
        for i in range(len(label_list)):
            if label_list[i] == "real":
                real_ = real_ + 1 
            else:
                fake_ = fake_ + 1
        
        final_label = "real" if real_ > fake_ else "fake"
        # print(y)
        print(f"第{x}条数据,原文本{text},\n原label:{label}\n最终输出label:{final_label}")
        
        if final_label == label:
            if label == "real":
                tp = tp + 1
            else:
                tn = tn + 1
        else:
            if label == "fake":
                fp = fp + 1
            else:
                fn = fn + 1
    else:
        continue
end_time = datetime.now()    
print(f"accuracy:{(tp+tn)/(tp+tn+fp+fn)}")
print(f"precision:{tp/(tp+fp)}")
print(f"recall:{tp/(tp+fn)}")
print(f"f1:{2*((tp/(tp+fp))*(tp/(tp+fn)))/(tp/(tp+fp)+tp/(tp+fn))}")
print(f"预测对的数量：{tp+tn}--预测错误的数量{fp+fn}")
print(f"数据集中real数量:{real_num}fake数量:{fake_num}")
print(f"预测中real数量:{tp+fp}fake数量:{tn+fn}")
print(f"预测对的中real的数量:{tp}fake的数量:{tn}")
print(f"预测错的中real的数量:{fp}fake的数量:{fn}")  
print(f"两步预测中进入第二步的数量:{one_step_true_num+one_step_false_num}，其中预测对的数量:{one_step_true_num}，预测错的数量:{one_step_false_num}")
print(f"第二步预测中预测对的改成错的数量:{t_f_num}，错的改成对的数量:{f_t_num}")
print(f"纠正模块中进行纠正的数量：{ct_f_num+cf_t_num}，其中把对的改成错的数量:{ct_f_num}，把错的改成对的数量:{cf_t_num}")
print(f"开始时间:{begin_time}\n结束时间:{end_time}")   