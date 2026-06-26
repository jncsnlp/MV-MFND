from cProfile import label
from doctest import OutputChecker
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
#fakeddit
with open("/home/jncsnlp4/SSD2/tb/data/MR2-en/dataset_items_test_filtered.json",'r',encoding='utf-8') as file:
    data = json.load(file)


with open("/home/jncsnlp4/tb/Qwen2-VL-main/MR2_knowledev/text_knowledge_ours_MR2-en2.csv",mode='r',encoding="utf-8")as file:
    csv_dict_reader = csv.reader(file)
    rows = list(csv_dict_reader)

with open('/home/jncsnlp4/tb/prompt-llava/test_index/MR_test_text_max_index.txt','r',encoding='utf-8') as file:
    text_indexs = file.read()

with open('/home/jncsnlp4/tb/prompt-llava/test_index/MR_test_image_max_index.txt','r',encoding='utf-8') as file:
    image_indexs = file.read()
    
with open('/home/jncsnlp4/tb/Qwen2-VL-main/MR2_knowledev/image_knowledge_MR2-en2.csv',mode='r',encoding="utf-8")as file:
    csv_dict_reader = csv.reader(file)
    image_rows = list(csv_dict_reader)


def remove_punctuation_manual(text):
    return ''.join(char for char in text if char != '"'and char != "," and char != ":")

def get_label(final_output):
    output_text = final_output.split()
    for i, item in enumerate(output_text):
        if "label" in item:
            index = i
            break
    output = remove_punctuation_manual(output_text[index+1])
    return output

def get_certain(final_output):
    output_text = final_output.split()
    index = 0
    for i, item in enumerate(output_text):
        if "certain" in item:
            index = i
            break
    if index ==0:
        return "unsure"    
    else:
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
                    "temperature": 0.5,
                    "top_p": 0.95,
                    "top_k": 2,
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
                    "temperature": 0.5,
                    "top_p": 0.95,
                    "top_k = 2,": 2,
                    "do_sample": True, 
                    "max_new_tokens": 512 #512
                })()
        model_name = get_model_name_from_path(args.model_path)
        response,real_prob,fake_prob=eval_model(args,model_name,tokenizer, model, image_processor)
        # print("--------------------------------")
        # print("response",response)   
        if type(real_prob)==int:
            real_prob = tensor(real_prob)
        if type(fake_prob)==int:
            fake_prob =tensor(fake_prob)
        return response,real_prob,fake_prob
    except Exception as e:
        print(f"An error occurred: {e}")
        print('————————————prompt again————————————')
        return "wrong",0,0
    
def get_part(text,target,end_word):
    start = text.find(target) + len(target)
    remaining = text[start:]
    end = remaining.find(end_word)
    explanation_text = remaining[:end]

    return explanation_text
    
y = -1
fake_num = 0
real_num = 0
tp = 0
tn = 0
fp = 0
fn = 0
rtp=0
rtn=0
ftp=0
ftn=0
two_step_num = 0
one_step_true_num = 0
one_step_false_num = 0
t_f_num = 0
f_t_num = 0
cf_t_num = 0
ct_f_num = 0
text_indexs = ast.literal_eval(text_indexs)
image_indexs = ast.literal_eval(image_indexs)
print(len(text_indexs))
print(len(image_indexs))  
text_real_prob = tensor(0)  
text_fake_prob = tensor(0)
image_real_prob = tensor(0)  
image_fake_prob = tensor(0)
mm_real_logits = tensor(0)  
mm_fake_logits = tensor(0)
begin_time = datetime.now()
real_logits = []
fake_logits = []
labels = []
num = -1
trusted_website = {"https://en.wikipedia.org/","https://www.bbc.com/news","https://www.nytimes.com","https://www.theguardian.com","https://www.reuters.com","https://www.ap.org","https://www.wikipedia.org/"}

for item in data.values():
    image_path = "/home/jncsnlp4/SSD2/tb/data/MR2-en/" + item['image_path'] #fakeddit
    if os.path.exists(image_path):
        y = y + 1
        text = item['caption']
        label = int(item['label'])
        # MR2 
        if label == 1:
            label = "fake"
            num = num + 1
        elif label == 0:
            label = "real"
            num = num + 1     
        role = "politician"
        # print(role)
        text_prompt = ("Assume you are a helpful {} and you are given a piece of **Input Text**. Your task is to predict whether misinformation is present. \
        The text comes from a post (or a report). \
        By detecting whether text violates common sense, please predict whether this is a post containing misinformation.Please follow the Rules below:\n"
                "Rules:\n"
                "Generate a JSON object with three properties: 'label', 'explanation','certain'.\n" 
                "The return value of 'label' property should be selected from [\"real\", \"fake\"].\n"
                "real indicates that no misinformation is detected. \n"
                "fake indicates that misinformation is detected. \n"
                "The return value of 'explanation' property should be a detailed reasoning for the given 'label'. \n"
                "The return value of 'certain' property should be selected from [\"sure\",\"unsure\"].\n"
                "sure indicates that you are confident to your answer.\n"
                "unsure indicates that you are not confident to your answer.\n"
                "Note that 'label' and 'explanation' should be consistent in their judgment of whether or not the news contains misinformation.\n"
                "Note that your response will be passed to the python interpreter, SO NO OTHER WORDS! Just only output the json object. Do not add ```json in front of json object or behind it! \n"
                "Example output (JSON):\n"
                "{{\n"
                    "\"label\": ,\n"
                    "\"explanation\": ,\n"
                    "\"certain\":\n"
                "}}\n"
                "Input Text:\n"
                "{}\n"
                "Let's think step by step."
                "Your Response:\n").format(role,text)
        # print(text_prompt)
        image_prompt = ("Assume you are a helpful {} and you are given a piece of **Input image**. Your task is to predict whether misinformation is present. \
        The image comes from a post (or a report). \
        By detecting whether image violates common sense, please predict whether this is a post containing misinformation.Please follow the Rules below:\n"
                "Rules:\n"
                "Generate a JSON object with three properties: 'label', 'explanation','certain'.\n" 
                "The return value of 'label' property should be selected from [\"real\", \"fake\"].\n"
                "real indicates that no misinformation is detected. \n"
                "fake indicates that misinformation is detected. \n"
                "The return value of 'explanation' property should be a detailed reasoning for the given 'label'. \n"
                "The return value of 'certain' property should be selected from [\"sure\",\"unsure\"].\n"
                "sure indicates that you are confident to your answer.\n"
                "unsure indicates that you are not confident to your answer.\n"
                "Note that 'label' and 'explanation' should be consistent in their judgment of whether or not the news contains misinformation.\n"
                "Note that your response will be passed to the python interpreter, SO NO OTHER WORDS! Just only output the json object. Do not add ```json in front of json object or behind it! \n"
                "Example output (JSON):\n"
                "{{\n"
                    "\"label\": ,\n"
                    "\"explanation\": ,\n"
                    "\"certain\":\n"
                "}}\n"
                "Let's think step by step."
                "Your Response:\n").format(role)

        role_response_prompt = ("Assume you are a helpful{} and you are given a piece of **Input Text** and an image. Your task is to predict whether misinformation is present in them based on your professional knowledge.\
        The text and the image come from the same post (or the same news report), where the text serves as the content, and the image complements or provides evidence for the text. \
        Please predict whether this is a post containing misinformation by verifying the consistency of text and images and detecting whether text and images violate common sense.\
        You will be punished if your answer is wrong. Please follow the Rules below:\n"
                    "Rules:\n"
                    "Generate a JSON object with three properties: 'label', 'explanation','certain'.\n" 
                    "The return value of 'label' property should be selected from [\"real\", \"fake\"].\n"
                    "real indicates that no misinformation is detected. \n"
                    "fake indicates that misinformation is detected. \n"
                    "The return value of 'explanation' property should be a detailed reasoning for the given 'label'. \n"
                    "The return value of 'certain' property should be selected from [\"sure\",\"unsure\"].\n"
                    "sure indicates that you are confident to your predict.\n"
                    "unsure indicates that you are not confident to your predict.\n"
                    "Note that your response will be passed to the python interpreter, SO NO OTHER WORDS! Just only output the json object. Do not add ```json in front of json object or behind it! \n"
                    "Example output (JSON):\n"
                    "{{\n"
                    "\"label\": ,\n"
                    "\"explanation\": ,\n"
                    "\"certain\":\n"
                    "}}\n"
                    "Input Text:\n"
                    "{}\n"
                    "Let's think step by step."
                    "Your Response:\n").format(role,text)
        # text_output1,text_real_prob,text_fake_prob = genai_chat_completion_response(image_path,text_prompt,types="text")
        # image_output1,image_real_prob,image_fake_prob= genai_chat_completion_response(image_path,image_prompt,types="mm")
        mm_output1,mm_real_logits,mm_fake_logits = genai_chat_completion_response(image_path,role_response_prompt,types="mm")
        # print(text_output)

        # text_label1 = get_label(text_output1)
        # image_label1 = get_label(image_output1)
        mm_label1 = get_label(mm_output1)

        # text_ = get_certain(text_output1)
        # image_ = get_certain(image_output1)
        mm_ = get_certain(mm_output1)
        # print(f"文本标签：{text_label1}，图像标签：{image_label1}，角色{role}的mm标签：{mm_label1}")
        # print(f"文本real概率：{text_real_prob}，图像real概率：{image_real_prob}，角色{role}的mmreal概率：{mm_real_logits}")
        # print(f"文本fake概率：{text_fake_prob}，图像fake概率：{image_fake_prob}，角色{role}的mmfake概率：{mm_fake_logits}")
        weather_knowledge = 0

        text_weather_knowledge = 1
        image_weather_knowledge = 1
        mm_weather_knowledge = 1

        text_index = text_indexs[y]
        image_index = image_indexs[y]
        
        knowledge = rows[y][1]
        image_knowledge = image_rows[y][1]
        
        text_extra_knowledge = ast.literal_eval(knowledge)
        if image_knowledge == "":
            image_extra_knowledge = []
        else:
            image_extra_knowledge = ast.literal_eval(image_knowledge)
        final_text_knowledge = text_extra_knowledge[text_index]['body']    
        print(f"源文本:{text}")
        print(f"文本知识:{final_text_knowledge}")
        if len(image_extra_knowledge) > 0:
            final_image_knowledge = image_extra_knowledge[image_index]['body']
        else:
            final_image_knowledge = ""
        print(f"图像知识:{final_image_knowledge}")
        # if max(text_real_prob,text_fake_prob)<0.6 or text_max_similarity > 1.6 :
        text_output = ""
        image_output = ""
        mm_output = ""
        # if text_ == "unsure" :    
        #     print("文本模态不确定，需要外部知识")
        #     text_prompt = ("A language model was asked: Predict whether the news is real or fake according to the text of the news.The text of the news is {}.\
        #     The model's answer was: {}.\n"
        #     "The model is not very sure of its answer. Give you an additional paragraph of external knowledge about the news that \
        #     is absolutely correct:{}, and your task is to rethink with the help of external knowledge and predict whether this is a news containing misinformation.\n"
        #     "Please follow the Rules below:\n"
        #             "Rules:\n"
        #             "Generate a JSON object with two properties: 'label', 'explanation'.\n" 
        #             "The return value of 'label' property should be selected from [\"real\", \"fake\"].\n"
        #             "real indicates that no misinformation is detected. \n"
        #             "fake indicates that misinformation is detected. \n"
        #             "The return value of 'explanation' property should be a detailed reasoning for the given 'label'. \n"
        #             "Note that 'label' and 'explanation' should be consistent in their judgment of whether or not the news contains misinformation.\n"
        #             "Note that your response will be passed to the python interpreter, SO NO OTHER WORDS! Just only output the json object. Do not add ```json in front of json object or behind it! \n"
        #             "Example output (JSON):\n"
        #             "{{\n"
        #             "\"label\": ,\n"
        #             "\"explanation\":\n"
        #             "}}\n"
        #             "Let's think step by step."
        #             "Your Response:\n").format(text,text_output1,final_text_knowledge)
            
        #     text_output,text_real_prob,text_fake_prob = genai_chat_completion_response(image_path,text_prompt,types="text")
        #     text_label = get_label(text_output)
        #     text_label1 = text_label

        # if max(image_real_prob,image_fake_prob)<0.6 or image_max_similarity > 1.6:
        # if image_ == "unsure":
        #     print("图像模态不确定，需要外部知识")
        #     image_prompt = ("A language model was asked: Predict whether the news is real or fake according to the image of the news.\
        #     The model's answer was: {}.\n"
        #     "The model is not very sure of its answer. Give you an additional paragraph of external knowledge about the news that \
        #     is absolutely correct:{}, and your task is to rethink with the help of external knowledge and predict whether this is a news containing misinformation.\n"
        #     "Please follow the Rules below:\n"
        #             "Rules:\n"
        #             "Generate a JSON object with two properties: 'label', 'explanation'.\n" 
        #             "The return value of 'label' property should be selected from [\"real\", \"fake\"].\n"
        #             "real indicates that no misinformation is detected. \n"
        #             "fake indicates that misinformation is detected. \n"
        #             "The return value of 'explanation' property should be a detailed reasoning for the given 'label'. \n"
        #             "Note that 'label' and 'explanation' should be consistent in their judgment of whether or not the news contains misinformation.\n"
        #             "Note that your response will be passed to the python interpreter, SO NO OTHER WORDS! Just only output the json object. Do not add ```json in front of json object or behind it! \n"
        #             "Example output (JSON):\n"
        #             "{{\n"
        #             "\"label\": ,\n"
        #             "\"explanation\":\n"
        #             "}}\n"
        #             "Let's think step by step."
        #             "Your Response:\n").format(image_output1,final_image_knowledge)
        #     image_output,image_real_prob,image_fake_prob = genai_chat_completion_response(image_path,text_prompt,types="mm")
        #     image_label = get_label(image_output)
        #     image_label1 = image_label    

        # if max(mm_real_logits,mm_fake_logits) < 0.6 or text_max_similarity > 1.6 or image_max_similarity > 1.6:
        if mm_ == "unsure" :
            print("多模态不确定，需要外部知识")
            final_mm_knowledge = "1."+final_text_knowledge + " 2." + final_image_knowledge
            mm_prompt = ("A language model was asked: Predict whether the news is real or fake according to the text and image of the news.The text of the news is {}.\
            The model's answer was: {}.\n"
            "The model is not very sure of its answer. Give you an additional paragraph of external knowledge about the news that \
            is absolutely correct:{}, and your task is to rethink with the help of external knowledge and predict whether this is a news containing misinformation.\n"
            "Please follow the Rules below:\n"
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
                    "Let's think step by step."
                    "Your Response:\n").format(text,mm_output1,final_mm_knowledge)
            mm_output,mm_real_logits,mm_fake_logits = genai_chat_completion_response(image_path,text_prompt,types="mm")
            mm_label = get_label(mm_output)
            mm_label1 = mm_label

        # current_label_list = [text_label1,image_label1,mm_label1]   
        # index_list = ["text","image","mm"]
        current_real_logits = [text_real_prob.item(),image_real_prob.item(),mm_real_logits.item()]
        current_fake_logits = [text_fake_prob.item(),image_fake_prob.item(),mm_fake_logits.item()]
        print(f"现在是第{y}条数据")

        final_real_prob = 0.3*text_real_prob.item()+0.4*image_real_prob.item()+0.3*mm_real_logits.item()
        final_fake_prob = 0.3*text_fake_prob.item()+0.4*image_fake_prob.item()+0.3*mm_fake_logits.item()
        final_label = "real" if final_real_prob>final_fake_prob else "fake"
        # final_label = mm_label1
        if final_label == label:
            if label == "real":
                tp = tp + 1
                print("111")
            else:
                tn = tn + 1
                print("222")
        else:
            if label == "fake":
                fp = fp + 1
                print("333")
            else:
                fn = fn + 1 
                print("444")       
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
# print(f"纠正模块中进行纠正的数量：{ct_f_num+cf_t_num}，其中把对的改成错的数量:{ct_f_num}，把错的改成对的数量:{cf_t_num}")
print(f"开始时间:{begin_time}\n结束时间:{end_time}")  