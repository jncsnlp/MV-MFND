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
with open("/home/jncsnlp4/tb/Qwen2-VL-main/train_fakeddit/train_data2.csv",'r',encoding='utf-8') as file:
    csv_dict_reader = csv.reader(file)
    datarows = list(csv_dict_reader)


with open("/home/jncsnlp4/tb/Qwen2-VL-main/train_fakeddit/text_knowledge.csv",mode='r',encoding="utf-8")as file:
    csv_dict_reader = csv.reader(file)
    text_rows = list(csv_dict_reader)


with open('/home/jncsnlp4/tb/Qwen2-VL-main/train_fakeddit/image_knowledge.csv',mode='r',encoding="utf-8")as file:
    csv_dict_reader = csv.reader(file)
    image_rows = list(csv_dict_reader)


with open('/home/jncsnlp4/tb/prompt-llava/fakeddittrain/fakeddit_text_max_index.txt','r',encoding='utf-8') as file:
    text_indexs = file.read()

with open('/home/jncsnlp4/tb/prompt-llava/fakeddittrain/fakeddit_image_max_index.txt','r',encoding='utf-8') as file:
    image_indexs = file.read()

with open('/home/jncsnlp4/tb/Qwen2-VL-main/train_fakeddit/text_question.txt','r',encoding = 'utf-8') as file:
    text_questions = file.readlines()

with open('/home/jncsnlp4/tb/Qwen2-VL-main/train_fakeddit/image_question.txt','r',encoding='utf-8') as file:
    image_questions = file.readlines()



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
                    "temperature": 0.5,
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
                    "temperature": 0.5,
                    "top_p": 0.95,
                    "num_beams": 1,
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
for x in range(len(datarows)):
    image_path = "/home/jncsnlp4/tb/Qwen2-VL-main/train_fakeddit/traindata_picture/post" + str(x) + ".jpg"
    if os.path.exists(image_path):
        text = datarows[x][1]
        label = datarows[x][3]
        if label == str(1) and real_num < 300:
            label = "real"
            real_num = real_num + 1
            num = num + 1
        elif label == str(0) and fake_num < 300:
            label = "fake"
            fake_num = fake_num + 1
            num = num + 1
        if label == str(1) and real_num >= 300:
            continue
        if label == str(0) and fake_num >= 300:
            continue      
        role = "politician"
        # print(role)
        text_prompt = ("Assume you are a helpful {} and you are given a piece of **Input Text**. Your task is to predict whether misinformation is present. \
        The text comes from a post (or a report). \
        By detecting whether text violates common sense, please predict whether this is a post containing misinformation.Please follow the Rules below:\n"
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
                "Your Response:\n").format(role,text)
        # print(text_prompt)
        image_prompt = ("Assume you are a helpful {} and you are given a piece of **Input image**. Your task is to predict whether misinformation is present. \
        The image comes from a post (or a report). \
        By detecting whether image violates common sense, please predict whether this is a post containing misinformation.Please follow the Rules below:\n"
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
                "Your Response:\n").format(role)

        role_response_prompt = ("Assume you are a helpful {} and you are given a piece of **Input Text** and an image. Your task is to predict whether misinformation is present in them based on your professional knowledge. \
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
                    "Your Response:\n").format(role,text)
        text_output1,text_real_prob,text_fake_prob = genai_chat_completion_response(image_path,text_prompt,types="text")
        image_output1,image_real_prob,image_fake_prob= genai_chat_completion_response(image_path,image_prompt,types="mm")
        mm_output1,mm_real_logits,mm_fake_logits = genai_chat_completion_response(image_path,role_response_prompt,types="mm")
        # print(text_output)

        text_label1 = get_label(text_output1)
        image_label1 = get_label(image_output1)
        mm_label1 = get_label(mm_output1)
        # print(f"文本标签：{text_label1}，图像标签：{image_label1}，角色{role}的mm标签：{mm_label1}")
        # print(f"文本real概率：{text_real_prob}，图像real概率：{image_real_prob}，角色{role}的mmreal概率：{mm_real_logits}")
        # print(f"文本fake概率：{text_fake_prob}，图像fake概率：{image_fake_prob}，角色{role}的mmfake概率：{mm_fake_logits}")
        weather_knowledge = 0

        text_weather_knowledge = 1
        image_weather_knowledge = 1
        mm_weather_knowledge = 1

        text_index = text_indexs[y]
        image_index = image_indexs[y]
        text_question = text_questions[num]
        image_question = image_questions[num]
        
        tknow = text_rows[num][text_index]
        final_text_knowledge = ast.literal_eval(tknow)
        final_text_knowledge = final_text_knowledge['body']
        print(f"源文本:{text}")
        print(f"文本知识:{final_text_knowledge}")
        iknow = image_rows[num][image_index]
        final_image_knowledge = ast.literal_eval(iknow)
        final_image_knowledge = final_image_knowledge['body']
        print(f"图像知识:{final_image_knowledge}")
        # if max(text_real_prob,text_fake_prob)<0.6 or text_max_similarity > 1.6 :
        text_output = ""
        image_output = ""
        mm_output = ""
        if max(text_real_prob,text_fake_prob)<0.8032 :    
            text_weather_knowledge = 0
            two_step_num = two_step_num + 1
            if text_label1 == label:
                one_step_true_num = one_step_true_num + 1
            else:
                one_step_false_num = one_step_false_num + 1
            print("文本模态不确定，需要外部知识")
            text_prompt = ("A language model was asked: Predict whether the news is real or fake according to the text of the news.The text of the news is {}.\
            The model's answer was: {}.\n"
            "The model is not very sure of its answer. Give you an additional paragraph of external knowledge about the news that \
            is absolutely correct, and your task is to rethink with the help of external knowledge and predict whether this is a news containing misinformation.\n"
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
                    "Extneral Knowledge:"
                    "{}"
                    "Let's think step by step."
                    "Your Response:\n").format(text,text_output1,final_text_knowledge)
            
            text_output,text_real_prob,text_fake_prob = genai_chat_completion_response(image_path,text_prompt,types="text")
            text_label = get_label(text_output)
            if text_label != text_label1:
                if text_label1 == label:
                    t_f_num = t_f_num + 1
                else:
                    f_t_num = f_t_num + 1
            text_label1 = text_label

        # if max(image_real_prob,image_fake_prob)<0.6 or image_max_similarity > 1.6:
        if max(image_real_prob,image_fake_prob)<0.8032:
            image_weather_knowledge = 0 
            two_step_num = two_step_num + 1
            if image_label1 == label:
                one_step_true_num = one_step_true_num + 1
            else:
                one_step_false_num = one_step_false_num + 1
            print("图像模态不确定，需要外部知识")
            image_prompt = ("A language model was asked: Predict whether the news is real or fake according to the image of the news.\
            The model's answer was: {}.\n"
            "The model is not very sure of its answer. Give you an additional paragraph of external knowledge about the news that \
            is absolutely correct, and your task is to rethink with the help of external knowledge and predict whether this is a news containing misinformation.\n"
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
                    "Extneral Knowledge:"
                    "{}"
                    "Let's think step by step."
                    "Your Response:\n").format(image_output1,final_image_knowledge)
            image_output,image_real_prob,image_fake_prob = genai_chat_completion_response(image_path,text_prompt,types="mm")
            image_label = get_label(image_output)
            if image_label != image_label1:
                if image_label1 == label:
                    t_f_num = t_f_num + 1
                else:
                    f_t_num = f_t_num + 1
            image_label1 = image_label    

        # # if max(mm_real_logits,mm_fake_logits) < 0.6 or text_max_similarity > 1.6 or image_max_similarity > 1.6:
        if max(mm_real_logits,mm_fake_logits) < 0.8032 :
            mm_weather_knowledge = 0
            two_step_num = two_step_num + 1
            if mm_label1 == label:
                one_step_true_num = one_step_true_num + 1
            else:
                one_step_false_num = one_step_false_num + 1
            print("多模态不确定，需要外部知识")
            final_mm_knowledge = "1."+final_text_knowledge + " 2." + final_image_knowledge
            mm_prompt = ("A language model was asked: Predict whether the news is real or fake according to the text and image of the news.The text of the news is {}.\
            The model's answer was: {}.\n"
            "The model is not very sure of its answer. Give you an additional paragraph of external knowledge about the news that \
            is absolutely correct, and your task is to rethink with the help of external knowledge and predict whether this is a news containing misinformation.\n"
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
                    "Extneral Knowledge:"
                    "{}"
                    "Let's think step by step."
                    "Your Response:\n").format(text,mm_output1,final_mm_knowledge)
            mm_output,mm_real_logits,mm_fake_logits = genai_chat_completion_response(image_path,text_prompt,types="mm")
            mm_label = get_label(mm_output)
            if mm_label != mm_label1:
                if mm_label1 == label:
                    t_f_num = t_f_num + 1
                else:
                    f_t_num = f_t_num + 1
            mm_label1 = mm_label

        # current_label_list = [text_label1,image_label1,mm_label1]   
        # index_list = ["text","image","mm"]
        current_real_logits = [text_real_prob.item(),image_real_prob.item(),mm_real_logits.item()]
        current_fake_logits = [text_fake_prob.item(),image_fake_prob.item(),mm_fake_logits.item()]
        print(f"现在是第{x}条数据")
        real_logits.append(current_real_logits)
        fake_logits.append(current_fake_logits)
        labels.append(label)
                
    else:
        continue

with open("fakeddittrain/fakeedit_real_logits0.8032.txt",'a',encoding='utf-8') as file:
    # real_logits = file.read()
    file.write(str(real_logits))
with open("fakeddittrain/fakeedit_fake_logits0.8032.txt",'a',encoding='utf-8') as file:
    # fake_logits = file.read()
    file.write(str(fake_logits))
with open("fakeddittrain/fakeedit_labels0.8032.txt",'a',encoding='utf-8') as file:
    # labels = file.read()
    file.write(str(labels))     
