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

with open("/home/jncsnlp4/tb/LEMMA-main/data/twitter/twitter.json",'r',encoding='utf-8') as file:
    data = json.load(file)

with open("/home/jncsnlp4/tb/Qwen2-VL-main/twitter_knowledge/extra_knowledge_ours_twitter_top52.csv",mode='r',encoding="utf-8")as file:
    csv_dict_reader = csv.reader(file)
    rows = list(csv_dict_reader)


with open('/home/jncsnlp4/tb/Qwen2-VL-main/twitter_knowledge/extra_image_knowledge_ours_twitter_top52.csv',mode='r',encoding="utf-8")as file:
    csv_dict_reader = csv.reader(file)
    image_rows = list(csv_dict_reader)


with open('/home/jncsnlp4/tb/Qwen2-VL-main/twitter_knowledge/twitter_text_question.txt','r',encoding = 'utf-8') as file:
    text_questions = file.readlines()

with open('/home/jncsnlp4/tb/Qwen2-VL-main/twitter_knowledge/twitter_image_question.txt','r',encoding='utf-8') as file:
    image_questions = file.readlines()

with open('/home/jncsnlp4/tb/prompt-llava/test_index/twitter_test_text_max_index.txt','r',encoding='utf-8') as file:
    text_indexs = file.read()

with open('/home/jncsnlp4/tb/prompt-llava/test_index/twitter_test_image_max_index.txt','r',encoding='utf-8') as file:
    image_indexs = file.read()


def remove_punctuation_manual(text):
    return ''.join(char for char in text if char != '"'and char != "," and char != ":")

def get_label(final_output):
    output_text = final_output.split()
    index =0
    for i, item in enumerate(output_text):
        if "label" in item:
            index = i
            break
    if index == 0:
        return "fake"    
    else:
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
                    "top_k": 2,
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
for x in range(len(data)):
    image_path = "/home/jncsnlp4/tb/prompt-llava/picture/post" + str(x) + ".jpg" #fakeddit
    if os.path.exists(image_path):
        y = y + 1
        number = x
        knowledge = rows[y][1]
        image_knowledge = image_rows[y][1]
        text_index = text_indexs[y]
        image_index = image_indexs[y]
        text = data[x]['original_post']
        label = data[x]['label']
        if label == 1:
            label = "fake"
            fake_num = fake_num + 1
        else:
            label = "real"
            real_num = real_num + 1
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
        # # print(f"文本标签：{text_label1}，图像标签：{image_label1}，角色{role}的mm标签：{mm_label1}")
        # # print(f"文本real概率：{text_real_prob}，图像real概率：{image_real_prob}，角色{role}的mmreal概率：{mm_real_logits}")
        # # print(f"文本fake概率：{text_fake_prob}，图像fake概率：{image_fake_prob}，角色{role}的mmfake概率：{mm_fake_logits}")
        # # # print(f"文本输出：{text_output1}，图像输出：{image_output1}，角色{role}的mm输出：{mm_output1}")
        # # weather_knowledge = 0
 

        
        # # #加外部知识 分开加
        text_weather_knowledge = 1
        image_weather_knowledge = 1
        mm_weather_knowledge = 1
        # if max(text_real_prob,text_fake_prob)<0.6 or max(image_real_prob,image_fake_prob)<0.6 or max(mm_real_logits,mm_fake_logits) < 0.6:
            
        text_extra_knowledge = ast.literal_eval(knowledge)
        image_extra_knowledge = ast.literal_eval(image_knowledge)  
        # if max(text_real_prob,text_fake_prob)<0.6 or max(image_real_prob,image_fake_prob)<0.6 or max(mm_real_logits,mm_fake_logits) < 0.6:
        if len(text_extra_knowledge)==0:
            final_text_knowledge = ""    
        else:
            final_text_knowledge = text_extra_knowledge[text_index]['body']
        if len(image_extra_knowledge)==0:
            final_image_knowledge = ""
        else:    
            final_image_knowledge = image_extra_knowledge[image_index]['body']
        print(f"选择的文本知识是{text_index,final_text_knowledge}，选择的图片知识是:{image_index,final_image_knowledge}")
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

        # # if max(image_real_prob,image_fake_prob)<0.6 or image_max_similarity > 1.6:
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

        # real_logits.append(current_real_logits)
        # fake_logits.append(current_fake_logits)
        # labels.append(label)
        final_real_prob = 0.5*text_real_prob.item()+0.1*image_real_prob.item()+0.4*mm_real_logits.item()
        final_fake_prob = 0.5*text_fake_prob.item()+0.1*image_fake_prob.item()+0.4*mm_fake_logits.item()
        final_label = "real" if final_real_prob>final_fake_prob else "fake"
        # # print(y)
        # print(f"第{x}条数据,原文本{text},\n原label:{label}\n文本输出:{current_label_list[0]}\n图像输出:{current_label_list[1]}\n角色{role}的mm输出:{current_label_list[2]}\n最终输出label:{final_label}")
        
        print(f"现在是第{x}条数据")
        if mm_label1 == label:
        #     # if text_weather_knowledge == 0 or image_weather_knowledge == 0 or text_weather_knowledge == 0:
        #     #     with open("best/true_data.txt",mode="a",encoding="utf-8") as file:
        #     #         file.write("第"+str(x)+"条数据\n"+"原文本"+text+"\n"+"原label:"+label+"\n"+"\n文本原输出:"+text_output1+"\n文本调整后的输出："+text_output+"\n文本输出logit:"+str(current_real_logits[0])+str(current_fake_logits[0])+"\n图像原输出:"+image_output1+"\n图像调整后输出:"+image_output+"\n图像输出logit:"+str(current_real_logits[1])+str(current_fake_logits[1])
        #     #             +"\nmm原输出:"+mm_output1+"\n角色"+role+"mm调整后输出:"+mm_output+"\n输出logit:"+str(current_real_logits[2])+str(current_fake_logits[2])+
        #     #             "\n最终输出label:"+final_label+"\n检索的问题是:"+text_question+"\n三个模态分别进行检索的情况是(0为检索):"+str(text_weather_knowledge)+str(image_weather_knowledge)+str(mm_weather_knowledge)+"\n选择的文本外部知识是第"+str(text_max_index)+"条的:\n"+final_text_knowledge+"\n选择的图像外部知识是第"+str(image_max_index)+"条的:\n"+final_image_knowledge+"\n---------------------------------\n")
        #     # else:
        #     #     with open("best/true_data.txt",mode="a",encoding="utf-8") as file:
        #     #         file.write("第"+str(x)+"条数据\n"+"原文本"+text+"\n"+"原label:"+label+"\n"+"文本输出："+text_output1+"\n文本输出logit:"+str(current_real_logits[0])+str(current_fake_logits[0])+"\n是否进行纠正:"+index_list[0]+"\n核查后的文本label:"+current_label_list[0]+"\n图像输出:"+image_output1+"\n图像输出logit:"+str(current_real_logits[1])+str(current_fake_logits[1])+"\n是否进行纠正:"+index_list[1]+"\n纠正后的图像label:"+current_label_list[1]+
        #     #             "\n角色"+role+"mm输出:"+mm_output1+"\n输出logit:"+str(current_real_logits[2])+str(current_fake_logits[2])+"\n是否进行纠正:"+index_list[2]+"\n纠正后的mmlabel:"+current_label_list[2]+
        #     #             "\n最终输出label:"+final_label+"\n---------------------------------\n")
            if label == "real":
                tp = tp + 1
            else:
                tn = tn + 1
        else:
            # if text_weather_knowledge == 0 or image_weather_knowledge == 0 or text_weather_knowledge == 0:
            #     with open("best/false_data.txt",mode="a",encoding="utf-8") as file:
            #         file.write("第"+str(x)+"条数据\n"+"原文本"+text+"\n"+"原label:"+label+"\n"+"\n文本原输出:"+text_output1+"\n文本调整后的输出："+text_output+"\n文本输出logit:"+str(current_real_logits[0])+str(current_fake_logits[0])+"\n图像原输出:"+image_output1+"\n图像调整后输出:"+image_output+"\n图像输出logit:"+str(current_real_logits[1])+str(current_fake_logits[1])
            #             +"\nmm原输出:"+mm_output1+"\n角色"+role+"mm调整后输出:"+mm_output+"\n输出logit:"+str(current_real_logits[2])+str(current_fake_logits[2])+
            #             "\n最终输出label:"+final_label+"\n检索的问题是:"+text_question+"\n三个模态分别进行检索的情况是(0为检索):"+str(text_weather_knowledge)+str(image_weather_knowledge)+str(mm_weather_knowledge)+"\n选择的文本外部知识是第"+str(text_max_index)+"条的:\n"+final_text_knowledge+"\n选择的图像外部知识是第"+str(image_max_index)+"条的:\n"+final_image_knowledge+"\n---------------------------------\n")
            # else:
            #     with open("best/false_data.txt",mode="a",encoding="utf-8") as file:
            #         file.write("第"+str(x)+"条数据\n"+"原文本"+text+"\n"+"原label:"+label+"\n"+"文本输出："+text_output1+"\n文本输出logit:"+str(current_real_logits[0])+str(current_fake_logits[0])+"\n是否进行纠正:"+index_list[0]+"\n核查后的文本label:"+current_label_list[0]+"\n图像输出:"+image_output1+"\n图像输出logit:"+str(current_real_logits[1])+str(current_fake_logits[1])+"\n是否进行纠正:"+index_list[1]+"\n纠正后的图像label:"+current_label_list[1]+
            #             "\n角色"+role+"mm输出:"+mm_output1+"\n输出logit:"+str(current_real_logits[2])+str(current_fake_logits[2])+"\n是否进行纠正:"+index_list[2]+"\n纠正后的mmlabel:"+current_label_list[2]+
            #             "\n最终输出label:"+final_label+"\n---------------------------------\n")
            if label == "fake":
                fp = fp + 1
            else:
                fn = fn + 1
    else:
        continue

# with open("fakeedit_real_logits单视角——知识0.7.txt",'a',encoding='utf-8') as file:
#     file.write(str(real_logits))
#     # real_logits = file.read()
# with open("fakeedit_fake_logits单视角——知识0.7.txt",'a',encoding='utf-8') as file:
#     file.write(str(fake_logits))
#     # fake_logits = file.read()
# with open("fakeedit_labels单视角——知识0.7.txt",'a',encoding='utf-8') as file:
#     file.write(str(labels))
    # labels = file.read()     
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

