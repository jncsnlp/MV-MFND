from email.mime import image
from typing import final
from sympy import im
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from text_sim import text_sim
from torch import ne, tensor
import matplotlib.pyplot as plt
import json
import os
import gc
import torch
import string
import requests
import csv
import ast
import re
from datetime import datetime
# default: Load the model on the available device(s)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "/home/jncsnlp4/SSD2/model/qwen2-vl-intruct", torch_dtype="auto", device_map="auto"
)
#fakeddit
# with open("/home/jncsnlp4/tb/prompt-llava/FAKEDDIT.json",'r',encoding='utf-8') as file:
#     data = json.load(file)
#twitter
with open("/home/jncsnlp4/tb/LEMMA-main/data/twitter/twitter.json",'r',encoding='utf-8') as file:
    data = json.load(file)

#twitter
# with open("/home/jncsnlp4/tb/Qwen2-VL-main/extra_knowledge_ours_twitter2.csv", mode='r', encoding='utf-8') as file:
#     csv_dict_reader = csv.reader(file)
#     rows = list(csv_dict_reader)

#fakeddit    
#top1
# with open("/home/jncsnlp4/tb/Qwen2-VL-main/knowledge2.csv", mode='r', encoding='utf-8') as file:
#     csv_dict_reader = csv.reader(file)
#     rows = list(csv_dict_reader)
#top5
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
# default processer
processor = AutoProcessor.from_pretrained("/home/jncsnlp4/SSD2/model/qwen2-vl-intruct")


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

def chat_response(image_path,prompt,prob,type):
    if type == "text":
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[text],
            padding=True,
            return_tensors="pt",
        )
    else:
        messages = [
            {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {"type": "text", "text": prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

    inputs = inputs.to("cuda")
        # Inference: Generation of the output
    
    generated = model.generate(**inputs, max_new_tokens=256,output_logits = True,return_dict_in_generate=True,temperature=0.5,top_p=0.75,top_k=2)
    logits = generated.logits
    probs = [torch.softmax(log, dim=-1) for log in logits]
        # print(probs)
    generated_ids = generated.sequences
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    logit_pro = 0
    # for i, token_id in enumerate(generated_ids[0][len(inputs.input_ids[0]):]):
    #     token_prob = probs[i][0, token_id].item()
    #     word = processor.decode(token_id)
    #     # if token_id == 7951 or token_id == 30570:
    #     print(f"Token ID: {token_id}, word:{word},Probability: {token_prob}")
    #         # logit_pro = token_prob
    real_prob = 0
    fake_prob = 0
    certain_prob = 0
    for i, token_id in enumerate(generated_ids[0][len(inputs.input_ids[0]):]):
        if processor.decode(token_id) == "sure":
            certain_prob = probs[i][0, token_id].item()
            # print("prob:",certain_prob,token_id)
        if token_id == 7951 or token_id == 30570:
            # index_prob = probs[i][0]
            top_value,top_index = torch.topk(probs[i][0],5)
            for x,y in zip(top_value,top_index):
                # print(x.item(),y.item())  
                word = processor.decode(y)
                if word == "fake":
                    fake_prob = x
                if word == "real":
                    real_prob = x 
            break        
    if prob == 0:
        return output_text[0]    
    elif prob == 1:
        return output_text[0],real_prob,fake_prob 
    else:
        return output_text[0],real_prob,fake_prob,certain_prob 
#Logit Margin
# def chat_response(image_path,prompt,prob,type):
#     if type == "text":
#         messages = [
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "text", "text": prompt},
#                 ],
#             }
#         ]
#         text = processor.apply_chat_template(
#             messages, tokenize=False, add_generation_prompt=True
#         )
#         inputs = processor(
#             text=[text],
#             padding=True,
#             return_tensors="pt",
#         )
#     else:
#         messages = [
#             {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "image",
#                     "image": image_path,
#                 },
#                 {"type": "text", "text": prompt},
#                 ],
#             }
#         ]
#         text = processor.apply_chat_template(
#             messages, tokenize=False, add_generation_prompt=True
#         )
#         image_inputs, video_inputs = process_vision_info(messages)
#         inputs = processor(
#             text=[text],
#             images=image_inputs,
#             videos=video_inputs,
#             padding=True,
#             return_tensors="pt",
#         )

#     inputs = inputs.to("cuda")
#         # Inference: Generation of the output
    
#     generated = model.generate(**inputs, max_new_tokens=256,output_logits = True,return_dict_in_generate=True,temperature=0.5,top_p=0.75,top_k=50)
#     logits = generated.logits # 这是一个 tuple，长度为生成的 token 数
#     generated_ids = generated.sequences
    
#     # 找到生成的第一个有效 token 的起始索引
#     input_len = inputs.input_ids.shape[1]
    
#     output_text = processor.batch_decode(
#         [generated_ids[0][input_len:]], skip_special_tokens=True, clean_up_tokenization_spaces=False
#     )

#     real_prob = 0
#     fake_prob = 0
#     certain_prob = 0
#     logit_margin = 0  # 新增：用于存储计算出的 Margin

#     for i, token_id in enumerate(generated_ids[0][input_len:]):
#         # 1. 记录 "sure" 的概率
#         if processor.decode(token_id) == "sure":
#             # 注意：probs[i] 是 softmax 后的，logits[i] 是原始分数
#             # 审稿人要的 margin 通常基于 logits (原始 score)
#             certain_prob = torch.softmax(logits[i], dim=-1)[0, token_id].item()

#         # 2. 找到关键位置（Real/Fake 出现的那个位置）
#         if token_id == 7951 or token_id == 30570:
#             current_token_logits = logits[i][0] # 形状: [vocab_size]
            
#             # --- 新增：计算 Logit Margin ---
#             # 找到最大的前两个 logit 值
#             top_vals, _ = torch.topk(current_token_logits, k=2)
#             logit_margin = (top_vals[0] - top_vals[1]).item()
#             # ----------------------------

#             # 保持你原有的概率获取逻辑（基于 Softmax）
#             current_token_probs = torch.softmax(current_token_logits, dim=-1)
#             top_v, top_idx = torch.topk(current_token_probs, 5)
#             for x, y in zip(top_v, top_idx):
#                 word = processor.decode(y)
#                 if word == "fake":
#                     fake_prob = x
#                 if word == "real":
#                     real_prob = x
#             break        

#     # 根据你的 prob 参数返回结果
#     if prob == 0:
#         return output_text[0]    
#     elif prob == 1:
#         # 你可以根据需要决定是否在这里也返回 logit_margin
#         return output_text[0], real_prob, fake_prob, logit_margin
#     else:
#         # 原有的返回，多加一个 logit_margin
#         return output_text[0], real_prob, fake_prob, certain_prob, logit_margin    

#entroy
# def chat_response(image_path,prompt,prob,type):
#     if type == "text":
#         messages = [
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "text", "text": prompt},
#                 ],
#             }
#         ]
#         text = processor.apply_chat_template(
#             messages, tokenize=False, add_generation_prompt=True
#         )
#         inputs = processor(
#             text=[text],
#             padding=True,
#             return_tensors="pt",
#         )
#     else:
#         messages = [
#             {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "image",
#                     "image": image_path,
#                 },
#                 {"type": "text", "text": prompt},
#                 ],
#             }
#         ]
#         text = processor.apply_chat_template(
#             messages, tokenize=False, add_generation_prompt=True
#         )
#         image_inputs, video_inputs = process_vision_info(messages)
#         inputs = processor(
#             text=[text],
#             images=image_inputs,
#             videos=video_inputs,
#             padding=True,
#             return_tensors="pt",
#         )

#     inputs = inputs.to("cuda")
    
#     # 注意：为了计算真实的熵，这里 top_k 不能设为 2，否则熵会被严重低估。
#     # 建议设为一个较大的值或默认值，以便获取更完整的分布。
#     generated = model.generate(
#         **inputs, 
#         max_new_tokens=256,
#         output_logits=True,
#         return_dict_in_generate=True,
#         temperature=0.5,
#         top_p=0.75,
#         top_k=50 # 增加 top_k 以获得更准确的分布统计
#     )
    
#     logits = generated.logits
#     generated_ids = generated.sequences
#     input_len = inputs.input_ids.shape[1]
    
#     output_text = processor.batch_decode(
#         [generated_ids[0][input_len:]], skip_special_tokens=True, clean_up_tokenization_spaces=False
#     )

#     real_prob = 0
#     fake_prob = 0
#     certain_prob = 0
#     entropy_val = 0 # 新增：用于存储计算出的信息熵

#     for i, token_id in enumerate(generated_ids[0][input_len:]):
#         # 记录 probs（基于当前 token 位置的 logits）
#         current_token_probs = torch.softmax(logits[i][0], dim=-1)

#         if processor.decode(token_id) == "sure":
#             certain_prob = current_token_probs[token_id].item()

#         if token_id == 7951 or token_id == 30570:
#             # --- 新增：计算信息熵 Entropy ---
#             # 为了数值稳定性，添加一个极小的 epsilon 防止 log(0)
#             eps = 1e-9
#             p = current_token_probs
#             entropy_val = -torch.sum(p * torch.log(p + eps)).item()
#             # -------------------------------

#             top_value, top_index = torch.topk(current_token_probs, 5)
#             for x, y in zip(top_value, top_index):
#                 word = processor.decode(y)
#                 if word == "fake":
#                     fake_prob = x
#                 if word == "real":
#                     real_prob = x
#             break        
            
#     if prob == 0:
#         return output_text[0]    
#     elif prob == 1:
#         # 返回文本，概率值，以及熵
#         return output_text[0], real_prob, fake_prob, entropy_val
#     else:
#         return output_text[0], real_prob, fake_prob, certain_prob, entropy_val

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
true_knowledeg_list = []
begin_time = datetime.now()
real_logits = []
fake_logits = []
labels = []
true_knowledeg_list = []
text_real_prob = tensor(0)  
text_fake_prob = tensor(0)
image_real_prob = tensor(0)  
image_fake_prob = tensor(0)
mm_real_logits = tensor(0)  
mm_fake_logits = tensor(0)
begin_time = datetime.now()

trusted_website = {"https://en.wikipedia.org/","https://www.bbc.com/news","https://www.nytimes.com","https://www.theguardian.com","https://www.reuters.com","https://www.ap.org","https://www.wikipedia.org/"}

for x in range(len(data)):
    # if x == 100:
    #     break
    image_path = "/home/jncsnlp4/tb/LEMMA-main/"+data[x]['image_url']
    # print(image_path)
    # image_path = "/home/jncsnlp4/tb/prompt-llava/picture/post" + str(x) + ".jpg" #fakeddit
    if os.path.exists(image_path):
        y = y + 1
        number = x
        knowledge = rows[y][1]
        image_knowledge = image_rows[y][1]
        # mm_knowledge = mm_rows[y][1]
        text = data[x]['original_post']
        label = data[x]['label']
        if label == 1:
            label = "fake"
            fake_num = fake_num + 1
        else:
            label = "real"
            real_num = real_num + 1
        # 
        # role_prompt = ("you are given a piece of **Input Text** and an image. The text and the image come from the same post (or the same news report). \
        # Your task is to give a role that would be helpful in predicting whether misinformation is present at them.\
        # Please follow the Rules below:\n"
        #         "Rules:\n"
        #         "Generate a JSON object with two properties: 'role', 'explanation'.\n" 
        #         "The return value of 'role' property must be selected in['politician','journalist','historian','fact-checker','image-analyst ','reasoning-expert','medical-expert'].\n"
        #         "The return value of 'explanation' property should be a detailed reasoning for the given 'role'. \n"
        #         "Note that your response will be passed to the python interpreter, SO NO OTHER WORDS! Just only output the json object. Do not add ```json in front of json object or behind it! \n"
        #         "Example output (JSON):\n"
        #         "{{\n"
        #             "\"role\": ,\n"
        #             "\"explanation\":\n"
        #         "}}\n"
        #         "Input Text:\n"
        #         "{}\n"
        #         "Let's think step by step."
        #         "Your Response:\n").format(text)
        
        # output_text = chat_response(image_path,role_prompt,prob=0,type="mm")
        # # print(output_text)
        # output_text = output_text.split()
        # # print(output_text)
        # for i, item in enumerate(output_text):
        #     if "role" in item:
        #         index = i
        #         break
        # role = output_text[index+1] 
        # role = remove_punctuation_manual(role)
        role = "politician"
        # role = "none"
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
        text_output1,text_real_prob,text_fake_prob = chat_response(image_path,text_prompt,prob=1,type="text")
        image_output1,image_real_prob,image_fake_prob= chat_response(image_path,image_prompt,prob=1,type="image")
        mm_output1,mm_real_logits,mm_fake_logits = chat_response(image_path,role_response_prompt,prob=1,type="mm")
        # print(text_output)

        text_label1 = get_label(text_output1)
        image_label1 = get_label(image_output1)
        mm_label1 = get_label(mm_output1)
        # print(f"文本标签：{text_label1}，图像标签：{image_label1}，角色{role}的mm标签：{mm_label1}")
        # print(f"文本real概率：{text_real_prob}，图像real概率：{image_real_prob}，角色{role}的mmreal概率：{mm_real_logits}")
        # print(f"文本fake概率：{text_fake_prob}，图像fake概率：{image_fake_prob}，角色{role}的mmfake概率：{mm_fake_logits}")
        weather_knowledge = 0
 

        
        # #加外部知识 分开加
        text_weather_knowledge = 1
        image_weather_knowledge = 1
        mm_weather_knowledge = 1
        # if max(text_real_prob,text_fake_prob)<0.6 or max(image_real_prob,image_fake_prob)<0.6 or max(mm_real_logits,mm_fake_logits) < 0.6:
            
        text_extra_knowledge = ast.literal_eval(knowledge)
        if image_knowledge == "":
            image_extra_knowledge = []
        else:
            image_extra_knowledge = ast.literal_eval(image_knowledge)
        # mm_extra_knowledge = ast.literal_eval(mm_knowledge)
        text_question = text_questions[y]
        image_question = image_questions[y]
        # mm_question = mm_questions[y][1]
        text_max_index = 0
        text_max_similarity = 0
        if len(text_extra_knowledge)==0:
            final_text_knowledge = ""
        else:
            for a in range(len(text_extra_knowledge)):
                link = text_extra_knowledge[a]['href']
                domain = re.search('https?://([A-Za-z_0-9.-]+).*', link).group(1)
                weight = 1
                if domain in trusted_website:
                    weight = 1.5
                temp = text_sim(text,text_extra_knowledge[a]['body'])
                temp2 = text_sim(text_question,text_extra_knowledge[a]['body'])
                if weight *(temp + temp2) > text_max_similarity:
                    text_max_similarity = weight *(temp + temp2)
                    text_max_index = a
            print(text_max_index,text_max_similarity)
            final_text_knowledge = text_extra_knowledge[text_max_index]['body']
        image_max_index = 0
        image_max_similarity = 0
        if len(image_extra_knowledge) == 0:
            final_image_knowledge = " "
        else:
            for a in range(len(image_extra_knowledge)):
                link = image_extra_knowledge[a]['href']
                domain = re.search('https?://([A-Za-z_0-9.-]+).*', link).group(1)
                weight = 1
                if domain in trusted_website:
                    weight = 1.5
                temp = text_sim(text,image_extra_knowledge[a]['body'])
                temp2 = text_sim(image_question,image_extra_knowledge[a]['body'])
                if weight*(temp + temp2) > image_max_similarity:
                    image_max_similarity = weight*(temp + temp2)
                    image_max_index = a
            print(image_max_index,image_max_similarity)
            final_image_knowledge = image_extra_knowledge[image_max_index]['body']
        
        # # if max(text_real_prob,text_fake_prob)<0.6 or text_max_similarity > 1.6 :
        if max(text_real_prob,text_fake_prob)<0.8013:
        # if text_margin >0.66: 
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
            is absolutely correct:{}, and your task is to rethink with the help of external knowledge and predict whether this is a news containing misinformation.\n"
            "Please follow the Rules below:\n"
                    "Rules:\n"
                    "Generate a JSON object with two properties: 'label', 'explanation'.\n" 
                    "The return value of 'label' property should be selected from [\"fake\", \"real\"].\n"
                    "fake indicates that misinformation is detected. \n"
                    "real indicates that no misinformation is detected. \n"
                    "The return value of 'explanation' property should be a detailed reasoning for the given 'label'. \n"
                    "Note that 'label' and 'explanation' should be consistent in their judgment of whether or not the news contains misinformation.\n"
                    "Note that your response will be passed to the python interpreter, SO NO OTHER WORDS! Just only output the json object. Do not add ```json in front of json object or behind it! \n"
                    "Example output (JSON):\n"
                    "{{\n"
                    "\"label\": ,\n"
                    "\"explanation\":\n"
                    "}}\n"
                    "Let's think step by step."
                    "Your Response:\n").format(text,text_output1,final_text_knowledge)
            
            text_output,text_real_prob,text_fake_prob,text_margin = chat_response(image_path,text_prompt,prob=1,type="text")
            text_label = get_label(text_output)
            if text_label != text_label1:
                if text_label1 == label:
                    t_f_num = t_f_num + 1
                else:
                    f_t_num = f_t_num + 1
            text_label1 = text_label

        if max(image_real_prob,image_fake_prob)<0.8013:
        # if image_margin >0.66: 
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
            is absolutely correct:{}, and your task is to rethink with the help of external knowledge and predict whether this is a news containing misinformation.\n"
            "Please follow the Rules below:\n"
                    "Rules:\n"
                    "Generate a JSON object with two properties: 'label', 'explanation'.\n" 
                    "The return value of 'label' property should be selected from [\"fake\", \"real\"].\n"
                    "fake indicates that misinformation is detected. \n"
                    "real indicates that no misinformation is detected. \n"
                    "The return value of 'explanation' property should be a detailed reasoning for the given 'label'. \n"
                    "Note that 'label' and 'explanation' should be consistent in their judgment of whether or not the news contains misinformation.\n"
                    "Note that your response will be passed to the python interpreter, SO NO OTHER WORDS! Just only output the json object. Do not add ```json in front of json object or behind it! \n"
                    "Example output (JSON):\n"
                    "{{\n"
                    "\"label\": ,\n"
                    "\"explanation\":\n"
                    "}}\n"
                    "Let's think step by step."
                    "Your Response:\n").format(image_output1,final_image_knowledge)
            image_output,image_real_prob,image_fake_prob,image_margin = chat_response(image_path,image_prompt,prob=1,type="image")
            image_label = get_label(image_output)
            if image_label != image_label1:
                if image_label1 == label:
                    t_f_num = t_f_num + 1
                else:
                    f_t_num = f_t_num + 1
            image_label1 = image_label    

        # # if max(mm_real_logits,mm_fake_logits) < 0.6 or text_max_similarity > 1.6 or image_max_similarity > 1.6:
        if max(mm_real_logits,mm_fake_logits) < 0.8013 :
        # if mm_margin >0.66: 
            mm_weather_knowledge = 0
            two_step_num = two_step_num + 1
            if mm_label1 == label:
                one_step_true_num = one_step_true_num + 1
            else:
                one_step_false_num = one_step_false_num + 1
            print("多模态不确定，需要外部知识")
            final_mm_knowledge = "1."+ final_text_knowledge + " 2." + final_image_knowledge
            mm_prompt = ("A language model was asked: Predict whether the news is real or fake according to the text and image of the news.The text of the news is {}.\
            The model's answer was: {}.\n"
            "The model is not very sure of its answer. Give you an additional paragraph of external knowledge about the news that \
            is absolutely correct:{}, and your task is to rethink with the help of external knowledge and predict whether this is a news containing misinformation.\n"
            "Please follow the Rules below:\n"
                    "Rules:\n"
                    "Generate a JSON object with two properties: 'label', 'explanation'.\n" 
                    "The return value of 'label' property should be selected from [\"fake\", \"real\"].\n"
                    "fake indicates that misinformation is detected. \n"
                    "real indicates that no misinformation is detected. \n"
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
            mm_output,mm_real_logits,mm_fake_logits,mm_margin = chat_response(image_path,mm_prompt,prob=1,type="mm")
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

        # real_logits.append(current_real_logits)
        # fake_logits.append(current_fake_logits)
        # labels.append(label)
        print(f"现在是第{x}条数据")
        # final_real_prob = sum(current_real_logits)
        # final_fake_prob = sum(current_fake_logits)
        #twitter数据集的权重
        final_real_prob =0.1*text_real_prob.item()+0.6*image_real_prob.item()+0.3*mm_real_logits.item()
        final_fake_prob =0.1*text_fake_prob.item()+0.6*image_fake_prob.item()+0.3*mm_fake_logits.item()

        # fakeddit数据集的权重
        # final_real_prob = 0.1*text_real_prob.item()+0.4*image_real_prob.item()+0.5*mm_real_logits.item()
        # final_fake_prob = 0.1*text_fake_prob.item()+0.4*image_fake_prob.item()+0.5*mm_fake_logits.item()

        #MR的权重
        # final_real_prob = 0.1*text_real_prob.item()+0.3*image_real_prob.item()+0.6*mm_real_logits.item()
        # final_fake_prob = 0.1*text_fake_prob.item()+0.3*image_fake_prob.item()+0.6*mm_fake_logits.item()
        final_label = "real" if final_real_prob>final_fake_prob else "fake"
        # final_label = mm_label1

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
        # print(f"现在是第{x}条数据")
    else:
        continue

     

# with open("twitter_real_logitsall.txt",'a',encoding='utf-8') as file:
#     file.write(str(real_logits))
#     # real_logits = file.read()
# with open("twitter_fake_logitsall.txt",'a',encoding='utf-8') as file:
#     file.write(str(fake_logits))
#     # fake_logits = file.read()
# with open("twitter_labelsall.txt",'a',encoding='utf-8') as file:
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

# /home/jncsnlp4/tb/Qwen2-VL-main/twitter_real_logits单视角_知识0.65.txt
# from mpl_toolkits.mplot3d import Axes3D
# import numpy as np
# from scipy.interpolate import griddata
# with open("twitter_real_logits单视角_知识0.65.txt",'r',encoding='utf-8') as file:
#     real_logits = file.read()
#     # file.write(str(real_logits))
# with open("twitter_fake_logits单视角_知识0.65.txt",'r',encoding='utf-8') as file:
#     fake_logits = file.read()
#     # file.write(str(fake_logits))
# with open("twitter_labels单视角_知识0.65.txt",'r',encoding='utf-8') as file:
#     labels = file.read()
#     # file.write(str(labels))

# rtp=0
# rtn=0
# ftp=0
# ftn=0
# tp = 0
# tn = 0
# fp = 0
# fn = 0
# real_logits = ast.literal_eval(real_logits)
# fake_logits = ast.literal_eval(fake_logits)
# labels = ast.literal_eval(labels)
# print(len(real_logits))
# print(len(fake_logits))
# print(len(labels))
# for x in range(len(real_logits)):
#     final_real_logits = 0.1*real_logits[x][0] + 0.4*real_logits[x][1] + 0.5*real_logits[x][2]
#     final_fake_logits = 0.1*fake_logits[x][0] + 0.4*fake_logits[x][1] + 0.5*fake_logits[x][2] 
#     label = labels[x]
#     final_label = "real" if final_real_logits > final_fake_logits else "fake"
#     if label=="real":
#         if final_label == label:
#             rtp = rtp + 1
#         else:
#             rtn = rtn + 1    
#     else:
#         if final_label == label:
#             ftp = ftp + 1
#         else:
#             ftn = ftn + 1  
#     if final_label == label:
#         if label == "real":
#             tp = tp + 1
#         else:
#             tn = tn + 1
#     else:
#         if label == "fake":
#             fp = fp + 1
#         else:
#             fn = fn + 1                 
# print(f"acc:{(rtp+ftp)/(rtp+rtn+ftp+ftn)}")
# print(f"non-rumor pre:{rtp/(rtp+ftn)}")
# print(f"non-rumor rec:{rtp/(rtp+rtn)}")
# print(f"non-rumor f1:{2*(rtp/(rtp+ftn)*rtp/(rtp+rtn))/(rtp/(rtp+ftn)+rtp/(rtp+rtn))}")
# print(f"rumor pre:{ftp/(ftp+rtn)}")
# print(f"rumor rec:{ftp/(ftp+ftn)}")
# print(f"rumor f1:{2*(ftp/(ftp+rtn)*ftp/(ftp+ftn))/(ftp/(ftp+rtn)+ftp/(ftp+ftn))}")

# print(f"accuracy:{(tp+tn)/(tp+tn+fp+fn)}")
# print(f"precision:{tp/(tp+fp)}")
# print(f"recall:{tp/(tp+fn)}")
# print(f"f1:{2*((tp/(tp+fp))*(tp/(tp+fn)))/(tp/(tp+fp)+tp/(tp+fn))}")

# from mpl_toolkits.mplot3d import Axes3D
# import numpy as np
# from scipy.interpolate import griddata
# with open("twitter_real_logits0.65.txt",'r',encoding='utf-8') as file:
#     # file.write(str(real_logits))
#     real_logits = file.read()
# with open("twitter_fake_logits0.65.txt",'r',encoding='utf-8') as file:
#     # file.write(str(fake_logits))
#     fake_logits = file.read()
# with open("twitter_labels0.65.txt",'r',encoding='utf-8') as file:
#     # file.write(str(labels))
#     labels = file.read()
# best_f1 = 0
# best_acc = 0
# best_pre = 0
# best_rec = 0
# # print(real_logits)
# print(type(real_logits))
# # print(len(real_logits))
# real_logits = ast.literal_eval(real_logits)
# fake_logits = ast.literal_eval(fake_logits)
# labels = ast.literal_eval(labels)
# print(len(real_logits))
# print(len(fake_logits))
# print(len(labels))
# X=[]
# Y=[]
# Z=[] 
# for a in range(1,11):
    
#     best_f12 = 0
#     best_acc2 = 0
#     best_pre2 = 0
#     best_rec2 = 0
#     best_parameters2 = {}
#     for b in range(1,11-a):
#         X.append(a/10.0)
#         Y.append(b/10.0)
#         c = 10 - a - b
#         tp = 0
#         tn = 0
#         fp = 0
#         fn = 0  
#         for x in range(len(real_logits)):
#             # aerfa = beita = sigema = 0.33
#             aerfa = a / 10.0
#             beita = b / 10.0
#             sigema = c / 10.0
#             final_real_logits = aerfa*real_logits[x][0] + beita*real_logits[x][1] + sigema*real_logits[x][2]
#             final_fake_logits = aerfa*fake_logits[x][0] + beita*fake_logits[x][1] + sigema*fake_logits[x][2] 
#             label = labels[x]
#             final_label = "real" if final_real_logits > final_fake_logits else "fake"
#             if final_label == label:
#                 if label == "real":
#                     tp = tp + 1
#                 else:
#                     tn = tn + 1
#             else:
#                 if label == "fake":
#                     fp = fp + 1
#                 else:
#                     fn = fn + 1
#         acc = (tp+tn)/(tp+tn+fp+fn)
#         pre = tp/(tp+fp)
#         rec = tp/(tp+fn)
#         f1 = 2*((tp/(tp+fp))*(tp/(tp+fn)))/(tp/(tp+fp)+tp/(tp+fn))
#         Z.append(f1)
#         # print(f"text:{aerfa},image:{beita},mm:{sigema}acc:{acc}f1:{f1}")
#         # with open("aaa.txt",'a',encoding='utf-8') as file:
#         #     file.write("text:"+str(aerfa)+",image:"+str(beita)+",mm:"+str(sigema)+"acc:"+str(acc)+"f1:"+str(f1)+"\n")
#         if f1 > best_f12:
#             best_f12 = f1
#             best_acc2 = acc
#             best_pre2 = pre
#             best_rec2 = rec
#             best_parameters2 = {'text':aerfa,'image':beita,'mm':sigema}
#         if f1 > best_f1:
#             best_acc = acc
#             best_pre = pre
#             best_rec = rec
#             best_f1 = f1
#             best_parameters = {'text':aerfa,'image':beita,'mm':sigema}
#     # Y.append(best_f12)
#     print(f"best_acc:{best_acc2}")
#     print(f"precision:{best_pre2}")
#     print(f"recall:{best_rec2}")
#     print(f"best_f1:{best_f12}")
#     print(f"best paramaters{best_parameters2}")  
#     print("--------------------------------")
# print(f"best_acc:{best_acc}")
# print(f"precision:{best_pre}")
# print(f"recall:{best_rec}")
# print(f"best_f1:{best_f1}")
# print(f"best paramaters{best_parameters}")   

# print(len(X))
# print(len(Y))
# print(len(Z))
# X = np.array(X)
# Y = np.array(Y)
# Z = np.array(Z)
# xi = np.linspace(min(X), max(X), 100)
# yi = np.linspace(min(Y), max(Y), 100)
# XI, YI = np.meshgrid(xi, yi)

# # 使用griddata进行插值
# ZI = griddata((X, Y), Z, (XI, YI), method='linear')

# # 绘制结果
# plt.figure(figsize=(8, 6))
# contour_levels = np.concatenate([
#     np.linspace(np.nanmin(ZI), 0.67, 10),     
#     np.linspace(0.67, np.nanmax(ZI), 5)      
# ])
# contour_levels = np.sort(np.unique(contour_levels))
# # contour_levels = np.linspace(np.nanmin(ZI), np.nanmax(ZI), 20)
# cf = plt.contourf(XI, YI, ZI, levels=contour_levels, cmap='coolwarm')  # 使用对比度高的颜色映射
# cbar = plt.colorbar(cf)
# cbar.set_label('Value')

# # 添加明显的等高线
# CS = plt.contour(XI, YI, ZI, levels=contour_levels, colors='k', linewidths=0.5)  # 黑色等高线
# plt.clabel(CS, inline=True, fontsize=8)  # 在等高线上标注数值

# plt.scatter(X, Y, c='red', s=20)  # 可选：在图上标记原始点
# plt.title("weight analysis")
# plt.xlabel('Text modal weight')
# plt.ylabel('Image modal weight')
# plt.show()
# # plt.show()
# # plt.figure()
# # plt.plot(X,Y)
# # plt.xlabel('aerfa')
# # plt.ylabel('best_f1')
# plt.savefig('twitter_arefa0.65.png', dpi=300)   