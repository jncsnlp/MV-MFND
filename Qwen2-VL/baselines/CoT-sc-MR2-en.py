from email.mime import image
from re import I
from shutil import which
from typing import final
from sympy import im
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from text_sim import text_sim
import matplotlib.pyplot as plt
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
import numpy as np
# default: Load the model on the available device(s)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "/home/jncsnlp4/SSD2/model/qwen2-vl-intruct", torch_dtype="auto", device_map="auto"
)
#fakeddit
with open("/home/jncsnlp4/SSD2/tb/data/MR2-en/dataset_items_test_filtered.json",'r',encoding='utf-8') as file:
    data = json.load(file)

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
    
    generated = model.generate(**inputs, max_new_tokens=256,output_logits = True,return_dict_in_generate=True,temperature=1,top_p=0.75,top_k=2)
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
    
def get_part(text,target,end_word):
    start = text.find(target) + len(target)
    remaining = text[start:]
    end = remaining.find(end_word)
    explanation_text = remaining[:end]

    return explanation_text
    
y = 0
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
text_real_prob = tensor(0)  
text_fake_prob = tensor(0)
image_real_prob = tensor(0)  
image_fake_prob = tensor(0)
mm_real_logits = tensor(0)  
mm_fake_logits = tensor(0)
true_knowledeg_list = []
begin_time = datetime.now()
real_logits = []
fake_logits = []
labels = []
for item in data.values():
    image_path = "/home/jncsnlp4/SSD2/tb/data/MR2-en/" + item['image_path'] #fakeddit
    if os.path.exists(image_path):
        y = y + 1
        text = item['caption']
        label = int(item['label'])
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
            mm_output1,mm_real_logits,mm_fake_logits = chat_response(image_path,role_response_prompt,prob=1,type="mm")
            # print(text_output)
            mm_label1 = get_label(mm_output1)
            label_list.append(mm_label1)
        
        real_ = 0
        fake_ = 0
        for i in range(len(label_list)):
            if label_list[i] == "real":
                real_ = real_ + 1 
            else:
                fake_ = fake_ + 1
        
        final_label = "real" if real_ > fake_ else "fake"
        print(f"现在是第{y}条数据")
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
print(f"开始时间:{begin_time}\n结束时间:{end_time}")   
