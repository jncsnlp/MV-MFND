from email.mime import image
from shutil import which
from typing import final
from sympy import im, true
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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
    
num = 0
falsenum = 0
truenum = 0
x = 0
# print(len(datarows))
for item in data.values():
    image_path = "/home/jncsnlp4/SSD2/tb/data/MR2-en/" + item['image_path'] #fakeddit
    if os.path.exists(image_path):
        x = x + 1
        text = item['caption']
        label = int(item['label'])
        text_question_prompt = ("You are given an Input Text. Your task is to predict whether misinformation is present. The text of the news is {} \
External sources can better help you make the prediction. Please come up with a question/phrase/sentence that you would like to search on a public search engine according to the text of the news, such as Google. \
You need to come up with the question about the text based on the image of the news article. The text of the news is as followed. \
Carefully design your question so that it can return the most helpful results for making your final prediction and reasoning. Please use English to generate your questions. \n"
        "Your response:").format(text)

        image_question_prompt = ("you are given an Input Image. Your task is to predict whether misinformation is present. \
External sources can better help you make the prediction. Please come up with a question/phrase/sentence that you would like to search on a public search engine based on the image of the news, such as Google. \
You need to come up with the question about the image based on the text of the news article. The text of the news is {}\
Carefully design your question so that it can return the most helpful results for making your final prediction and reasoning. Please use English to generate your questions. \n"
        "Your response:").format(text)

        text_question = chat_response(image_path,text_question_prompt,prob=0,type="mm")
        image_question = chat_response(image_path,image_question_prompt,prob=0,type="mm")

        print(f"现在是第{x}条数据")
        print(text_question)
        print("--------------------")
        print(image_question)
        
        with open("/home/jncsnlp4/tb/Qwen2-VL-main/MR2_knowledevtext_question.txt",'a',encoding='utf-8') as file:
            file.write(text_question+"\n")
        with open("/home/jncsnlp4/tb/Qwen2-VL-main/MR2_knowledevimage_question.txt",'a',encoding='utf-8') as file:
            file.write(image_question+"\n")     
    
print(f"总数量:{truenum+falsenum}")
