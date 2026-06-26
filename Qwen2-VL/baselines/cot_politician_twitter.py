from email.mime import image
from typing import final
from sympy import im
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from text_sim import text_sim
import json
import os
import gc
import torch
import string
import requests
import csv
import ast
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

true_knowledeg_list = []
begin_time = datetime.now()
for x in range(len(data)):
    image_path = "/home/jncsnlp4/tb/LEMMA-main/"+data[x]['image_url']
    if os.path.exists(image_path):
        y = y + 1
        number = x
        knowledge = rows[y][1]
        image_knowledge = image_rows[y][1]
        text = data[x]['original_post']
        label = data[x]['label']
        if label == 1:
            label = "fake"
            fake_num = fake_num + 1
        else:
            label = "real"
            real_num = real_num + 1
            
        # role = "politician"
        # role_response_prompt = ("Assume you are a helpful{} and you are given a piece of **Input Text** and an image. Your task is to predict whether misinformation is present in them based on your professional knowledge.\
        # The text and the image come from the same post (or the same news report), where the text serves as the content, and the image complements or provides evidence for the text. \
        # Please predict whether this is a post containing misinformation by verifying the consistency of text and images and detecting whether text and images violate common sense.\
        # You will be punished if your answer is wrong. Please follow the Rules below:\n"
        #             "Rules:\n"
        #             "Generate a JSON object with three properties: 'label', 'explanation','certain'.\n" 
        #             "The return value of 'label' property should be selected from [\"real\", \"fake\"].\n"
        #             "real indicates that no misinformation is detected. \n"
        #             "fake indicates that misinformation is detected. \n"
        #             "The return value of 'explanation' property should be a detailed reasoning for the given 'label'. \n"
        #             "The return value of 'certain' property should be selected from [\"sure\",\"unsure\"].\n"
        #             "sure indicates that you are confident to your predict.\n"
        #             "unsure indicates that you are not confident to your predict.\n"
        #             "Note that your response will be passed to the python interpreter, SO NO OTHER WORDS! Just only output the json object. Do not add ```json in front of json object or behind it! \n"
        #             "Example output (JSON):\n"
        #             "{{\n"
        #             "\"label\": ,\n"
        #             "\"explanation\": ,\n"
        #             "\"certain\":\n"
        #             "}}\n"
        #             "Input Text:\n"
        #             "{}\n"
        #             "Let's think step by step."
        #             "Your Response:\n").format(role,text)

        role_response_prompt = ("You are given a piece of **Input Text** and an image. Your task is to predict whether misinformation is present in them based on your professional knowledge.\
        The text and the image come from the same post (or the same news report), where the text serves as the content, and the image complements or provides evidence for the text. \
        Please predict whether this is a post containing misinformation by verifying the consistency of text and images and detecting whether text and images violate common sense.\
        You will be punished if your answer is wrong. Please follow the Rules below:\n"
                    "Rules:\n"
                    "Generate a JSON object with two properties: 'label', 'explanation'.\n" 
                    "The return value of 'label' property should be selected from [\"real\", \"fake\"].\n"
                    "real indicates that no misinformation is detected. \n"
                    "fake indicates that misinformation is detected. \n"
                    "The return value of 'explanation' property should be a detailed reasoning for the given 'label'. \n"
                    "Note that your response will be passed to the python interpreter, SO NO OTHER WORDS! Just only output the json object. Do not add ```json in front of json object or behind it! \n"
                    "Example output (JSON):\n"
                    "{{\n"
                    "\"label\": ,\n"
                    "\"explanation\": ,\n"
                    "}}\n"
                    "Input Text:\n"
                    "{}\n"
                    "Let's think step by step."
                    "Your Response:\n").format(text)  
        mm_output1,mm_real_logits,mm_fake_logits = chat_response(image_path,role_response_prompt,prob=1,type="mm")
        mm_label = get_label(mm_output1)
        # mm_ = get_certain(mm_output1)
        # print(text_output)

        # text_weather_knowledge = 1
        # image_weather_knowledge = 1
        # mm_weather_knowledge = 1
        # # if max(text_real_prob,text_fake_prob)<0.6 or max(image_real_prob,image_fake_prob)<0.6 or max(mm_real_logits,mm_fake_logits) < 0.6:
            
        # text_extra_knowledge = ast.literal_eval(knowledge)
        # image_extra_knowledge = ast.literal_eval(image_knowledge)
        # # mm_extra_knowledge = ast.literal_eval(mm_knowledge)
        # text_question = text_questions[y]
        # image_question = image_questions[y]
        # # mm_question = mm_questions[y][1]
        # text_max_index = 0
        # text_max_similarity = 0
        # if len(text_extra_knowledge)==0:
        #     final_text_knowledge = ""
        # else:
        #     for a in range(len(text_extra_knowledge)):
        #         temp = text_sim(text,text_extra_knowledge[a]['body'])
        #         temp2 = text_sim(text_question,text_extra_knowledge[a]['body'])
        #         if temp + temp2 > text_max_similarity:
        #             text_max_similarity = temp + temp2
        #             text_max_index = a
        #     print(text_max_index,text_max_similarity)
        #     final_text_knowledge = text_extra_knowledge[text_max_index]['body']
        # image_max_index = 0
        # image_max_similarity = 0
        # if len(image_extra_knowledge) == 0:
        #     final_image_knowledge = " "
        # else:
        #     for a in range(len(image_extra_knowledge)):
        #         temp = text_sim(text,image_extra_knowledge[a]['body'])
        #         temp2 = text_sim(image_question,image_extra_knowledge[a]['body'])
        #         if temp + temp2 > image_max_similarity:
        #             image_max_similarity = temp + temp2
        #             image_max_index = a
        #     print(image_max_index,image_max_similarity)
        #     final_image_knowledge = image_extra_knowledge[image_max_index]['body']

        # mm_output = ""
        # if mm_ == "unsure" :
        #     mm_weather_knowledge = 0
        #     two_step_num = two_step_num + 1
        #     if mm_label == label:
        #         one_step_true_num = one_step_true_num + 1
        #     else:
        #         one_step_false_num = one_step_false_num + 1
        #     print("多模态不确定，需要外部知识")
        #     final_mm_knowledge = "1."+final_text_knowledge + " 2." + final_image_knowledge
        #     mm_prompt = ("A language model was asked: Predict whether the news is real or fake according to the text and image of the news.The text of the news is {}.\
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
        #             "Your Response:\n").format(text,mm_output1,final_mm_knowledge)
        #     mm_output,mm_real_logits,mm_fake_logits = chat_response(image_path,mm_prompt,prob=1,type="mm")
        #     mm_label = get_label(mm_output)
        
        final_label = mm_label
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