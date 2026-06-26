from email.mime import image
from typing import final
from sympy import im
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from text_sim import text_sim
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

#top5
with open("/home/jncsnlp4/tb/Qwen2-VL-main/twitter_knowledge/extra_knowledge_ours_twitter_top52.csv",mode='r',encoding="utf-8")as file:
    csv_dict_reader = csv.reader(file)
    rows = list(csv_dict_reader)


with open('/home/jncsnlp4/tb/Qwen2-VL-main/MR2_knowledev/extra_picture_knowledge_LEMMA_twitter22.csv',mode='r',encoding="utf-8")as file:
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
            output = remove_punctuation_manual(output_text[index+1])
            break
        else:
            output = "unsure"
    # output = remove_punctuation_manual(output_text[index+1])
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
real_logits = []
fake_logits = []
labels = []
two_step_num = 0
one_step_true_num = 0
one_step_false_num = 0
t_f_num = 0
f_t_num = 0
cf_t_num = 0
ct_f_num = 0
true_knowledeg_list = []
begin_time = datetime.now()
trusted_website = {"https://en.wikipedia.org/","https://www.bbc.com/news","https://www.nytimes.com","https://www.theguardian.com","https://www.reuters.com","https://www.ap.org","https://www.wikipedia.org/"}

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
        # print(role)
        role = "politician"
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
        text_output1,text_real_prob,text_fake_prob = chat_response(image_path,text_prompt,prob=1,type="text")
        image_output1,image_real_prob,image_fake_prob= chat_response(image_path,image_prompt,prob=1,type="image")
        mm_output1,mm_real_logits,mm_fake_logits = chat_response(image_path,role_response_prompt,prob=1,type="mm")
        # print(text_output)
        # print(image_output)
        # print(mm_output)

        text_label1 = get_label(text_output1)
        image_label1 = get_label(image_output1)
        mm_label1 = get_label(mm_output1)

        text_ = get_certain(text_output1)
        image_ = get_certain(image_output1)
        mm_ = get_certain(mm_output1)

        print(f"文本标签：{text_label1}，图像标签：{image_label1}，角色{role}的mm标签：{mm_label1}")
        print(f"文本real概率：{text_real_prob}，图像real概率：{image_real_prob}，角色{role}的mmreal概率：{mm_real_logits}")
        print(f"文本fake概率：{text_fake_prob}，图像fake概率：{image_fake_prob}，角色{role}的mmfake概率：{mm_fake_logits}")
        weather_knowledge = 0
 

        
        #加外部知识 分开加
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
                temp = text_sim(text,image_extra_knowledge[a])
                if temp > image_max_similarity:
                    image_max_similarity = temp 
                    image_max_index = a
            print(image_max_index,image_max_similarity)
            final_image_knowledge = image_extra_knowledge[image_max_index]
        # mm_max_index = 0
        # mm_max_similarity = 0
        # for a in range(len(mm_extra_knowledge)):
        #     temp = text_sim(text,mm_extra_knowledge[a]['body'])
        #     temp2 = text_sim(mm_question,mm_extra_knowledge[a]['body'])
        #     if temp + temp2 > mm_max_similarity:
        #         mm_max_similarity = temp
        #         mm_max_index = a
        # print(mm_max_similarity)
        # final_mm_knowledge = mm_extra_knowledge[mm_max_index]['body'] 
            # print(f"final knowledge:{final_knowledge}")  
        
        # if max(text_real_prob,text_fake_prob)<0.6 or text_max_similarity > 1.6 :
        text_output = ""
        image_output = ""
        mm_output = ""
        if text_ == "unsure" :    
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
                    "Your Response:\n").format(text,text_output,final_text_knowledge)
            
            text_output,text_real_prob,text_fake_prob = chat_response(image_path,text_prompt,prob=1,type="text")
            text_label = get_label(text_output)
            if text_label != text_label1:
                if text_label1 == label:
                    t_f_num = t_f_num + 1
                else:
                    f_t_num = f_t_num + 1
            text_label1 = text_label

        # if max(image_real_prob,image_fake_prob)<0.6 or image_max_similarity > 1.6:
        if image_ == "unsure":
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
                    "Your Response:\n").format(image_output,final_image_knowledge)
            image_output,image_real_prob,image_fake_prob = chat_response(image_path,image_prompt,prob=1,type="image")
            image_label = get_label(image_output)
            if image_label != image_label1:
                if image_label1 == label:
                    t_f_num = t_f_num + 1
                else:
                    f_t_num = f_t_num + 1
            image_label1 = image_label    

        # if max(mm_real_logits,mm_fake_logits) < 0.6 or text_max_similarity > 1.6 or image_max_similarity > 1.6:
        if mm_ == "unsure" :
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
                    "Your Response:\n").format(text,mm_output,final_mm_knowledge)
            mm_output,mm_real_logits,mm_fake_logits = chat_response(image_path,mm_prompt,prob=1,type="mm")
            mm_label = get_label(mm_output)
            if mm_label != mm_label1:
                if mm_label1 == label:
                    t_f_num = t_f_num + 1
                else:
                    f_t_num = f_t_num + 1
            mm_label1 = mm_label

        current_label_list = [text_label1,image_label1,mm_label1]   
        index_list = ["text","image","mm"]
        current_real_logits = [text_real_prob.item(),image_real_prob.item(),mm_real_logits.item()]
        current_fake_logits = [text_fake_prob.item(),image_fake_prob.item(),mm_fake_logits.item()]

        # real_logits.append(current_real_logits)
        # fake_logits.append(current_fake_logits)
        # labels.append(label)
        print(f"第{x}条数据")
        final_real_prob =0.1*text_real_prob.item()+0.6*image_real_prob.item()+0.3*mm_real_logits.item()
        final_fake_prob =0.1*text_fake_prob.item()+0.6*image_fake_prob.item()+0.3*mm_fake_logits.item()
        final_label = "real" if final_real_prob>final_fake_prob else "fake"
        # print(y)
        # print(f"第{x}条数据,原文本{text},\n原label:{label}\n文本输出:{current_label_list[0]}\n图像输出:{current_label_list[1]}\n角色{role}的mm输出:{current_label_list[2]}\n最终输出label:{final_label}")
       
        
        if final_label == label:
            # if text_weather_knowledge == 0 or image_weather_knowledge == 0 or text_weather_knowledge == 0:
            #     with open("best/certain_true_data.txt",mode="a",encoding="utf-8") as file:
            #         file.write("第"+str(x)+"条数据\n"+"原文本"+text+"\n"+"原label:"+label+"\n"+"\n文本原输出:"+text_output1+"\n文本调整后的输出："+text_output+"\n文本输出logit:"+str(current_real_logits[0])+str(current_fake_logits[0])+"\n图像原输出:"+image_output1+"\n图像调整后输出:"+image_output+"\n图像输出logit:"+str(current_real_logits[1])+str(current_fake_logits[1])
            #             +"\nmm原输出:"+mm_output1+"\n角色"+role+"mm调整后输出:"+mm_output+"\n输出logit:"+str(current_real_logits[2])+str(current_fake_logits[2])+
            #             "\n最终输出label:"+final_label+"\n检索的问题是:"+text_question+"\n三个模态分别进行检索的情况是(0为检索):"+str(text_weather_knowledge)+str(image_weather_knowledge)+str(mm_weather_knowledge)+"\n选择的文本外部知识是第"+str(text_max_index)+"条的:\n"+final_text_knowledge+"\n选择的图像外部知识是第"+str(image_max_index)+"条的:\n"+final_image_knowledge+"\n---------------------------------\n")
            # else:
            #     with open("best/certain_ttrue_data.txt",mode="a",encoding="utf-8") as file:
            #         file.write("第"+str(x)+"条数据\n"+"原文本"+text+"\n"+"原label:"+label+"\n"+"文本输出："+text_output1+"\n文本输出logit:"+str(current_real_logits[0])+str(current_fake_logits[0])+"\n是否进行纠正:"+index_list[0]+"\n核查后的文本label:"+current_label_list[0]+"\n图像输出:"+image_output1+"\n图像输出logit:"+str(current_real_logits[1])+str(current_fake_logits[1])+"\n是否进行纠正:"+index_list[1]+"\n纠正后的图像label:"+current_label_list[1]+
            #             "\n角色"+role+"mm输出:"+mm_output1+"\n输出logit:"+str(current_real_logits[2])+str(current_fake_logits[2])+"\n是否进行纠正:"+index_list[2]+"\n纠正后的mmlabel:"+current_label_list[2]+
            #             "\n最终输出label:"+final_label+"\n---------------------------------\n")
            if label == "real":
                tp = tp + 1
            else:
                tn = tn + 1
        else:
            # if text_weather_knowledge == 0 or image_weather_knowledge == 0 or text_weather_knowledge == 0:
            #     with open("best/certain_tfalse_data.txt",mode="a",encoding="utf-8") as file:
            #         file.write("第"+str(x)+"条数据\n"+"原文本"+text+"\n"+"原label:"+label+"\n"+"\n文本原输出:"+text_output1+"\n文本调整后的输出："+text_output+"\n文本输出logit:"+str(current_real_logits[0])+str(current_fake_logits[0])+"\n图像原输出:"+image_output1+"\n图像调整后输出:"+image_output+"\n图像输出logit:"+str(current_real_logits[1])+str(current_fake_logits[1])
            #             +"\nmm原输出:"+mm_output1+"\n角色"+role+"mm调整后输出:"+mm_output+"\n输出logit:"+str(current_real_logits[2])+str(current_fake_logits[2])+
            #             "\n最终输出label:"+final_label+"\n检索的问题是:"+text_question+"\n三个模态分别进行检索的情况是(0为检索):"+str(text_weather_knowledge)+str(image_weather_knowledge)+str(mm_weather_knowledge)+"\n选择的文本外部知识是第"+str(text_max_index)+"条的:\n"+final_text_knowledge+"\n选择的图像外部知识是第"+str(image_max_index)+"条的:\n"+final_image_knowledge+"\n---------------------------------\n")
            # else:
            #     with open("best/certain_tfalse_data.txt",mode="a",encoding="utf-8") as file:
            #         file.write("第"+str(x)+"条数据\n"+"原文本"+text+"\n"+"原label:"+label+"\n"+"文本输出："+text_output1+"\n文本输出logit:"+str(current_real_logits[0])+str(current_fake_logits[0])+"\n是否进行纠正:"+index_list[0]+"\n核查后的文本label:"+current_label_list[0]+"\n图像输出:"+image_output1+"\n图像输出logit:"+str(current_real_logits[1])+str(current_fake_logits[1])+"\n是否进行纠正:"+index_list[1]+"\n纠正后的图像label:"+current_label_list[1]+
            #             "\n角色"+role+"mm输出:"+mm_output1+"\n输出logit:"+str(current_real_logits[2])+str(current_fake_logits[2])+"\n是否进行纠正:"+index_list[2]+"\n纠正后的mmlabel:"+current_label_list[2]+
            #             "\n最终输出label:"+final_label+"\n---------------------------------\n")
            if label == "fake":
                fp = fp + 1
            else:
                fn = fn + 1
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
# print(f"预测对的数量：{tp+tn}--预测错误的数量{fp+fn}")
# print(f"数据集中real数量:{real_num}fake数量:{fake_num}")
# print(f"预测中real数量:{tp+fp}fake数量:{tn+fn}")
# print(f"预测对的中real的数量:{tp}fake的数量:{tn}")
# print(f"预测错的中real的数量:{fp}fake的数量:{fn}")  
# print(f"两步预测中进入第二步的数量:{one_step_true_num+one_step_false_num}，其中预测对的数量:{one_step_true_num}，预测错的数量:{one_step_false_num}")
# print(f"第二步预测中预测对的改成错的数量:{t_f_num}，错的改成对的数量:{f_t_num}")
# print(f"纠正模块中进行纠正的数量：{ct_f_num+cf_t_num}，其中把对的改成错的数量:{ct_f_num}，把错的改成对的数量:{cf_t_num}")
print(f"开始时间:{begin_time}\n结束时间:{end_time}")   