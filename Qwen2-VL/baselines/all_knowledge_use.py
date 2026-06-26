import datetime
from email.mime import image
from typing import final
from sympy import im
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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
    "/home/jncsnlp4/lxt/model/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
)
#fakeddit
with open("/home/jncsnlp4/tb/prompt-llava/FAKEDDIT.json",'r',encoding='utf-8') as file:
    data = json.load(file)
#twitter
# with open("/home/jncsnlp4/tb/LEMMA-main/data/twitter/twitter.json",'r',encoding='utf-8') as file:
#     data = json.load(file)

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
with open("extra_knowledge_ours_fakeddit_top52.csv",mode='r',encoding="utf-8")as file:
    csv_dict_reader = csv.reader(file)
    rows = list(csv_dict_reader)


with open('extra_image_knowledge_ours_fakeddit_top52.csv',mode='r',encoding="utf-8")as file:
    csv_dict_reader = csv.reader(file)
    image_rows = list(csv_dict_reader)


with open('fakeddit_question.txt','r',encoding = 'utf-8') as file:
    questions = file.readlines()

with open('fakeedit_image_question.txt','r',encoding='utf-8') as file:
    image_questions = file.readlines()
# default processer
processor = AutoProcessor.from_pretrained("/home/jncsnlp4/lxt/model/Qwen2-VL-7B-Instruct")


def remove_punctuation_manual(text):
    return ''.join(char for char in text if char != '"'and char != ",")

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
    
    generated = model.generate(**inputs, max_new_tokens=256,output_logits = True,return_dict_in_generate=True,temperature=0.1)
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
    
y = -1
fake_num = 0
real_num = 0
tp = 0
tn = 0
fp = 0
fn = 0
true_knowledeg_list = []
begin_time = datetime.now()
print(begin_time)
for x in range(len(data)):
    # if x == 100:
    #     break
    # image_path = "/home/jncsnlp4/tb/prompt-llava/picture/post" + str(data[x]['post_id']) + ".jpg"
    # print(image_path)
    image_path = "/home/jncsnlp4/tb/prompt-llava/picture/post" + str(x) + ".jpg" #fakeddit
    # image_path = "/home/jncsnlp4/tb/LEMMA-main/"+data[x]['image_url']
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

        # response = requests.post(url, headers=headers, json=data)
        #无外部知识
        # text_prompt = ("You are given a piece of **Input Text**. Your task is to predict whether misinformation is present.\
        # The text comes from a post (or a report). \
        # By detecting whether text violates common sense, please predict whether this is a post containing misinformation.Please follow the Rules below:\n"
        # "Rules:\n"
        # "Generate a JSON object with three properties: 'label', 'explanation','certain'.\n" 
        # "The return value of 'label' property should be selected from [\"real\", \"fake\"].\n"
        # "real indicates that no misinformation is detected. \n"
        # "fake indicates that misinformation is detected. \n"
        # "The return value of 'explanation' property should be a detailed reasoning for the given 'label'. \n"
        # "The return value of 'certain' property should be selected from [\"sure\",\"unsure\"].\n"
        # "sure indicates that you are confident to your predict.\n"
        # "unsure indicates that you are not confident to your predict.\n"
        # "Note that your response will be passed to the python interpreter, SO NO OTHER WORDS! Just only output the json object. Do not add ```json in front of json object or behind it! \n"
        # "Example output (JSON):\n"
        # "{{\n\
        #     \"label\": "",\n\
        #     \"explanation\":""\n\
        #     \"certain\":""\n\
        # }}\n"
        # "Input Text:\n"
        # "{}\n"
        # "Let's think step by step."
        # "Your Response:\n").format(text)

    #     text_role_prompt = ("you are given a piece of **Input Text**. The text comes from a post. \
    # Your task is to give a role that would be helpful in predicting whether misinformation is present at it based on the input text.\
    # Please follow the Rules below:\n"
    #             "Rules:\n"
    #             "Generate a JSON object with two properties: 'role', 'explanation'.\n" 
    #             "The return value of 'role' property must be a selected in['politician','journalist','historian','fact-checker','image-analyst ','reasoning-expert','medical-experts'].\n"
    #             "The return value of 'explanation' property should be a detailed reasoning for the given 'role'. \n"
    #             "Note that your response will be passed to the python interpreter, SO NO OTHER WORDS! Just only output the json object. Do not add ```json in front of json object or behind it! \n"
    #             "Example output (JSON):\n"
    #             "{{\n\
    #                 \"role\": \"\",\n\
    #                 \"explanation\":""\n\
    #             }}\n"
    #             "Input Text:\n"
    #             "{}\n"
    #             "Let's think step by step."
    #             "Your Response:\n").format(text)
        
    #     output_text = chat_response(image_path,text_role_prompt,prob=0,type="text")
    #     # print(output_text)
    #     output_text = output_text.split()
    #     # print(output_text)
    #     for i, item in enumerate(output_text):
    #         if "role" in item:
    #             index = i
    #             break
    #     role = output_text[index+1] 
    #     text_role = remove_punctuation_manual(role)

        #         text_prompt = ("You are given a piece of Input text. Your task is to predict whether misinformation is present.\
# The image comes from a post (or a report). \
# By detecting whether image violates common sense, please predict whether this is a post containing misinformation.Please follow the Rules below:\n"
#         "Rules:\n"
#         "Generate a JSON object with three properties: 'label', 'explanation','certain'.\n" 
#         "The return value of 'label' property should be selected from [\"real\", \"fake\"].\n"
#         "real indicates that no misinformation is detected. \n"
#         "fake indicates that misinformation is detected. \n"
#         "The return value of 'explanation' property should be a detailed reasoning for the given 'label'. \n"
#         "The return value of 'certain' property should be selected from [\"sure\",\"unsure\"].\n"
#         "sure indicates that you are confident to your predict.\n"
#         "unsure indicates that you are not confident to your predict.\n"
#         "Note that your response will be passed to the python interpreter, SO NO OTHER WORDS! Just only output the json object. Do not add ```json in front of json object or behind it! \n"
#         "Example output (JSON):\n"
#         "{{\n\
#             \"label\": "",\n\
#             \"explanation\":""\n\
#             \"certain\":""\n\
#         }}\n"
#         "Let's think step by step."
#         "Your Response:\n")

#         image_prompt = ("You are given a piece of Input image. Your task is to predict whether misinformation is present.\
#         The image comes from a post (or a report). \
#         By detecting whether image violates common sense, please predict whether this is a post containing misinformation.Please follow the Rules below:\n"
#         "Rules:\n"
#         "Generate a JSON object with three properties: 'label', 'explanation','certain'.\n" 
#         "The return value of 'label' property should be selected from [\"real\", \"fake\"].\n"
#         "real indicates that no misinformation is detected. \n"
#         "fake indicates that misinformation is detected. \n"
#         "The return value of 'explanation' property should be a detailed reasoning for the given 'label'. \n"
#         "The return value of 'certain' property should be selected from [\"sure\",\"unsure\"].\n"
#         "sure indicates that you are confident to your predict.\n"
#         "unsure indicates that you are not confident to your predict.\n"
#         "Note that your response will be passed to the python interpreter, SO NO OTHER WORDS! Just only output the json object. Do not add ```json in front of json object or behind it! \n"
#         "Example output (JSON):\n"
#         "{{\n\
#             \"label\": "",\n\
#             \"explanation\":""\n\
#             \"certain\":""\n\
#         }}\n"
#         "Let's think step by step."
#         "Your Response:\n")
#         role_response_prompt = ("You are a {} and you are given a piece of **Input Text** and an image. Your task is to predict whether misinformation is present in them based on your expertise as a {}.\
#     The text and the image come from the same post (or the same news report), where the text serves as the content, and the image complements or provides evidence for the text,  \
#     Please predict whether this is a post containing misinformation by detecting whether text and images violate common sense and verifying if there is anything in the image that conflicts with the text.\
#     You will be punished if your answer is wrong. Please follow the Rules below:\n"
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
#             "{{\n\
#                 \"label\": "",\n\
#                 \"explanation\":""\n\
#                 \"certain\":""\n\
#             }}\n"
#             "Input Text:\n"
#             "{}\n"
#             "Let's think step by step."
#             "Your Response:\n").format(role,role,text)
    #     image_role_prompt = ("You are a {} and you are given a piece of **Input image**. The image comes from a post and as followed. \
    # Your task is to give a role that would be helpful in predicting whether misinformation is present at it based on the input image.\
    # Please follow the Rules below:\n"
    #             "Rules:\n"
    #             "Generate a JSON object with two properties: 'role', 'explanation'.\n" 
    #             "The return value of 'role' property must be a selected in['politician','journalist','historian','fact-checker','image-analyst ','reasoning-expert','medical-experts'].\n"
    #             "The return value of 'explanation' property should be a detailed reasoning for the given 'role'. \n"
    #             "Note that your response will be passed to the python interpreter, SO NO OTHER WORDS! Just only output the json object. Do not add ```json in front of json object or behind it! \n"
    #             "Example output (JSON):\n"
    #             "{{\n\
    #                 \"role\": \"\",\n\
    #                 \"explanation\":""\n\
    #             }}\n"
    #             "Let's think step by step."
    #             "Your Response:\n")
        
    #     output_text = chat_response(image_path,image_role_prompt,prob=0,type="mm")
    #     # print(output_text)
    #     output_text = output_text.split()
    #     # print(output_text)
    #     for i, item in enumerate(output_text):
    #         if "role" in item:
    #             index = i
    #             break
    #     role = output_text[index+1] 
    #     image_role = remove_punctuation_manual(role)
        role_prompt = ("you are given a piece of **Input Text** and an image. The text and the image come from the same post (or the same news report). \
Your task is to give a role that would be helpful in predicting whether misinformation is present at them.\
Please follow the Rules below:\n"
                "Rules:\n"
                "Generate a JSON object with two properties: 'role', 'explanation'.\n" 
                "The return value of 'role' property must be selected in['politician','journalist','historian','fact-checker','image-analyst ','reasoning-expert','medical-expert'].\n"
                "The return value of 'explanation' property should be a detailed reasoning for the given 'role'. \n"
                "Note that your response will be passed to the python interpreter, SO NO OTHER WORDS! Just only output the json object. Do not add ```json in front of json object or behind it! \n"
                "Example output (JSON):\n"
                "{{\n\
                    \"role\": \"\",\n\
                    \"explanation\":""\n\
                }}\n"
                "Input Text:\n"
                "{}\n"
                "Let's think step by step."
                "Your Response:\n").format(text)
        
        output_text = chat_response(image_path,role_prompt,prob=0,type="mm")
        # print(output_text)
        output_text = output_text.split()
        # print(output_text)
        for i, item in enumerate(output_text):
            if "role" in item:
                index = i
                break
        role = output_text[index+1] 
        role = remove_punctuation_manual(role)  

        text_prompt = ("You are given a piece of **Input Text**. Your task is to predict whether misinformation is present.\
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
        "{{\n\
            \"label\": "",\n\
            \"explanation\":""\n\
        }}\n"
        "Input Text:\n"
        "{}\n"
        "Let's think step by step."
        "Your Response:\n").format(text)
        image_prompt = ("You are given a piece of **Input image**. Your task is to predict whether misinformation is present.\
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
        "{{\n\
            \"label\": "",\n\
            \"explanation\":""\n\
        }}\n"
        "Let's think step by step."
        "Your Response:\n").format()
        role_response_prompt = ("You are a {} and you are given a piece of **Input Text** and an image. Your task is to predict whether misinformation is present in them based on your professional knowledge.\
The text and the image come from the same post (or the same news report), where the text serves as the content, and the image complements or provides evidence for the text,  \
Please predict whether this is a post containing misinformation by verifying the consistency of text and images and detecting whether text and images violate common sense.\
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
            "{{\n\
                \"label\": "",\n\
                \"explanation\":""\n\
            }}\n"
            "Input Text:\n"
            "{}\n"
            "Let's think step by step."
            "Your Response:\n").format(role,text)
    #     role_response_prompt = ("You are a {} and you are given a piece of **Input Text** and an image. Your task is to predict whether misinformation is present in them based on your expertise as a {}.\
    # The text and the image come from the same post (or the same news report), where the text serves as the content, and the image complements or provides evidence for the text,  \
    # Please predict whether this is a post containing misinformation by detecting whether text and images violate common sense and verifying if there is anything in the image that conflicts with the text.\
    # You will be punished if your answer is wrong. Please follow the Rules below:\n"
    #         "Rules:\n"
    #         "Generate a JSON object with three properties: 'label', 'explanation','certain'.\n" 
    #         "The return value of 'label' property should be selected from [\"real\", \"fake\"].\n"
    #         "real indicates that no misinformation is detected. \n"
    #         "fake indicates that misinformation is detected. \n"
    #         "The return value of 'explanation' property should be a detailed reasoning for the given 'label'. \n"
    #         "The return value of 'certain' property should be selected from [\"sure\",\"unsure\"].\n"
    #         "sure indicates that you are confident to your predict.\n"
    #         "unsure indicates that you are not confident to your predict.\n"
    #         "Note that your response will be passed to the python interpreter, SO NO OTHER WORDS! Just only output the json object. Do not add ```json in front of json object or behind it! \n"
    #         "Example output (JSON):\n"
    #         "{{\n\
    #             \"label\": "",\n\
    #             \"explanation\":""\n\
    #             \"certain\":""\n\
    #         }}\n"
    #         "Input Text:\n"
    #         "{}\n"
    #         "Let's think step by step."
    #         "Your Response:\n").format(role,role,text)
        #多角色
    #     role_prompt = ("you are given a piece of **Input Text** and an image. The text and the image come from the same post (or the same news report). \
    # Your task is to give three different roles that would be helpful in predicting whether misinformation is present at them based on the input text and image.\
    # Please follow the Rules below:\n"
    #             "Rules:\n"
    #             "Generate a JSON object with two properties: 'roles', 'explanation'.\n" 
    #             "The return value of 'roles' property must be three different roles selected in['politician','journalist','historian','fact-checker','image-analyst ','reasoning-expert','medical-experts'].\n"
    #             "The return value of 'explanation' property should be a detailed reasoning for the given 'role'. \n"
    #             "Note that your response will be passed to the python interpreter, SO NO OTHER WORDS! Just only output the json object. Do not add ```json in front of json object or behind it! \n"
    #             "Example output (JSON):\n"
    #             "{{\n\
    #                 \"role\": \"['','','']\",\n\
    #                 \"explanation\":""\n\
    #             }}\n"
    #             "Input Text:\n"
    #             "{}\n"
    #             "Let's think step by step."
    #             "Your Response:\n").format(text)
    #     output_text = chat_response(image_path,role_prompt,prob=0,type="mm")
    #     # print(output_text)
    #     output_text = output_text.split()
    #     # print(output_text)
    #     for i, item in enumerate(output_text):
    #         if "role" in item:
    #             index = i
    #             break
    #     role = output_text[index+1] 
    #     cleaned_string = role.strip("[]")
    #     words_list = [word.strip("'") for word in cleaned_string.split(",")]
    #     if len(words_list) < 3:
    #         words_list[1] = words_list[0]
    #         words_list.append(words_list[0])
    #     role_response_output = []
    #     role_response_output_logit = []
    #     for i in range(0,3):
    #         role = remove_punctuation_manual(words_list[i])
    #         role_response_prompt = ("You are a {} and you are given a piece of **Input Text** and an image. Your task is to predict whether misinformation is present in them based on your professional knowledge.\
    # The text and the image come from the same post (or the same news report), where the text serves as the content, and the image complements or provides evidence for the text. \
    # Please predict whether this is a post containing misinformation by verifying the consistency of text and images and detecting whether text and images violate common sense.\
    # Please follow the Rules below:\n"
    #             "Rules:\n"
    #             "Generate a JSON object with two properties: 'label', 'explanation'.\n" 
    #             "The return value of 'label' property should be selected from [\"real\", \"fake\"].\n"
    #             "real indicates that no misinformation is detected. \n"
    #             "fake indicates that misinformation is detected. \n"
    #             "The return value of 'explanation' property should be a detailed reasoning for the given 'label'. \n"
    #             "Note that your response will be passed to the python interpreter, SO NO OTHER WORDS! Just only output the json object. Do not add ```json in front of json object or behind it! \n"
    #             "Example output (JSON):\n"
    #             "{{\n\
    #                 \"label\": "",\n\
    #                 \"explanation\":""\n\
    #             }}\n"
    #             "Input Text:\n"
    #             "{}\n"
    #             "Let's think step by step."
    #             "Your Response:\n").format(role,text)
    #         temp_response,temp_logits = chat_response(image_path,role_response_prompt,prob=1,type="mm")
    #         role_response_output.append(temp_response)
    #         role_response_output_logit.append(temp_logits)

        #单角色
    #     role_prompt = ("you are given a piece of **Input Text** and an image. The text and the image come from the same post (or the same news report). \
    # Your task is to give a role that would be helpful in predicting whether misinformation is present at them based on the input text and image.\
    # Please follow the Rules below:\n"
    #             "Rules:\n"
    #             "Generate a JSON object with two properties: 'role', 'explanation'.\n" 
    #             "The return value of 'role' property must be a selected in['politician','journalist','historian','fact-checker','image-analyst ','reasoning-expert','medical-experts'].\n"
    #             "The return value of 'explanation' property should be a detailed reasoning for the given 'role'. \n"
    #             "Note that your response will be passed to the python interpreter, SO NO OTHER WORDS! Just only output the json object. Do not add ```json in front of json object or behind it! \n"
    #             "Example output (JSON):\n"
    #             "{{\n\
    #                 \"role\": \"\",\n\
    #                 \"explanation\":""\n\
    #             }}\n"
    #             "Input Text:\n"
    #             "{}\n"
    #             "Let's think step by step."
    #             "Your Response:\n").format(text)
        
    #     output_text = chat_response(image_path,role_prompt,prob=0,type="mm")
    #     # print(output_text)
    #     output_text = output_text.split()
    #     # print(output_text)
    #     for i, item in enumerate(output_text):
    #         if "role" in item:
    #             index = i
    #             break
    #     role = output_text[index+1] 
    #     role = remove_punctuation_manual(role)    

    #     role_response_prompt = ("You are a {} and you are given a piece of **Input Text** and an image. Your task is to predict whether misinformation is present in them based on your professional knowledge.\
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
    #             "{{\n\
    #                 \"label\": "",\n\
    #                 \"explanation\":""\n\
    #                 \"certain\":""\n\
    #             }}\n"
    #             "Input Text:\n"
    #             "{}\n"
    #             "Let's think step by step."
    #             "Your Response:\n").format(role,text)

    #     role_response_prompt = ("You are a {} and you are given a piece of **Input Text** and an image. Your task is to predict whether misinformation is present in them based on your professional knowledge.\
    # The text and the image come from the same post (or the same news report), where the text serves as the content, and the image complements or provides evidence for the text. \
    # Please predict whether this is a post containing misinformation by verifying the consistency of text and images and detecting whether text and images violate common sense.\
    # You will be punished if your answer is wrong. Please follow the Rules below:\n"
    #             "Rules:\n"
    #             "Generate a JSON object with two properties: 'label', 'explanation'.\n" 
    #             "The return value of 'label' property should be selected from [\"real\", \"fake\"].\n"
    #             "real indicates that no misinformation is detected. \n"
    #             "fake indicates that misinformation is detected. \n"
    #             "The return value of 'explanation' property should be a detailed reasoning for the given 'label'. \n"
    #             "Note that 'label' and 'explanation' should be consistent in their judgment of whether or not the news contains misinformation.\n"
    #             "Note that your response will be passed to the python interpreter, SO NO OTHER WORDS! Just only output the json object. Do not add ```json in front of json object or behind it! \n"
    #             "Example output (JSON):\n"
    #             "{{\n\
    #                 \"label\": "",\n\
    #                 \"explanation\":""\n\
    #             }}\n"
    #             "Input Text:\n"
    #             "{}\n"
    #             "Let's think step by step."
    #             "Your Response:\n").format(role,text)
        
        # mm_prompt = ("You are given a piece of **Input Text** and an image. Your task is to predict whether misinformation is present.\
        # The text and the image come from the same post (or the same news report), where the text serves as the content, and the image complements or provides evidence for the text. \
        # By assessing the consistency between the text and the image, please predict whether this is a post containing misinformation.Please follow the Rules below:\n"
        # "Rules:\n"
        # "Generate a JSON object with two properties: 'label', 'explanation'.\n" 
        # "The return value of 'label' property should be selected from [\"real\", \"fake\"].\n"
        # "real indicates that no misinformation is detected. \n"
        # "fake indicates that misinformation is detected. \n"
        # "The return value of 'explanation' property should be a detailed reasoning for the given 'label'. \n"
        # "Note that your response will be passed to the python interpreter, SO NO OTHER WORDS! Just only output the json object. Do not add ```json in front of json object or behind it! \n"
        # "Example output (JSON):\n"
        # "{{\n\
        #     \"label\": "",\n\
        #     \"explanation\":""\n\
        # }}\n"
        # "Input Text:\n"
        # "{}\n"
        # "Let's think step by step."
        # "Your Response:\n").format(text)
        
        # text_output,text_logit_prob = chat_response(image_path,text_prompt,prob=1,type="text")
        # image_output,image_logit_prob = chat_response(image_path,image_prompt,prob=1,type="image")
        # mm_output,mm_output_logits = chat_response(image_path,role_response_prompt,prob=1,type="mm")
        
        # text_output,text_real_prob,text_fake_prob,text_certain = chat_response(image_path,text_prompt,prob=2,type="text")
        # image_output,image_real_prob,image_fake_prob,image_certain = chat_response(image_path,image_prompt,prob=2,type="image")
        # mm_output,mm_real_logits,mm_fake_logits,mm_certain = chat_response(image_path,role_response_prompt,prob=2,type="mm")
        
        text_output,text_real_prob,text_fake_prob = chat_response(image_path,text_prompt,prob=1,type="text")
        image_output,image_real_prob,image_fake_prob= chat_response(image_path,image_prompt,prob=1,type="image")
        mm_output,mm_real_logits,mm_fake_logits = chat_response(image_path,role_response_prompt,prob=1,type="mm")
        # print(text_output)

        text_label = get_label(text_output)
        image_label = get_label(image_output)
        mm_label = get_label(mm_output)

        # text_ = get_certain(text_output)
        # image_ = get_certain(image_output)
        # mm_ = get_certain(mm_output)


        # if text_label == image_label == mm_label:
        weather_knowledge = 0
        # else:
        #     weather_knowledge = 1       

        
        #加外部知识 分开加
        # if max(text_real_prob,text_fake_prob)<0.6 or max(image_real_prob,image_fake_prob)<0.6 or max(mm_real_logits,mm_fake_logits) <0.6:
        extra_knowledge = ast.literal_eval(knowledge)
        image_extra_knowledge = ast.literal_eval(image_knowledge)
        # print(type(extra_knowledge))
        question = questions[y]
        image_question = image_questions[y]
        tfidf_vectorizer = TfidfVectorizer()
        similarity = []
        similarity2 = []
        similarity3 = []
        for a in range(len(extra_knowledge)):
            documents = [text, extra_knowledge[a]['body']]
            documents2 = [question, extra_knowledge[a]['body']]
            documents3 = [image_question,image_extra_knowledge[a]['body']]
            tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
            tfidf_matrix2 = tfidf_vectorizer.fit_transform(documents2)
            tfidf_matrix3 = tfidf_vectorizer.fit_transform(documents3)
            similarity.append(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2]))
            similarity2.append(cosine_similarity(tfidf_matrix2[0:1], tfidf_matrix2[1:2]))
            similarity3.append(cosine_similarity(tfidf_matrix3[0:1], tfidf_matrix3[1:2]))
            # print(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2]))
            # print(cosine_similarity(tfidf_matrix2[0:1], tfidf_matrix2[1:2]))  
        max_index = 0
        max_similarity = 0
        for a in range(len(similarity)):
            if similarity[a]+similarity2[a] > max_similarity:
                max_index = a
                max_similarity = similarity[a]+similarity2[a]
        final_image_knowledge = image_extra_knowledge[similarity3.index(max(similarity3))]['body']
        final_knowledge = extra_knowledge[max_index]['body']      
        print(f"final knowledge:{final_knowledge}")  

        #外部知识库
        # extra_knowledge = ast.literal_eval(knowledge)
        # image_extra_knowledge = ast.literal_eval(image_knowledge)
        # total_knowledge = extra_knowledge + image_extra_knowledge
        # total_knowledge_body = []
        # for a in range(len(total_knowledge)):
        #     total_knowledge_body.append(total_knowledge[a]['body'])
        # total_knowledge_body = total_knowledge_body + true_knowledeg_list    
        # # print(type(extra_knowledge))
        # question = questions[y]
        # image_question = image_questions[y]
        # tfidf_vectorizer = TfidfVectorizer()
        # similarity = []
        # similarity2 = []
        # similarity3 = []
        # for a in range(len(total_knowledge_body)):
        #     documents = [text, total_knowledge_body[a]]
        #     documents2 = [question, total_knowledge_body[a]]
        #     documents3 = [image_question,total_knowledge_body[a]]
        #     tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
        #     tfidf_matrix2 = tfidf_vectorizer.fit_transform(documents2)
        #     tfidf_matrix3 = tfidf_vectorizer.fit_transform(documents3)
        #     similarity.append(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2]))
        #     similarity2.append(cosine_similarity(tfidf_matrix2[0:1], tfidf_matrix2[1:2]))
        #     similarity3.append(cosine_similarity(tfidf_matrix3[0:1], tfidf_matrix3[1:2]))
        #     # print(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2]))
        #     # print(cosine_similarity(tfidf_matrix2[0:1], tfidf_matrix2[1:2]))  
        # max_index = 0
        # max_similarity = 0
        # for a in range(len(similarity)):
        #     if similarity[a]+similarity2[a] > max_similarity:
        #         max_index = a
        #         max_similarity = similarity[a]+similarity2[a]
        # final_image_knowledge = total_knowledge_body[similarity3.index(max(similarity3))]
        # final_knowledge = total_knowledge_body[max_index]   
        # print(f"final knowledge:{final_knowledge}") 
        # if text_logit_prob <= 0.55:
        
        text_prompt = ("A language model was asked: Predict whether the news is real or fake according to the text of the news.The text of the news is {}.\
The model's answer was: {}.\n"
"The model is not very sure of its answer. Give you an additional paragraph of external knowledge about the news that\
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
    "{{\n\
        \"label\": "",\n\
        \"explanation\":""\n\
    }}\n"
    "External knowledge:"
    "{}"
    "Let's think step by step."
    "Your Response:\n").format(text,text_output,final_knowledge)
        
        text_output,text_real_prob,text_fake_prob = chat_response(image_path,text_prompt,prob=1,type="text")
        text_label = get_label(text_output)

        image_prompt = ("A language model was asked: Predict whether the news is real or fake according to image of the news.The image of the news is as followed.\
The model's answer was: {}.\n"
"The model is not very sure of its answer. Give you an additional paragraph of external knowledge about the news that\
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
    "{{\n\
        \"label\": "",\n\
        \"explanation\":""\n\
    }}\n"
    "External knowledge:"
    "{}"
    "Let's think step by step."
    "Your Response:\n").format(image_output,final_image_knowledge)
        image_output,image_real_prob,image_fake_prob = chat_response(image_path,image_prompt,prob=1,type="image")
        image_label = get_label(image_output)

        mm_prompt = ("A language model was asked: Predict whether the news is real or fake according to the text and image of the news.The text of the news is {}.The image of the news is as followed.\
The model's answer was: {}.\n"
"The model is not very sure of its answer. Give you an additional paragraph of external knowledge about the news that\
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
    "{{\n\
        \"label\": "",\n\
        \"explanation\":""\n\
    }}\n"
    "External knowledge:"
    "{}"
    "Let's think step by step."
    "Your Response:\n").format(text,mm_output,final_knowledge)
        mm_output,mm_real_logits,mm_fake_logits = chat_response(image_path,mm_prompt,prob=1,type="mm")
        mm_label = get_label(mm_output)

        current_label_list = [text_label,image_label,mm_label]   
        index_list = ["text","image","mm"]
        current_real_logits = [text_real_prob,image_real_prob,mm_real_logits]
        current_fake_logits = [text_fake_prob,image_fake_prob,mm_fake_logits]
        # final_label_list = []    
        # final_real_logits = []
        # final_fake_logits = []
        g = 0
        for item in (text_output,image_output,mm_output):
            check_prompt = ("A language model was asked: Predict whether the news is real or fake according to the text and image of the news.\
The model's answer was: {}.\n"
"The model's answer has two properties: 'label', 'explanation'.\n"
"The return value of 'label' property is selected from [\"real\", \"fake\"].\n"
"real indicates that no misinformation is detected. \n"
"fake indicates that misinformation is detected. \n"
"The return value of 'explanation' property is a detailed reasoning for the given 'label'. \n"
"Your task is to check that 'label' and 'explanation' of the model's answer are consistent in their judgment of whether or\
 not the news contains misinformation."
"Please follow the Rules below:\n"
"If the 'label' and 'explanation' of the model's answer are not consistent,correct the 'label' according to the 'explanation' and output the corrected 'label'." 
"If the 'label' and 'explanation' of the model's answer are consistent,output the original 'label'"
"Note that your response will be passed to the python interpreter, SO NO OTHER WORDS! Just only output the 'label'."
            "Your Response:\n").format(item)
            check_output,check_real,check_fake = chat_response("111",check_prompt,prob=1,type="text")
            if check_output != current_label_list[g]:
                print(index_list[g]+"进行了纠正")  
                index_list[g] = "corrected_text"  
                current_label_list[g] = check_output
                ttt = current_real_logits[g]
                current_real_logits[g] = current_fake_logits[g]
                current_fake_logits[g] = ttt
            g = g + 1    
            # print(check_output)
            # final_label_list.append(check_output)
            # final_real_logits.append(check_real)
            # final_fake_logits.append(check_fake)
        # mm_output,mm_logit_prob = chat_response(image_path,mm_prompt,prob=1,type="mm")
        #多角色
        # mm_label1 = get_label(role_response_output[0])
        # mm_label2 = get_label(role_response_output[1])
        # mm_label3 = get_label(role_response_output[2])            
        #多角色
        # labellist = [mm_label1,mm_label2,mm_label3,text_label,image_label]
        # problist = [role_response_output_logit[0],role_response_output_logit[1],role_response_output_logit[2],text_logit_prob,image_logit_prob]
        #单角色投票
        # labellist = [mm_label,text_label,image_label]
        # problist = [max(mm_real_logits,mm_fake_logits),max(text_real_prob,text_fake_prob),max(image_fake_prob,image_real_prob)]
        
        # realnum = 0
        # fakenum = 0
        # index = 0
        # fakemaxprob = 0
        # fakeminprob = 1
        # realmaxprob = 0
        # realminprob = 1
        # realindex = 0
        # fakeindex = 0
        # realminindex = 0
        # fakeminindex = 0
        # for i in range(len(labellist)):
        #     if labellist[i] == "real":
        #         if problist[i] > realmaxprob:
        #             realmaxprob = problist[i]
        #             realindex = i
        #         if problist[i] < realminprob:
        #             realminprob = problist[i]   
        #             realminindex = i  
        #         realnum = realnum + 1 
        #     else:
        #         if problist[i] > fakemaxprob:
        #             fakemaxprob = problist[i]
        #             fakeindex = i
        #         if problist[i] < fakeminprob:
        #             fakeminprob = problist[i]   
        #             fakeminindex = i
        #         fakenum = fakenum + 1
            
        # if realnum > fakenum:
        #     major_label = "real"
        # else:
        #     major_label = "fake"
        # for i in range(len(problist)):
        #     if problist[realindex] > problist[fakeindex]:
        #         if major_label == "real":
        #             final_label = "real"
        #         else:
        #             if problist[realindex]-problist[fakeindex] > 0.35:
        #                 final_label = "real"
        #             else:
        #                 final_label = "fake"
        #     else:
        #         if major_label == "fake":
        #             final_label = "fake"
        #         else:
        #             if problist[fakeindex]-problist[realindex] > 0.35:
        #                 final_label = "fake"
        #             else:
        #                 final_label = "real"
        #CQP prompt
#         text_CQP_prompt = ("A language model was asked: Predict whether the news is real or fake.The text of the news is{}.\
# Options were: [real,fake].\
# The model's answer was: {}.\n"
# "Analyse its answer given other options.How certain are you of the model's answer?\n"
# "a. Very Certain\n"
# "b. Fairly Certain\n"
# "c. Moderately Certain\n"
# "d. Somewhat Certain\n"
# "e. Not Certain\n"
# "f. Very Uncertain\n"
# "Only output the six option above,NO OTHER WORDS!"
# "Your response:").format(text,text_label)
#         text_CQP = chat_response(image_path,text_CQP_prompt,prob=0,type="text")
#         try:
#             if text_CQP == "a. Very Certain":
#                 text_CQP_value = 1
#             elif text_CQP == "b. Fairly Certain":
#                 text_CQP_value = 0.8    
#             elif text_CQP == "c. Moderately Certain":
#                 text_CQP_value = 0.6   
#             elif text_CQP == "d. Somewhat Certain":
#                 text_CQP_value = 0.4   
#             elif text_CQP == "e. Not Certain":
#                 text_CQP_value = 0.2        
#             elif text_CQP == "f. Very Uncertain":
#                 text_CQP_value = 0.0 
#             print(text_CQP)
#             print(text_CQP_value)      
#         except:
#             if "very" in text_CQP.lower():
#                 if "f" in text_CQP.lower():
#                     text_CQP_value = 0.0 
#                 else:
#                     text_CQP_value = 1
#             elif "fairly" in text_CQP.lower():
#                 text_CQP_value = 0.8    
#             elif "moderately" in text_CQP.lower():
#                 text_CQP_value = 0.6 
#             elif "somewhat" in text_CQP.lower():
#                 text_CQP_value = 0.4 
#             elif "Not" in text_CQP.lower():
#                 text_CQP_value = 0.2                 
#             print(text_CQP)
#             print(text_CQP_value)
#         image_CQP_prompt = ("A language model was asked: Predict whether the news is real or fake.The image of the news is as followed. \
# Options were: [real,fake].\
# The model's answer was: {}.\n"
# "Analyse its answer given other options.How certain are you of the model's answer?\n"
# "a. Very Certain\n"
# "b. Fairly Certain\n"
# "c. Moderately Certain\n"
# "d. Somewhat Certain\n"
# "e. Not Certain\n"
# "f. Very Uncertain\n"
# "Only output the six option above,NO OTHER WORDS!"
# "Your response:").format(image_label)
#         image_CQP = chat_response(image_path,image_CQP_prompt,prob=0,type="mm")
#         try:
#             if image_CQP == "a. Very Certain":
#                 image_CQP_value = 1
#             elif image_CQP == "b. Fairly Certain":
#                 image_CQP_value = 0.8    
#             elif image_CQP == "c. Moderately Certain":
#                 image_CQP_value = 0.6   
#             elif image_CQP == "d. Somewhat Certain":
#                 image_CQP_value = 0.4   
#             elif image_CQP == "e. Not Certain":
#                 image_CQP_value = 0.2        
#             elif image_CQP == "f. Very Uncertain":
#                 image_CQP_value = 0.0  
#             print(image_CQP)
#             print(image_CQP_value)    
#         except:        
#             if "very" in image_CQP.lower():
#                 if "f" in image_CQP.lower():
#                     image_CQP_value = 0.0 
#                 else:
#                     image_CQP_value = 1
#             elif "fairly" in image_CQP.lower():
#                 image_CQP_value = 0.8    
#             elif "moderately" in image_CQP.lower():
#                 image_CQP_value = 0.6 
#             elif "somewhat" in image_CQP.lower():
#                 image_CQP_value = 0.4 
#             elif "Not" in image_CQP.lower():
#                 image_CQP_value = 0.2                  
#             print(image_CQP)
#             print(image_CQP_value)
#         mm_CQP_prompt = ("A language model was asked: As a {} to predict whether the news is real or fake.The text of the news is{},the image is as followed. \
# Options were: [real,fake].\
# The model's answer was: {}.\n"
# "Analyse its answer given other options.How certain are you of the model's answer?\n"
# "a. Very Certain\n"
# "b. Fairly Certain\n"
# "c. Moderately Certain\n"
# "d. Somewhat Certain\n"
# "e. Not Certain\n"
# "f. Very Uncertain\n"
# "Only output the six option above,NO OTHER WORDS!"
# "Your response:").format(role,text,mm_label)
#         mm_CQP = chat_response(image_path,mm_CQP_prompt,prob=0,type="mm")
#         try:
#             if mm_CQP == "a. Very Certain":
#                 mm_CQP_value = 1
#             elif mm_CQP == "b. Fairly Certain":
#                 mm_CQP_value = 0.8    
#             elif mm_CQP == "c. Moderately Certain":
#                 mm_CQP_value = 0.6   
#             elif mm_CQP == "d. Somewhat Certain":
#                 mm_CQP_value = 0.4   
#             elif mm_CQP == "e. Not Certain":
#                 mm_CQP_value = 0.2        
#             elif mm_CQP == "f. Very Uncertain":
#                 mm_CQP_value = 0.0           
#             print(f"---{mm_CQP}---")
#             print(mm_CQP_value)
#         except:
#             if "very" in mm_CQP.lower():
#                 if "f" in mm_CQP.lower():
#                     mm_CQP_value = 0.0 
#                 else:
#                     mm_CQP_value = 1
#             elif "fairly" in mm_CQP.lower():
#                 mm_CQP_value = 0.8    
#             elif "moderately" in mm_CQP.lower():
#                 mm_CQP_value = 0.6 
#             elif "somewhat" in mm_CQP.lower():
#                 mm_CQP_value = 0.4 
#             elif "Not" in mm_CQP.lower():
#                 mm_CQP_value = 0.2  
#             print(mm_CQP)
#             print(mm_CQP_value)                           
#         #单角色概率融合
#         final_real_prob = 0
#         final_fake_prob = 0
        # final_real_prob = text_CQP_value*text_real_prob + image_CQP_value*image_real_prob + mm_CQP_value*mm_real_logits
        # final_fake_prob = text_CQP_value*text_fake_prob + image_CQP_value*image_fake_prob + mm_CQP_value*mm_fake_logits
        final_real_prob = sum(current_real_logits)
        final_fake_prob = sum(current_fake_logits)
        # final_real_prob = text_real_prob + image_real_prob + mm_real_logits
        # final_fake_prob = text_fake_prob + image_fake_prob + mm_fake_logits
        final_label = "real" if final_real_prob>final_fake_prob else "fake"
#         if text_real_prob > text_fake_prob:
#             final_real_prob = final_real_prob + text_CQP_value*text_real_prob
#             final_fake_prob = final_fake_prob + (1-text_CQP_value)*text_fake_prob
#         else:
#             final_real_prob = final_real_prob + (1-text_CQP_value)*text_real_prob
#             final_fake_prob = final_fake_prob + text_CQP_value*text_fake_prob  

#         if image_real_prob > image_fake_prob:
#             final_real_prob = final_real_prob + image_CQP_value*image_real_prob
#             final_fake_prob = final_fake_prob + (1-image_CQP_value)*image_fake_prob
#         else:
#             final_real_prob = final_real_prob + (1-image_CQP_value)*image_real_prob
#             final_fake_prob = final_fake_prob + image_CQP_value*image_fake_prob     

#         if mm_real_logits > mm_fake_logits:
#             final_real_prob = final_real_prob + mm_CQP_value*mm_real_logits
#             final_fake_prob = final_fake_prob + (1-mm_CQP_value)*mm_fake_logits
#         else:
#             final_real_prob = final_real_prob + (1-mm_CQP_value)*mm_real_logits
#             final_fake_prob = final_fake_prob + mm_CQP_value*mm_fake_logits
#         final_label = "real" if final_real_prob>final_fake_prob else "fake" 
        #多数投票
        # label_list = [text_label,image_label,mm_label]
        # fake_index = 0
        # real_index = 0
        # for a in range(len(label_list)):
        #     if label_list[a] == "real":
        #         real_index = real_index + 1
        #     else:
        #         fake_index = fake_index + 1    
        # final_label = "real" if real_index>fake_index else "fake"        
        #multi-role    
        # print(f"第{x}条数据,原文本{text},\n原label:{label}\n文本输出:{text_label}\n图像输出:{image_label}\n角色{words_list[0]}的mm输出:{mm_label1}\n角色{words_list[1]}mm输出:{mm_label2}\n角色{words_list[2]}mm输出:{mm_label3}\n最终输出label:{final_label}")
        #single-role
        # print(f"第{x}条数据,原文本{text},\n原label:{label}\n文本输出:{text_label}\n图像输出:{image_label}\n角色{role}的mm输出:{mm_label}\n最终输出label:{final_label}")
        print(f"第{x}条数据,原文本{text},\n原label:{label}\n文本输出:{current_label_list[0]}\n图像输出:{current_label_list[1]}\n角色{role}的mm输出:{current_label_list[2]}\n最终输出label:{final_label}")
        
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
    
                