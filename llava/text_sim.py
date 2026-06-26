from sentence_transformers import SentenceTransformer
import json
import os
import gc
import torch
import string
import requests
import csv
import ast
import re
model = SentenceTransformer("/home/jncsnlp4/tb/Qwen2-VL-main/sentence-transformers/all-MiniLM-L6-v2")

with open("/home/jncsnlp4/SSD2/tb/data/MR2-en/dataset_items_test_filtered.json",'r',encoding='utf-8') as file:
    data = json.load(file)

with open("/home/jncsnlp4/tb/Qwen2-VL-main/MR2_knowledev/text_knowledge_ours_MR2-en2.csv",mode='r',encoding="utf-8")as file:
    csv_dict_reader = csv.reader(file)
    rows = list(csv_dict_reader)


with open('/home/jncsnlp4/tb/Qwen2-VL-main/MR2_knowledev/image_knowledge_MR2-en2.csv',mode='r',encoding="utf-8")as file:
    csv_dict_reader = csv.reader(file)
    image_rows = list(csv_dict_reader)

with open('/home/jncsnlp4/tb/Qwen2-VL-main/MR2_knowledev/MR2_ours_text_question2.txt','r',encoding = 'utf-8') as file:
    text_questions = file.readlines()

with open('/home/jncsnlp4/tb/Qwen2-VL-main/MR2_knowledev/MR2_knowledevimage_question.txt','r',encoding='utf-8') as file:
    image_questions = file.readlines()

def text_sim(sentence1,sentence2):
    # Compute embeddings for both lists
    embeddings1 = model.encode(sentence1)
    embeddings2 = model.encode(sentence2)

    # Compute cosine similarities
    similarities = model.similarity(embeddings1, embeddings2)

    # print(similarities.item())
    return similarities.item()
trusted_website = {"https://en.wikipedia.org/","https://www.bbc.com/news","https://www.nytimes.com","https://www.theguardian.com","https://www.reuters.com","https://www.ap.org","https://www.wikipedia.org/"}

text_index = []
image_index = []
fake_num = 0
real_num = 0
num = -1
for item in data.values():
    image_path = "/home/jncsnlp4/SSD2/tb/data/MR2-en/" + item['image_path'] #fakeddit
    if os.path.exists(image_path):
        text = item['caption']
        label = int(item['label'])
        num = num + 1  
        knowledge = rows[num][1]
        image_knowledge = image_rows[num][1]
        text_extra_knowledge = ast.literal_eval(knowledge)
        if image_knowledge == "":
            image_extra_knowledge = []
        else:
            image_extra_knowledge = ast.literal_eval(image_knowledge)
        text_question = text_questions[num]
        image_question = image_questions[num]
        text_max_index = 0
        text_max_similarity = 0
        for a in range(len(text_extra_knowledge)):
            link = text_extra_knowledge[a]['href']
            domain = re.search('https?://([A-Za-z_0-9.-]+).*', link).group(1)
            weight = 1
            if domain in trusted_website:
                weight = 1.5
            temp = text_sim(text,text_extra_knowledge[a]['body'])
            temp2 = text_sim(text_question,text_extra_knowledge[a]['body'])
            if weight*(temp + temp2) > text_max_similarity:
                text_max_similarity = weight*(temp + temp2)
                text_max_index = a
        print(text_max_index,text_max_similarity)
        final_text_knowledge = text_extra_knowledge[text_max_index]['body']
        image_max_index = 0
        image_max_similarity = 0
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
        text_index.append(text_max_index)
        image_index.append(image_max_index)
        print(f"现在是第{num}条数据")
    else:
        continue    
with open("test_index/MR_test_text_max_index.txt",'w',encoding='utf-8') as file:
    file.write(str(text_index))
with open("test_index/MR_test_image_max_index.txt",'w',encoding='utf-8') as file:
    file.write(str(image_index))        
# Output the pairs with their score
# for idx_i, sentence1 in enumerate(sentences1):
#     print(sentence1)
#     for idx_j, sentence2 in enumerate(sentences2):
#         print(f" - {sentence2: <30}: {similarities[idx_i][idx_j]:.4f}")