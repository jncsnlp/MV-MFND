import csv
import time
import pandas as pd
from ddgs import DDGS
import json
from pprint import pprint

with open('MR2_knowledevimage_question.txt',mode='r',encoding="utf-8")as file:
    image_rows = file.readlines()

for x in range(len(image_rows)):

    mm_query = image_rows[x].strip().lower()
    mm_query = mm_query.replace('"', '')
    print(mm_query)
    mm_results = DDGS().text(mm_query, region='en', safesearch='off', max_results=5)
    with open("image_knowledge_MR2-en.csv", 'a', encoding='utf-8') as file2:
        writer = csv.writer(file2)
        writer.writerow([x, mm_results])
    print(f"{x}条数据已完成")
    time.sleep(4)
