import os
import json
import argparse
import string
import pandas as pd
import ast
import re
import torch
import csv
# from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
# from qwen_vl_utils import process_vision_info
from utils import topic_relevance_filter,evidence_extraction
from lemma_component import LemmaComponent
# from retrieval import get_evidence, visual_search, driver_quit
from configs import out_root, definition_path
from utils import save, process_multilines_output, perror
import traceback
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     "/home/jncsnlp4/lxt/model/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
# )
# processor = AutoProcessor.from_pretrained("/home/jncsnlp4/lxt/model/Qwen2-VL-7B-Instruct")
rumor_types=["true", "satire/parody", "misleading content", "text image contradiction", "manipulated content", "unverified"]
            

def get_label(final_output,target):
    output_text = final_output.split()
    for i, item in enumerate(output_text):
        if target in item:
            index = i
            break
    # print(output_text)    
    output = output_text[index+1] + output_text[index+2]
    return output
def find_label(refine_output):
    refine_output = refine_output.lower()
    # print(refine_output)
    for i,item in enumerate(rumor_types):
        if item in refine_output:
            return item
    return "unverified"

def remove_punctuation_manual(text):
    return ''.join(char for char in text if char not in string.punctuation)


def get_part(text,target,end_word):
    start = text.find(target) + len(target)
    remaining = text[start:]
    end = remaining.find(end_word)
    explanation_text = remaining[:end]

    return explanation_text

def source_filter(results):
    global untrusted_sources
    if results == None or results == []:
        return []
    for result in results:
        link=result['href']
        domain = re.search('https?://([A-Za-z_0-9.-]+).*', link).group(1)
        if domain in untrusted_sources:   
            results.remove(result)
    return results





# Arg parser
parser = argparse.ArgumentParser()
parser.add_argument('--input_file_name', type=str, default='exampleinput.json', help='Input file name')
parser.add_argument('--use_cache', action='store_true', default=False, help='Use cache for modules except final prediction')
parser.add_argument('--resume', action='store_true', default=False, help='Resume from the last time')
parser.add_argument('--use_offline_image', action='store_true', default=False, help='Use offline image flag')
args = parser.parse_args()

# Input file
input_file = args.input_file_name

with open(input_file, encoding='utf-8') as file:
    data = json.load(file)
# knowledge = pd.read_csv("/home/jncsnlp4/tb/Qwen2-VL-main/extra_knowledge_LEMMA2.csv", header=None) ##fakeddit
knowledge = pd.read_csv("/home/jncsnlp4/tb/LEMMA-main/mm_knowledge_MR2-en2.csv", header=None) ##twitter
# all_query = pd.read_csv("/home/jncsnlp4/tb/Qwen2-VL-main/search_quesion.csv", header=None) #fakeddit
all_query = pd.read_csv("/home/jncsnlp4/tb/LEMMA-main/search_quesion_MR2.csv", header=None) #twitter
# picture_knowledge = pd.read_csv("/home/jncsnlp4/tb/Qwen2-VL-main/extra_picture_knowledge_LEMMA2.csv",header=None) #fakeddit
picture_knowledge = pd.read_csv("/home/jncsnlp4/tb/LEMMA-main/extra_picture_knowledge_LEMMA_MR22.csv",header=None) #twitter
untrusted_sources={"www.reddit.com","www.weibo.com","twitter.com","www.tiktok.com","www.douyin.com","www.instagram.com","www.taobao.com","www.jd.com","www.amazon.com","www.ebay.com","www.imdb.com","www.douban.com","steamcommunity.com","m.ixigua.com","www.bilibili.com","www.netflix.com",}

# Output file
# output_score = out_root + "llava_lemma_fakeedit"+"score"
# output_result = out_root + "llava_lemma_fakeedit"+"output.json"
# output_score = out_root + "llava_lemma_twitter"+"score"
# output_result = out_root + "llava_lemma_twitter"+"output.json"
output_score = out_root + "llava_lemma_MR2第三次"+"score"
output_result = out_root + "llava_lemma_MR2第三次"+"output.json"
if not os.path.exists(out_root):
    os.makedirs(out_root)   
    
# Resume
labels = []
direct_labels = []
final_preds = []
total_data_size = len(data)
current_index = -1
logger = []
if args.resume and os.path.exists(output_score):
    with open(output_score, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        labels = []
        for char in lines[1]:
            if char.isdigit():
                labels.append(int(char))
        direct_labels = []
        for char in lines[3]:
            if char.isdigit():
                direct_labels.append(int(char))
        final_preds = []
        for char in lines[5]:
            if char.isdigit():
                final_preds.append(int(char))
        current_index = int(lines[6].split(':')[1].strip())
    with open(output_result, 'r', encoding='utf-8') as f:
        logger = json.load(f)
    total_data_size = len(data)
    data = data[current_index + 1:]
if current_index==-1:
    print('Starting from index 0')
else:
    print('Resuming from index:', current_index, ', Next index:', current_index + 1)


# LEMMA Components Initialization
# direct_module = LemmaComponent(prompt='lemma_direct.md', name='Direct', model='Qwen-2-vl', using_cache=args.use_cache,
#                                   online_image=args.use_offline_image, max_retry=3, max_tokens=1000, temperature=0.1,
#                                   post_process=lambda x: json.loads(x))
direct_module = LemmaComponent(prompt='lemma_direct.md', name='Direct', model='Qwen-2-vl', using_cache=args.use_cache,
                                  online_image=args.use_offline_image, max_retry=3, max_tokens=1000, temperature=0.5)
# external_knowledge_module = LemmaComponent(prompt='external_knowledge.md', name='external_knowledge',       
#                                   model='Qwen-2-vl', using_cache=args.use_cache,
#                                   online_image=args.use_offline_image, max_retry=3, max_tokens=1000, temperature=0.1,
#                                   post_process=lambda x: json.loads(x))
external_knowledge_module = LemmaComponent(prompt='external_knowledge.md', name='external_knowledge',       
                                  model='Qwen-2-vl', using_cache=args.use_cache,
                                  online_image=args.use_offline_image, max_retry=3, max_tokens=1000, temperature=0.5)                                  
# question_gen_module = LemmaComponent(prompt='question_gen.md', name='question_gen', model='gpt4v', using_cache=args.use_cache,
#                                      online_image=args.use_offline_image, max_retry=3, max_tokens=1000, temperature=0.1,
#                                      post_process=lambda x: json.loads(x))
# refine_prediction_module = LemmaComponent(prompt='refined_prediction.md', name='modify_reasoning', model='Qwen-2-vl',
#                                          using_cache=False,
#                                          online_image=args.use_offline_image, max_retry=3, max_tokens=1000, temperature=0.1,
#                                          post_process=process_multilines_output)
question_gen_module = LemmaComponent(prompt='question_gen.md', name='question_gen', model='Qwen-2-vl', using_cache=args.use_cache,
                                     online_image=args.use_offline_image, max_retry=3, max_tokens=1000, temperature=0.5)
refine_prediction_module = LemmaComponent(prompt='refined_prediction.md', name='modify_reasoning', model='Qwen-2-vl',
                                         using_cache=False,
                                         online_image=args.use_offline_image, max_retry=3, max_tokens=1000, temperature=0.5)
# Test
i = 0
j = 0
# i = 267
# j = 267
x = -1
for item in data.values():
    # url = "/home/jncsnlp4/tb/prompt-llava/picture/post" + str(x) + ".jpg" ##fakeddit dataset
    x = x + 1
    text = item['caption']
    label = int(item['label'])
    url = "/home/jncsnlp4/SSD2/tb/data/MR2-en/" + item['image_path']
    if os.path.exists(url):
        current_index += 1
        print('Processing index {}/{}'.format(current_index, total_data_size))

        # Get input data
        # url = item["image_url"]
        

        # Direct Prediction
        direct = str(direct_module(TEXT=text, image=url))
        print(direct)
        
        if direct is None: continue
        # direct_pred = 0 if "real" in direct['label'].lower() else 1
        # direct_explain = direct['explanation']
        direct_pred = 0 if "real" in get_label(direct,target="label").lower() else 1 
        direct_explain = remove_punctuation_manual(get_part(direct,"explanation","}"))
        print(f"direct_pred{direct_pred},direct_explain{direct_explain}")

        # External Knowledge
        # Decide whether external knowledge is needed to further examine the input sample
        decision_external = str(external_knowledge_module(REASONING = direct_explain, TEXT = text, image=url))
        # direct_external = 0 if "no" in decision_external['external knowledge'].lower() else 1 
        direct_external = 1 if "yes" in get_label(decision_external,"external").lower() else 0
        print("######################")
        print("Need External Knowledge:", direct_external)
        print(decision_external)
        print("######################")

        retrieved_text = None
        all_search_results = {}
        titles_seen = set()
        if knowledge.iloc[i,0] == x:
            title = all_query.iloc[j,1]
            j = j + 3
            # print(title)
        weather_retri = 0
        query_set = []
        # if direct_external == 1:
        #     # Query Generation
        #     question_gen = question_gen_module(TEXT=text, 
        #                                         PREDICTION=direct_pred, 
        #                                         REASONING=direct_explain,
        #                                         image=url)
            
        #     if question_gen is None: continue
        #     # title, questions = question_gen['title'], question_gen['questions']
        #     try:
        #         question_gen = json.loads(question_gen)
        #         title, questions = question_gen['title'], question_gen['questions']
        #     except:
        #         print("发生异常了---------------")
        #         title = "fake new " + get_part(question_gen,"title","questions")
        #         questions = get_part(question_gen,"questions","]")
        #     # print(title)
        #     # print(questions[0],questions[1])
        #     # print("Lemma Component Evidence Retrieval: Starting...")
        #     print(title)
        #     print(questions)
        #     # 保存检索question
        #     with open("search_quesion_MR2.csv",'a',encoding="utf-8") as file:
        #         writer = csv.writer(file)
        #         writer.writerow([x,"fake new "+title])
        #     for i,question in enumerate(questions):
        #         with open("search_quesion_MR2.csv",'a',encoding="utf-8") as file:
        #             writer = csv.writer(file)
        #             writer.writerow([x,question])    
            # Evidence Retrieval
            # print("Lemma Component Evidence Retrieval: Starting...")
            # try:
            #     retrieved_text = get_evidence(text, title, questions)
            # except Exception as e:
            #     perror(traceback.format_exc())
            #     retrieved_text = ""
        while knowledge.iloc[i,0] == x:
            print(f"帖子下标:{x}")
            print(f"文本知识辅助下标{knowledge.iloc[i,0]}")
            print(f"视觉知识辅助下标{picture_knowledge.iloc[x,0]}")
            # print(i,knowledge.iloc[i,0])
            weather_retri = 1
            results = knowledge.iloc[i,1]
            results = ast.literal_eval(results)
            index = x
            query = all_query.iloc[i,1]
            results = source_filter(results)
            results = results[:5]
            query_set.append(all_query.iloc[i,1])
            for result in results:
                if result['title'] in titles_seen: continue
                else: titles_seen.add(result['title'])
                try:
                    all_search_results[query].append(result)
                except:
                    all_search_results[query]=[result]
            i = i + 1 
            #twitter
            # if i == 1520:
            #     break
            #fakeedit
            # if i == 1197:
            #     break
            #MR2-en
            if i == 819:
                break
            # try:
            #     visual_retrieved_text = visual_search(url, text)
            # except Exception as e:
            #     perror(traceback.format_exc())
            #     visual_retrieved_text = ""
        if weather_retri == 1:
            # Topic Relevance Filter
            enhanced_text =f"Title: {title}. \n {text}"
            all_search_results = topic_relevance_filter(enhanced_text, all_search_results, 5, query_set)
            # print(type(all_search_results))
            # print(all_search_results)
            if isinstance(all_search_results,list):
                print("----------------")
                all_search_results={title:[]}
                # print(all_search_results)
            # all_search_results.pop(title,None)
            # Evidence Extraction
            all_search_results[title]=evidence_extraction(all_search_results[title], enhanced_text)

            # Formatting
            retrieved_dict = {f"Infomation might relate to '{title}'":json.dumps(all_search_results[title])}
            all_search_results.pop(title,None)
            for question, evidences in all_search_results.items():
                info_list = [] 
                for evidence in evidences[:2]:
                    # Source: {urlparse(evidence['href']).hostname}
                    info_list.append(f"Title: {evidence['title']}.\n {evidence['body']}")
                # print(f"infolist:--{info_list}")    
                retrieved_dict[f"Infomation might relate to '{question}'"] = info_list 

            retrieved_text = json.dumps(retrieved_dict)
            print(f"retrieved_text:--{retrieved_text}") 
            visual_retrieved_text = picture_knowledge.iloc[x,1]
            # Refined Prediction
            refine_result = refine_prediction_module(TEXT=text,
                                                ORIGINAL_REASONING=direct_explain,
                                                EXTERNAL=retrieved_text,
                                                EXTERNAL_VISUAL=visual_retrieved_text,
                                                DEFINITION=open(definition_path, 'r').read(),
                                                image=url)
            print(f"refine result:{refine_result}")
            # Result Postprocessing
            if refine_result is None: continue
            
            # refined_pred = refine_result["label"]
            # refined_explain =  refine_result
            refined_pred = find_label(refine_result)
            # print(refined_pred)
            refined_explain =  refine_result

            for rumor_type in rumor_types:
                if rumor_type.lower() in refined_pred.lower():
                    refined_pred = rumor_type
                    break  
            if refined_pred == 'true':
                final_pred = 0
            elif refined_pred == 'unverified':
                final_pred = direct_pred   # If model is not sure, go back to direct prediction
            else:
                final_pred = 1
            print('Refined Prediction:', refined_pred)
            final_explain = refined_explain
            # retrieved_text = retrieved_text + visual_retrieved_text

        else:
            final_pred = direct_pred
            final_explain = direct_explain

        # Logging
        labels.append(label)
        direct_labels.append(direct_pred)
        final_preds.append(final_pred)
        logger.append({
            'text': text,
            'image_url': url,
            # 'tool_learning_text': retrieved_text,
            'label': label,
            'prediction': final_pred,
            'explain': final_explain,
            'direct': direct_pred,
            'direct_explain': direct_explain,
        })

        print('\nLabel:', label, ', Refined Prediction:', final_pred, ', Direct:', direct_pred)
        print('Refined Explain:', final_explain)

        save(labels, final_preds, direct_labels, current_index, logger, output_result, output_score)
    else:
        current_index += 1
        print(f"第{current_index}条数据没有图片")
        print('Processing index {}/{}'.format(current_index, total_data_size))
        continue
# driver_quit()