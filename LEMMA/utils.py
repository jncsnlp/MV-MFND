
import io
import sys
import json
import base64
import requests
# from openai import OpenAI
from langdetect import detect
from configs import out_root, prompts_root, cache_root, imgbed_root, OPENAI_KEY
from email.mime import image
from typing import final
from sympy import im
# from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
# from qwen_vl_utils import process_vision_info
import json
import os
import gc
import torch
import string
import requests
import csv
import re
import pandas as pd 
import ast
# client = OpenAI()
# Qwen
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     "/home/jncsnlp4/SSD2/model/qwen2-vl-intruct", torch_dtype="auto", device_map="auto"
# )
# processor = AutoProcessor.from_pretrained("/home/jncsnlp4/SSD2/model/qwen2-vl-intruct")

#llava
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

def remove_trailing_comma(json_str):

    json_str = json_str.strip()

    if json_str.startswith('[') and json_str.endswith(']'):
        json_str = json_str[1:-1].strip()

    json_str = re.sub(r',\s*([\}\]])', r'\1', json_str)

    return json_str

def chat_response(image_path,prompt,types):
    # if types == "text":
    #     messages = [
    #         {
    #             "role": "user",
    #             "content": [
    #                 {"type": "text", "text": prompt},
    #             ],
    #         }
    #     ]
    #     text = processor.apply_chat_template(
    #         messages, tokenize=False, add_generation_prompt=True
    #     )
    #     inputs = processor(
    #         text=[text],
    #         padding=True,
    #         return_tensors="pt",
    #     )
    # else:
    #     messages = [
    #         {
    #         "role": "user",
    #         "content": [
    #             {
    #                 "type": "image",
    #                 "image": image_path,
    #             },
    #             {"type": "text", "text": prompt},
    #             ],
    #         }
    #     ]
    #     text = processor.apply_chat_template(
    #         messages, tokenize=False, add_generation_prompt=True
    #     )
    #     image_inputs, video_inputs = process_vision_info(messages)
    #     inputs = processor(
    #         text=[text],
    #         images=image_inputs,
    #         videos=video_inputs,
    #         padding=True,
    #         return_tensors="pt",
    #     )

    # inputs = inputs.to("cuda")
    #     # Inference: Generation of the output
    
    # generated = model.generate(**inputs, max_new_tokens=512,output_logits = True,return_dict_in_generate=True,temperature=0.5,top_p=0.75,top_k=2)
    # logits = generated.logits
    # probs = [torch.softmax(log, dim=-1) for log in logits]
    #     # print(probs)
    # generated_ids = generated.sequences
    # generated_ids_trimmed = [
    #     out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    # ]
    # output_text = processor.batch_decode(
    #     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    # )
    # logit_pro = 0
    # for i, token_id in enumerate(generated_ids[0][len(inputs.input_ids[0]):]):
    #     token_prob = probs[i][0, token_id].item()
    #     word = processor.decode(token_id)
    #     if token_id == 7951 or token_id == 30570:
    #         # print(f"Token ID: {token_id}, word:{word},Probability: {token_prob}")
    #         logit_pro = token_prob

    # return output_text[0]    

    #llava
    try:
        if types == "mm":
            args = type('Args', (), {
                    "model_path": model_path,
                    "model_base": None,
                    "model_name": get_model_name_from_path(model_path),
                    "query": prompt,
                    "conv_mode": None,
                    "image_file": image_path,
                    "sep": ",",
                    "temperature": 0.5,
                    "top_p": 0.75,
                    "num_beams": 1,
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
                    "top_p": 0.75,
                    "num_beams": 1,
                    "max_new_tokens": 512 #512
                })()
        model_name = get_model_name_from_path(args.model_path)
        response,real_prob,fake_prob=eval_model(args,model_name,tokenizer, model, image_processor)
        # print("--------------------------------")
        # print("response",response)   
        
        return response
    except Exception as e:
        print(f"An error occurred: {e}")
        print('————————————prompt again————————————')
        return "wrong",0,0


def perror(str):
    print("\033[91m"+str+"\033[0m")

def pwarn(str):
    print("\033[33m"+str+"\033[0m")
    
def process_multilines_output(x):
    lines=x.split("\n")
    label=lines[-1].strip().lower()
    explanation="\n".join(lines[:-1]) if len(lines)>1 else ""
    return {"label":label,"explanation":explanation}


# def onlineImg_process(prompt, url, model="gpt-4-vision-preview", max_tokens=1000, temperature=0.1):
#     response = client.chat.completions.create(
#         model=model,
#         messages=[
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "text", "text": prompt},
#                     {
#                         "type": "image_url",
#                         "image_url": {
#                             "url": f"{url}",
#                         },
#                     },
#                 ],
#             }
#         ],
#         max_tokens=max_tokens,
#         temperature=temperature
#     )
#     return response.choices[0].message.content

def topic_relevance_filter(text, all_results, top_k, query_set, cutoff_index=150):
    # Structure flattern
    all_results_flatterned = []
    for qid, query in enumerate(all_results.keys()):
        results= all_results[query]
        for i,result in enumerate(results):
            id = qid*top_k + i        # we can use id//top_k to determine which query it belongs later
            result['body']=result['body'][:cutoff_index]
            all_results_flatterned.append({id:result})

    # print(all_results_flatterned)
    # Prompt formation
    text=text[:cutoff_index+50]
    # print(f"{text}-----")
    # print(json.dumps(all_results_flatterned, ensure_ascii=False, indent=4))
    topic_relevance_prompt = ("Your task is to filter the off-topic search result. You will be provided a piece of text.\
You have to determine the topic of the text. Then, you will be provided the search result in JSON format. For each entry,\
there is a unique integer key serving as the id of each entry. The value of each entry consists of three attributes: title,\
body, url. And  You have to filter the off-topic search result according to the content of the title and body. For each entry\
in the list, output a binary label (\"true\" means that the content is relevant to the topic of text, \"false\" means irrelevant). Put all the labels \
in a JSON dict. You must output a binary label for each entry in the list.\n"

"Example output format:\n"

"{{\"0\":true, \"1\":false, \"2\":false, \"3\":true, \"4\":false, \"5\":true, \"6\":false, \"7\":true}}\n"

"Text input that you are going to determine the topic:\n"

"{}\n"

"Search result in JSON format:\n"

"{}\n"

"Your answer (don't include the Markdown syntax like ```json. just directly outputs JSON list object in the format of the Example output format. Don't output anything else):\n"
).format(text, json.dumps(all_results_flatterned, ensure_ascii=False, indent=4))
    response = chat_response("111",topic_relevance_prompt,types="text")
    # Post process
    print(response)
    response = remove_trailing_comma(response)
    print(response)
    try:
        relevance_labels=json.loads(response)
        # print(relevance_labels)
        # print(1)
    except:
        pwarn("Tool learning Warning: Invalid response from topic_relevance_filter. Remain unchanged.")
        return results
    
    # Wash the string keys to int, and remove the non-integer keys
    temp = {}
    for id, value in relevance_labels.items():
        if type(id)==str: 
            if id.isdigit(): 
                id=int(id)
            else: continue
        temp[id] = value
    relevance_labels = temp
        
    # Filter the results and restructure the results
    all_filtered_results = {}  
    for query in query_set:
        all_filtered_results[query]=[]
    for temp in all_results_flatterned:
        id, result = list(temp.items())[0]
        if id in relevance_labels and relevance_labels[id]==True:
            qid=id//top_k               # use id//top_k to determine which query it belongs
            query=query_set[qid]
            all_filtered_results[query].append(result)
    return all_filtered_results

def pwarn(str):
    print("\033[33m"+str+"\033[0m")

def evidence_extraction(search_results, query, pre_max_len=2000, after_max_len=250, max_items = 3):
    documents = {}
    headers = {}
    for id, search_result in enumerate(search_results):
        title = search_result['title']
        link = search_result['href']
        body = search_result["body"]
        # full_text = scraper(link,pre_max_len)
        # if len(full_text)<30:
        documents[str(id)] = body
        # else:  
        #     documents[str(id)] = full_text
        #  Source: {urlparse(link).hostname}
        headers[str(id)] = f"Title: {title}."

    # Prompt formation
    evidence_extraction_prompt = ("You are given a Query. You are then given a dictionary called Documents, whose key is the document ID and value is the documen retrieved from the Internet. For each document," 
"- if some segments are relevant to any key information in Query, quote them."
"- if the whole page is relevant to Query, summarize it comprehensively and concisely"
"- if it is irrelevant to Query, return empty string"
"Please output a new dictionary, whose key is still document ID and value is the document segments relevant to the Query. Try to only include the relevant part instead of returning the whole thing back.But do not be too strict."

"### Example output format"
"{{\"0\":\"Funding has been awarded to nine pioneering projects to help Scottish remanufacturing businesses make the most efficient\
use of material. The Scottish\", \"1\":\"New Institute of Remanufacture to drive Scotland's circular economy\",\"2\": \"'The Scottish\
Government defines a circular economy as a system in which “resources are kept in use for as long as possible” – in other words, recycling.\"\
,\"3\":\"Our circular economy strategy to build a strong economy, protect our resources and support the environment.\"}}"

"### Your turn"

"**Query**"
"{}"

"**Documents**"
"{}"

"**Output: (Don't output anything else except for the JSON object. Don't add Markdown syntax like ```json):**"
).format(query,json.dumps(documents))
    response = chat_response("111",evidence_extraction_prompt,types="text")
    evidences = []
    try:
        extracted_results=json.loads(response)

        for id, extracted_result in extracted_results.items():
            if extracted_result != "":
                evidence = headers[str(id)] + extracted_result 
                evidences.append(evidence)
    except:
        pwarn("Tool learning Warning: Invalid response from evidence_extraction. Remain unchanged.")
        evidences = list(documents.values())
    evidences=[evidence[:after_max_len] for evidence in evidences if len(evidence)>0]
    return evidences[:max_items]

def offlineImg_process(prompt, image_path, model_name="Qwen-2-vl", max_tokens=1000, temperature=0.1):
    # Encode function
    # def encode_image(image_path):
    #     with open(image_path, "rb") as image_file:
    #         return base64.b64encode(image_file.read()).decode('utf-8')

    # # Getting the base64 string
    # base64_image = encode_image(image_path)

    # headers = {
    #     "Content-Type": "application/json",
    #     "Authorization": f"Bearer {OPENAI_KEY}"
    # }
    #llava
    # response = chat_response(image_path,prompt,types="mm")
    # return response
    #QWen
    response = chat_response("111",prompt,types="text")
    return response


    # payload = {
    #     "model": model,
    #     "messages": [
    #         {
    #             "role": "user",
    #             "content": [
    #                 {
    #                     "type": "text",
    #                     "text": prompt
    #                 },
    #                 {
    #                     "type": "image_url",
    #                     "image_url": {
    #                         "url": f"data:image/jpeg;base64,{base64_image}"
    #                     }
    #                 }
    #             ]
    #         }
    #     ],
    #     "max_tokens": max_tokens,
    #     "temperature": temperature
    # }

    # response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    # return eval(response.text)["choices"][0]["message"]["content"]


def gpt_no_image(prompt, model_name="Qwen-2-t", max_tokens=1000, temperature=0.1):
    
    # response = client.chat.completions.create(
    #     model=model,
    #     messages=[
    #         {"role": "user", "content": prompt}
    #     ],
    #     max_tokens=max_tokens,
    #     temperature=temperature
    # )
    # return response.choices[0].message.content
    #llava
    # response = chat_response("111",prompt,types="text")
    # return response
    #Qwen
    response = chat_response("111",prompt,types="text")
    return response
    
    

# def image_caption(source, is_url=True):

#     with open(prompts_root + "img_caption.md", "r") as f:
#         image_caption_prompt = f.read()
        
#     with open(cache_root + "img_caption.json","r") as f:
#         image_caption_cache = json.loads(f.read())

#     if source=="" or source is None:
#         return ""
#     elif source in image_caption_cache:
#         return image_caption_cache[source]
#     else:
#         # Get the image caption
#         try:
#             if is_url:
#                 image_path=source
#                 if "http" not in source:
#                     image_path = imgbed_root + source
#                 caption= onlineImg_process(image_caption_prompt, image_path, max_tokens=1000)
#             else:
#                 caption= offlineImg_process(image_caption_prompt, image_path, max_tokens=1000)
#             image_caption_cache[source]=caption
#         except:
#             return ""
#         with open(cache_root+"img_caption.json","w") as f:
#             f.write(json.dumps(image_caption_cache))
#         return caption
    
        # prompt=prompt.format(CAPTION)
def metric(labels, pred_labels):
    def confusion_matrix(truth, pred):
        tp = sum((l == 1 and p == 1) for l, p in zip(truth, pred))
        fp = sum((l == 0 and p == 1) for l, p in zip(truth, pred))
        fn = sum((l == 1 and p == 0) for l, p in zip(truth, pred))
        tn = sum((l == 0 and p == 0) for l, p in zip(truth, pred))

        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        return tp, fp, fn, tn, precision, recall, f1

    accuracy = sum((l == p) for l, p in zip(labels, pred_labels)) / len(labels)

    rumor_labels = labels
    rumor_pred_labels = pred_labels

    non_rumor_labels = [1 - l for l in labels]
    non_rumor_pred_labels = [1 - p for p in pred_labels]

    rumor_metrics = confusion_matrix(rumor_labels, rumor_pred_labels)
    non_rumor_metrics = confusion_matrix(non_rumor_labels, non_rumor_pred_labels)

    return {
        'labels': labels,
        'predictions': pred_labels,
        'accuracy': accuracy,
        'rumor': {
            'true_positives': rumor_metrics[0],
            'false_positives': rumor_metrics[1],
            'false_negatives': rumor_metrics[2],
            'true_negatives': rumor_metrics[3],
            'precision': rumor_metrics[4],
            'recall': rumor_metrics[5],
            'f1': rumor_metrics[6]
        },
        'non_rumor': {
            'true_positives': non_rumor_metrics[0],
            'false_positives': non_rumor_metrics[1],
            'false_negatives': non_rumor_metrics[2],
            'true_negatives': non_rumor_metrics[3],
            'precision': non_rumor_metrics[4],
            'recall': non_rumor_metrics[5],
            'f1': non_rumor_metrics[6]
        }
    }


def write_metric_result(file_name, data, mode='w', prefix=''):
    with open(file_name, mode, encoding='utf-8') as f:
        if prefix:
            f.write('{}\n'.format(prefix))
        f.write('Labels:\n{}\nPredictions:\n{}\n\n'.format(data['labels'], data['predictions']))

        f.write('Accuracy: {}\n\n'.format(data['accuracy']))

        f.write('Rumor Section:\n')
        f.write('True positives: {}\n'.format(data['rumor']['true_positives']))
        f.write('False positives: {}\n'.format(data['rumor']['false_positives']))
        f.write('False negatives: {}\n'.format(data['rumor']['false_negatives']))
        f.write('True negatives: {}\n'.format(data['rumor']['true_negatives']))
        f.write('Precision: {}\n'.format(data['rumor']['precision']))
        f.write('Recall: {}\n'.format(data['rumor']['recall']))
        f.write('F1 Score: {}\n\n'.format(data['rumor']['f1']))

        f.write('Non-rumor Section:\n')
        f.write('True positives: {}\n'.format(data['non_rumor']['true_positives']))
        f.write('False positives: {}\n'.format(data['non_rumor']['false_positives']))
        f.write('False negatives: {}\n'.format(data['non_rumor']['false_negatives']))
        f.write('True negatives: {}\n'.format(data['non_rumor']['true_negatives']))
        f.write('Precision: {}\n'.format(data['non_rumor']['precision']))
        f.write('Recall: {}\n'.format(data['non_rumor']['recall']))
        f.write('F1 Score: {}\n\n'.format(data['non_rumor']['f1']))


def stats(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    num_items = len(data)
    labels = []
    predictions = []
    zero_shot_predictions = []
    total_correct = 0
    total_incorrect = 0
    zero_shot_correct = 0
    zero_shot_incorrect = 0
    total_modified = 0
    total_modified_0_to_1 = 0
    total_modified_0_to_1_correct = 0
    total_modified_0_to_1_incorrect = 0
    total_modified_1_to_0 = 0
    total_modified_1_to_0_correct = 0
    total_modified_1_to_0_incorrect = 0
    total_unmodified = 0
    total_modified_correct = 0
    total_modified_incorrect = 0

    for item in data:
        labels.append(item['label'])
        predictions.append(item['prediction'])
        zero_shot_predictions.append(item['direct'])
        if item['label'] == item['direct']:
            if item['prediction'] != item['label']:
                total_incorrect += 1
                zero_shot_correct += 1
                total_modified += 1
                if item['direct'] == 0:
                    total_modified_0_to_1 += 1
                    total_modified_0_to_1_incorrect += 1
                else:
                    total_modified_1_to_0 += 1
                    total_modified_1_to_0_incorrect += 1
                total_modified_incorrect += 1
            else:
                total_correct += 1
                zero_shot_correct += 1
                total_unmodified += 1
        else:
            if item['prediction'] == item['label']:
                total_correct += 1
                zero_shot_incorrect += 1
                total_modified += 1
                if item['direct'] == 0:
                    total_modified_0_to_1 += 1
                    total_modified_0_to_1_correct += 1
                else:
                    total_modified_1_to_0 += 1
                    total_modified_1_to_0_correct += 1
                total_modified_correct += 1
            else:
                total_incorrect += 1
                zero_shot_incorrect += 1
                total_unmodified += 1

    print('Total items: {}'.format(num_items))
    print('Total correct: {}'.format(total_correct))
    print('Total incorrect: {}'.format(total_incorrect))
    print('Total Accuracy: {}'.format(total_correct / num_items))
    print('Zero-shot correct: {}'.format(zero_shot_correct))
    print('Zero-shot incorrect: {}'.format(zero_shot_incorrect))
    print('Zero-shot Accuracy: {}'.format(zero_shot_correct / num_items))
    print(
        'Total modified: {}\n\t| 0 -> 1: {}\n\t\t| Correct: {}\n\t\t| Incorrect : {}\n\t| 1-> 0: {}\n\t\t| Correct: {}\n\t\t| Incorrect : {}'.format(
            total_modified, total_modified_0_to_1, total_modified_0_to_1_correct, total_modified_0_to_1_incorrect,
            total_modified_1_to_0, total_modified_1_to_0_correct, total_modified_1_to_0_incorrect))
    print('Total unmodified: {}'.format(total_unmodified))
    print('Total modified correct: {}'.format(total_modified_correct))
    print('Total modified incorrect: {}'.format(total_modified_incorrect))


def stats_str(path):
    sio = io.StringIO()
    sys.stdout = sio
    stats(path)
    sys.stdout = sys.__stdout__
    sio.seek(0)
    return sio.read()


def predict_region(s):
    lang = detect(s)
    region_map = {
        'en': 'us-en',
        # 'ca': 'ct-ca',
        'zh-cn': 'tw-tzh',
        'zh-tw': 'tw-tzh',
        # 'fr': 'fr-fr',
        # 'tr': 'tr-tr',
        # 'nl': 'nl-nl',
    }
    if lang in region_map:
        return region_map[lang]
    else:
        return 'us-en'


def save(labels, pred_labels, zero_shot_labels, current_index, all_results, output_result, output_score):
    with open(output_result, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)
    with open(output_score, 'w', encoding='utf-8') as f:
        f.write('Labels:\n{}\nZero-shot:\n{}\nPredictions:\n{}\nCurrent Index:{}\n'.format(labels, zero_shot_labels,
                                                                                           pred_labels, current_index))
        f.write(stats_str(output_result))

    evaluation_result = metric(labels, pred_labels)
    write_metric_result(output_score, evaluation_result, 'a', prefix='lemma section')

    evaluation_result = metric(labels, zero_shot_labels)
    write_metric_result(output_score, evaluation_result, 'a', prefix='zero shot section')


def save_baseline(labels, pred_labels, current_index, all_results, output_result, output_score):
    with open(output_result, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)
    with open(output_score, 'w', encoding='utf-8') as f:
        f.write('Labels:\n{}\nPredictions:\n{}\nCurrent Index:{}\n'.format(labels, pred_labels, current_index))

    evaluation_result = metric(labels, pred_labels)
    write_metric_result(output_score, evaluation_result, 'a', prefix='lemma section')

    evaluation_result = metric(labels, pred_labels)
    write_metric_result(output_score, evaluation_result, 'a', prefix='zero shot section')
    
if __name__ == '__main__':
    path = 'out/lemma_twitter_output.json'
    with open(path, 'r', encoding = "utf-8") as f:
        data = json.load(f)

    labels = [items['label'] for items in data]
    preds = [items['prediction'] for items in data]

    print(metric(labels, preds))
    print(stats(path))

