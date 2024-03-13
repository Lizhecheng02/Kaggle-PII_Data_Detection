# -*- coding: utf-8 -*-
# @Time       : 2024/03/04 15:00
# @Author     : Bian Shengtao
# @File       : label_name_async_reask.py
# @Description: 异步标注人名，并进行二次确认
import pandas as pd
from tqdm import tqdm
import asyncio
import os
import json
import requests
from dotenv import load_dotenv, find_dotenv
import psutil
import aiohttp
import threading
import spacy

# 加载英语模型
nlp = spacy.load("en_core_web_sm")
tqdm.pandas()
load_dotenv(find_dotenv())
data_dir = 'data/raw'
data_path = 'Fake_data_same_nb_1000_0.json'
output_path = f'labeled_{data_path}'
DATA_PATH = os.path.join(data_dir, data_path)
OUTPUT_PATH = os.path.join(data_dir, output_path)


async def gpt_sdk_async(message, model="gpt-4-1106-preview"):
    url = "http://aichat.jd.com/api/openai/multimodalChat/completions"
    payload = json.dumps({
        "messages": [
            {
                "role": "system",
                "content": "\nYou are an expert at extracting PERSON_NAME entities from text. Your job is to extract named entities mentioned in text, and classify them into PERSON_NAME. PERSON_NAME means the entity is a person's name.\n\nNow I want you to label the following text:\n"
            },
            {
                "role": "user",
                "content": f"\Text:\n{message}\n\n\n----------------\nYour job is to extract named entities mentioned in text, nd classify them into PERSON_NAME. PERSON_NAME means the entity is a person's name. You will return the answer in CSV format, with two columns seperated by the % character. First column is the extracted entity and second column is the category. Rows in the CSV are separated by new line character.你只需要标注人名即可\n----------------\nOutput: "
            }
        ],
        "stream": False,
        "model": model,
        "temperature": 0.5,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "max_tokens": 2000
    })
    headers = {
        'Cookie': '__jdu=1679553177141853513866; shshshfpa=6fd1aac0-14ba-b064-df6d-4b275b280d7d-1679553203; shshshfpx=6fd1aac0-14ba-b064-df6d-4b275b280d7d-1679553203; shshshfp=e79f3daecd7b403296d8e631262029b5; unpl=JF8EAJ5nNSttXRxUDU8CE0UVTVgBW1hYSR4LaWdXAVleSFcNEwIdRRJ7XlVdWBRKFB9uYhRUW1NLVg4bCysSEHtdVV9eDkIQAmthNWRVUCVXSBtsGHwQBhAZbl4IexcCX2cCUlVcT1YFHQQbFxBCWVZaXQpCEARfZjVUW2h7ZAQrAysTIAAzVRNdD00fB2tlBVJbWE5UDB8AHxISQlpTblw4SA; cn=25; jd.erp.lang=zh_CN; jdd69fo72b8lfeoe=J4ASAANBZVJIOCXQPKBIUS35QQP2AHC52GKS6J6NO3DKZA2BRZ6AIYQU2CJYOZDBMM356O6BE3M2VMF2YMCHHXDKKU; ipLoc-djd=1-2809-51231-0; wlfstk_smdl=zfbn0iycj1kgpyjaqihyf6ttr8gdy4zx; qid_uid=e4db96a4-dba9-43f1-abab-fd8c1fdffd3a; qid_fs=1708505632737; qid_ls=1708505632737; qid_ts=1708505632740; qid_vis=1; new_isp_address=15_1213_1214_52674; 3AB9D23F7A4B3CSS=jdd03EQOCVE4H36QOWQGBYCJU5X3S6ZDWMT6UUFJQKZEMAAVF4W5WJGP4J2ZQEF5Z5Z5AYUDVARV34YAQQU6P6HPT2ZTPOEAAAAMNZPE77QAAAAAACLKTLQBMRGO6VYX; shshshfpb=BApXexJbCyOhAJwTQ5v1e0G2kIlDg7rynBzsUNSxE9xJ1MqTte4O2; joyya=1708521029.1708521034.38.1j3nbnz; mba_muid=1679553177141853513866; __jdv=137720036|direct|-|none|-|1708930177550; JSESSIONID=4C394A9BC07B59EA296BB14C6CD97524.s1; token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjNmMmdYeklJTjlKZWxVc0ZnT1JlIiwiZXhwIjoxNzA5ODYwNTA1LCJpYXQiOjE3MDkyNTU3MDV9.rnrVE-yJNzIR-hgau5hWfYmAl_PSj32dGFRo2idKi5I; 3AB9D23F7A4B3C9B=EQOCVE4H36QOWQGBYCJU5X3S6ZDWMT6UUFJQKZEMAAVF4W5WJGP4J2ZQEF5Z5Z5AYUDVARV34YAQQU6P6HPT2ZTPOE; focus-token=ee.090292f724f37461417ccf4232c1771c; focus-team-id=00046419; focus-client=WEB; __jdc=137579179; __jda=137579179.1679553177141853513866.1679553177.1709535014.1709541275.482; sso.jd.com=BJ.DDB27AAED07B716F10FEDD630E7D9A2A.9220240304174342; ssa.global.ticket=B08B54C6DADACDFA75BD37E915921BB0',
        'Content-Type': 'application/json'
    }

    async with aiohttp.ClientSession() as session:
        for _ in range(5):
            async with session.post(url, headers=headers, data=payload) as response:
                if response.status == 200:
                    text = await response.text()
                    res = json.loads(text)
                    try:
                        return res['choices'][0]['message']['content']
                    except KeyError:
                        if _ != 4:
                            print(f"Request failed, retrying... {res}")
                            continue
                        else:
                            return res
                else:
                    print(f"Request failed with status {response.status}, retrying...")
    return None


async def reask_gpt_sdk_async(message, entity, model="gpt-4-1106-preview"):
    api_key = os.environ["OPENAI_API_KEY"]
    url = os.environ["OPENAI_API_BASE"]
    payload = json.dumps({
        "messages": [
            {
                "role": "system",
                "content": "我会给你提供一篇英文文章，以及一个英文，你需要告诉我在这篇文章中，这个英文是不是人的名字，如果是则返回True，否则返回False。"
            },
            {
                "role": "user",
                "content": f"\n文章:\n{message}\n----------------\n英文单词:{entity}\n----------------\nOutput: "
            }
        ],
        "stream": False,
        "model": model,
        "temperature": 0,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "max_tokens": 2000
    })
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }

    async with aiohttp.ClientSession() as session:
        for _ in range(5):
            async with session.post(url, headers=headers, data=payload) as response:
                if response.status == 200:
                    text = await response.text()
                    res = json.loads(text)
                    try:
                        return res['choices'][0]['message']['content']
                    except KeyError:
                        if _ != 4:
                            print(f"Request failed, retrying... {res}")
                            continue
                        else:
                            return res
                else:
                    print(f"Request failed with status {response.status}, retrying...")
    return None


async def func(row):
    text = row['full_text_no_email']
    token_list = row['tokens']
    label_list = ["O"] * len(token_list)
    model_list = ['gpt-4-1106-preview']
    response_list = [gpt_sdk_async(text, model) for model in model_list]
    response_list = await asyncio.gather(*response_list)
    if all(isinstance(response, dict) and response.get('error') for response in response_list):
        return 'failed'
    else:
        # 删除所有的error
        response_list = [response for response in response_list if not (isinstance(response, dict) and response.get('error'))]
    # print(f'response_list: {response_list}')

    entity_list_org = []
    for response in response_list:
        # response转换为dataframe
        res = response.split('\n')
        res = [i.split('%') for i in res]
        res = [i for i in res if len(i) == 2]
        res = pd.DataFrame(res, columns=['entity', 'category'])
        # 如果entity是是Mr. Mrs. Dr. Miss.等称开头的，去掉，只保留名字
        tmp_prefix_list = [
            'Mr.', 'Ms.', 'Mrs.', 'Dr.', 'Miss.', 
            'Mr', 'Ms', 'Mrs',
            'Dr', 'Miss', 'mr.', 'ms.', 'mrs.', 
            'dr.', 'miss.', 'mr', 'mrs', 'dr', 'miss'
        ]
        res['entity'] = res['entity'].apply(lambda x: ' '.join(x.split(' ')[1:]) if x.split(' ')[0] in tmp_prefix_list else x)
        # 去重
        res.drop_duplicates(subset=['entity'], keep='first', inplace=True)
        entity_list_org.append(res['entity'].tolist())

    entity_list_all = list(set([j for i in entity_list_org for j in i]))
    entity_list_reask = []
    for entity in entity_list_all:
        response = await reask_gpt_sdk_async(text, entity)
        if isinstance(response, dict) and response.get('error'):
            return 'failed'
        else:
            if response == 'True':
                entity_list_reask.append(entity)
    entity_list = list(set(entity_list_reask))
    # 对每个entity_list去token_list中进行查找，找到对应的位置，如果有多个位置，每个位置都要返回

    def find_entity_position(entity):
        print(f'entity: {entity}')
        entity_list = [token.text for token in nlp(entity)]
        # print('token_list:', token_list)
        # print(f'entity_list: {entity_list}')
        position_list = []
        for idx, token in enumerate(token_list):
            entity_position_list = []
            if entity_list == token_list[idx:idx+len(entity_list)]:
                for i, entity in enumerate(entity_list):
                    if i == 0:
                        entity_position_list.append((idx + i, entity, 'B-NAME'))
                    else:
                        entity_position_list.append((idx + i, entity, 'I-NAME'))
                position_list.append(entity_position_list)
        return position_list
    position = [find_entity_position(entity) for entity in entity_list]
    res = pd.DataFrame({'entity': entity_list, 'position': position})
    # print(res)

    # 将position转换为标签
    for i, r in res.iterrows():
        for position in r['position']:
            for p in position:
                label_list[p[0]] = p[2]
    # print(f'label_list: {label_list}')
    return label_list


async def merge_text_with_no_email(row):
    text = ''
    for token, ws, label in zip(row['tokens'], row['trailing_whitespace'], row['labels']):
        if label == 'B-EMAIL' or label == 'I-EMAIL':
            text += 'EMAIL'
        else:
            text += token
        if ws:
            text += ' '
    return text


async def main(data_path):
    data_dir = 'data/raw'
    output_path = f'labeled_{data_path}'
    DATA_PATH = os.path.join(data_dir, data_path)
    OUTPUT_PATH = os.path.join(data_dir, output_path)
    semaphore = asyncio.Semaphore(30)
    df = pd.read_json(DATA_PATH)

    async def process_row_with_semaphore(row, func):
        async with semaphore:
            return await func(row)

    full_text_no_email_tasks = [process_row_with_semaphore(row, merge_text_with_no_email) for idx, row in df.iterrows()]
    df['full_text_no_email'] = await asyncio.gather(*full_text_no_email_tasks)

    label_name_tasks_tasks = [process_row_with_semaphore(row, func) for idx, row in df.iterrows()]
    res = await asyncio.gather(*label_name_tasks_tasks)
    df['label_name'] = res
    df.to_json(OUTPUT_PATH, orient='records')


asyncio.run(main('Fake_data_same_nb_1000_0.json'))
asyncio.run(main('nb_mixtral-8x7b-v1_no-email.json'))
