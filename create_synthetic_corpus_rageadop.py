#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 09:03:32 2024

@author: crodrig1
version:0.2
"""
from langchain_community.llms import HuggingFaceEndpoint
import os, jsonlines, json
from huggingface_hub import InferenceClient
import time
endpoint_url = 'https://xxxxxxxxxxxxxxxx.us-east-1.aws.endpoints.huggingface.cloud'
hf_token = 'hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
def connect_llm(endpoint_url, max_new_tokens=4000, top_k=10, top_p=0.95, typical_p=0.95, temperature=0.3, repetition_penalty=1):
    print("Connecting the model...")
    llm = HuggingFaceEndpoint(
    endpoint_url=endpoint_url,
    max_new_tokens=max_new_tokens,
    top_k=top_k,
    top_p=top_p,
    typical_p=typical_p,
    temperature=temperature,
    repetition_penalty=repetition_penalty,
    huggingfacehub_api_token=hf_token)
    return llm
mixtral_llm = connect_llm(endpoint_url, max_new_tokens=4000, top_k=30, top_p=0.95, typical_p=0.95, temperature=0.3, repetition_penalty=1)

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")


source = "normativa_multilingue_complete.jsonl"

prompt = """<s>Your task is to write in {lang} a factoid question and an answer given a context.
Your factoid question should be answerable with a specific, concise piece of factual information from the context.
Your factoid question should be formulated in the same style as questions users could ask in a search engine.
Your factoid question should not be ambiguous, and should refer to a specific fact.
This means that your factoid question MUST NOT mention something like "aquesta norma", "según el contexto" or "d'acord amb el text".
Avoid self-referential or ambiguous questions, such as "Quines lleis són derogades per la nova llei?" or ""Quin és l'objecte de la Llei?"".
You should also provide an exact and extensive quotation of the text from the context on which you based your answer. Write the actual text, not just the reference
The answer should be written in {lang}, and so that it can be understand by anyone, even if it is not an expert in the subject.
The question and the answer should be written in {lang}. Do not write it in English.
Provide your answer in a json format, adding the curly brackets at the start and end of the output, as follows:

Output::: 
"question": (your factoid question), 
"answer": (your answer to the factoid question), 
"quote": (exact quotation)

Now here is the context.

Context: {context}

Output:::</s>"""


def segment(text,tokenizer,n=7000):
    toks = tokenizer.tokenize(text)
    total = len(toks)
    sublists = []
    for i in range(0, total, n):
        sublist = toks[i:i + n]
        sublists.append(sublist)
    return sublists

    

def create_with_hf_endpoint(j, mixtral_llm,tokenizer, prompt):
    qa = {}
    print(j['Número de control'])
    if "TEXT" in j.keys():
        for tokens in segment(j["TEXT"],tokenizer,n=7000):
            context = tokenizer.convert_tokens_to_string(tokens)
            language = "Catalan"
            pr = prompt.format(context=context,lang=language)
            respuesta = mixtral_llm.predict(pr,max_new_tokens=1500,temperature=0.2,repetition_penalty=1)
            respuesta = respuesta.replace("\n","")
            print(respuesta)
            caj=None
            #caj = json.loads(respuesta.strip())      
            try:
                if respuesta.strip().startswith("{"):
                    caj = json.loads(respuesta.strip())
                else:
                    caj = json.loads("{"+respuesta.strip()+"}")
            except Exception as e:
                print(e)
                print(respuesta)
            if caj:
                if 'ca' in qa.keys():
                    listado = qa['ca']
                    listado.append(caj)
                    qa['ca'] = listado
                else:
                    qa['ca'] = [caj]
    if "TEXT_ES" in j.keys():
        for tokens in segment(j["TEXT_ES"],tokenizer,n=7000):
            #toks = tokenizer.tokenize(j["TEXT_ES"])[:7000]
            context = tokenizer.convert_tokens_to_string(tokens)
            language = "Castellano"
            pr = prompt.format(context=context,lang=language)
            respuesta = mixtral_llm.predict(pr,max_new_tokens=1500,temperature=0.2,repetition_penalty=1)
            respuesta = respuesta.replace("\n","")
            print(respuesta)
            esj = None#json.loads(respuesta.strip())
            try:
                if respuesta.strip().startswith("{"):
                    esj = json.loads(respuesta.strip())
                else:
                    esj = json.loads("{"+respuesta.strip()+"}")
            except Exception as e:
                print(e)
                print(respuesta)
            if esj:
                if 'es' in qa.keys():
                    listado = qa['es']
                    listado.append(esj)
                    qa['es'] = listado
                else:
                    qa['es'] = [esj]

    j['qa'] = qa
    return j

#addqa = []

with jsonlines.open("normativa_multilingue_withqavBeyonder.jsonl","w") as writer:
    with jsonlines.open(source) as reader:
        for j in reader:
            jbis = create_with_hf_endpoint(j, mixtral_llm,tokenizer, prompt)
            #addqa.append(jbis)
            writer.write(jbis)
