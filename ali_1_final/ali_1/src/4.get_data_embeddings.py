import csv
import sys
import os
import torch
from tqdm import tqdm
import numpy as np
import argparse
 
sys.path.append("..")
from model_cl import BertForCL
from transformers import AutoTokenizer

# 运行前检查使用的Pooler是否一致，检查tokenizer是否一致
device = "cuda:4"
path = os.getcwd()
tokenizer = AutoTokenizer.from_pretrained("../model/")
batch_size = 128
use_pinyin = False

model_path = "../result/baseline/checkpoint-1400/"

def encode_fun(texts, model):
    inputs = tokenizer.batch_encode_plus(texts, padding=True, truncation=True, return_tensors="pt", max_length=115)
    inputs.to(device)
    with torch.no_grad():
        embeddings = model(**inputs, sent_emb=True).pooler_output
        embeddings = embeddings.squeeze(0).cpu().numpy()
    return embeddings

def construct_ensemble_input():
    pass

def get_ensemble_embedding():
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=None)
    args = parser.parse_args()

    model_path = args.model_path
    if model_path is None:
       print("Error: model path is none !")

    model = BertForCL.from_pretrained(model_path)
    model.to(device)

    corpus = [line[1] for line in csv.reader(open("../data/processed_data/corpus.tsv"), delimiter='\t')]
    query = [line[1] for line in csv.reader(open("../data/processed_data/dev.query.txt"), delimiter='\t')]


    # 初赛提交：query_embedding
    query_embedding_file = csv.writer(open(model_path + 'query_embedding', 'w'), delimiter='\t')

    for i in tqdm(range(0, len(query), batch_size)):
        batch_text = query[i:i + batch_size]
        temp_embedding = encode_fun(batch_text, model)
        for j in range(len(temp_embedding)):
            writer_str = temp_embedding[j].tolist()
            writer_str = [format(s, '.8f') for s in writer_str]
            writer_str = ','.join(writer_str)
            query_embedding_file.writerow([i + j + 200001, writer_str])

    # 初赛提交：doc_embedding
    # doc_embedding_file = csv.writer(open(model_path + 'doc_embedding', 'w'), delimiter='\t')
    # for i in tqdm(range(0, len(corpus), batch_size)):
    #     batch_text = corpus[i:i + batch_size]
    #     temp_embedding = encode_fun(batch_text, model)
    #     for j in range(len(temp_embedding)):
    #         writer_str = temp_embedding[j].tolist()
    #         writer_str = [format(s, '.8f') for s in writer_str]
    #         writer_str = ','.join(writer_str)
    #         doc_embedding_file.writerow([i + j + 1, writer_str])

    train_data_path = "../data/processed_data/train.query.txt"
    train_dev_query_data = [line[1] for line in csv.reader(open(train_data_path), delimiter='\t')]
    
    # 复赛 train_query.npy  
    train_query_data = train_dev_query_data[:-5000]
    train_query_list = []
    for i in tqdm(range(0, len(train_query_data), batch_size)):
        batch_text = train_query_data[i:i + batch_size]
        temp_embedding = encode_fun(batch_text, model)
        train_query_list.append(temp_embedding)
    train_query_np = np.array(train_query_list)
    train_query_np = np.concatenate(train_query_np, axis=0)  
    train_query_embedding_save_path = model_path + 'train_query.npy'
    np.save(train_query_embedding_save_path, train_query_np)
    
    # 复赛 dev_query.npy
    dev_query_data = train_dev_query_data[-5000:]
    dev_query_list = []
    for i in tqdm(range(0, len(dev_query_data), batch_size)):
        batch_text = dev_query_data[i:i + batch_size]
        temp_embedding = encode_fun(batch_text, model)
        dev_query_list.append(temp_embedding)
    dev_query_np = np.array(dev_query_list)
    dev_query_np = np.concatenate(dev_query_np, axis=0)
    dev_query_embedding_save_path = model_path + 'dev_query.npy'
    np.save(dev_query_embedding_save_path, dev_query_np)

