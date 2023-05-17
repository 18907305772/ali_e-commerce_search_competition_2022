import csv
from multiprocessing import context
import random
from random import shuffle, choice
from torch import neg_
from tqdm import tqdm
import pandas as pd

random.seed(2022)

def to_dict(data_path):
    data_dict = {}
    with open(data_path) as f:
        for line in f.readlines():
            id = int(line.split('\t')[0])
            text = line.split('\t')[1]
            data_dict[id] = text
    return data_dict

def construct_negtive_sample(qrels_path, cpr_qrels_path, corpus_dict, query_dict, cpr_corpus_dict, cpr_query_dict, train_data_path, validate_data_path, validate_num=1000):
    # 首先，筛选出负样本id
    neg_ids = set(corpus_dict.keys())
    all_query_doc = []
    cpr_query_doc = []
    with open(qrels_path) as f:
        for line in f.readlines():
            query_id = int(line.split('\t')[0])
            doc_id = int(line.split('\t')[1].rstrip('\n'))
            all_query_doc.append([query_id, doc_id])
            # 正样本doc id 不作为负样本id
            neg_ids.remove(doc_id)
    
    # 加入cpr数据
    with open(cpr_qrels_path) as f:
        for line in f.readlines():
            query_id = int(line.split('\t')[0])
            doc_id = int(line.split('\t')[2].rstrip('\n'))
            cpr_query_doc.append([query_id, doc_id])
    
    neg_ids = list(neg_ids)

    shuffle(all_query_doc)
    shuffle(cpr_query_doc)

    # 构造带负样本的训练集
    train_query_doc = all_query_doc[:-validate_num]
    train_data = []
    with open(train_data_path, 'w') as f:
        train_data.append('query,doc,doc_neg\n')
        for line in train_query_doc:
            query_text = query_dict[line[0]].rstrip('\n')
            doc_text = corpus_dict[line[1]].rstrip('\n')
            # 从ne_ids中随机选择负样本
            negtive_text = corpus_dict[choice(neg_ids)].rstrip('\n')
            # 长度截断
            if len(negtive_text) > 64:
                negtive_text = negtive_text[:64]
            
            # 添加引号处理文本中自带逗号的问题
            if query_text.find(',') != -1:
                query_text = "\"" + query_text + "\""
            if doc_text.find(',') != -1:
                doc_text = "\"" + doc_text + "\""
            if negtive_text.find(',') != -1:
                negtive_text = "\"" + negtive_text + "\""

            train_data.append(query_text + ',' + doc_text + ',' + negtive_text + '\n')
        # cpr数据
        for line in cpr_query_doc:
            query_text = cpr_query_dict[line[0]].rstrip('\n')
            doc_text = cpr_corpus_dict[line[1]].rstrip('\n')
            # 从ne_ids中随机选择负样本
            negtive_text = corpus_dict[choice(neg_ids)].rstrip('\n')
            # 长度截断
            if len(negtive_text) > 64:
                negtive_text = negtive_text[:64]
            
            # 添加引号处理文本中自带逗号的问题
            if query_text.find(',') != -1:
                query_text = "\"" + query_text + "\""
            if doc_text.find(',') != -1:
                doc_text = "\"" + doc_text + "\""
            if negtive_text.find(',') != -1:
                negtive_text = "\"" + negtive_text + "\""

            train_data.append(query_text + ',' + doc_text + ',' + negtive_text + '\n')
        f.writelines(train_data)

    # 构造带负样本的验证集
    validate_query_doc = all_query_doc[-validate_num:]
    validate_data = []
    with open(validate_data_path, 'w') as f:
        validate_data.append('query,doc,doc_neg\n')
        for line in validate_query_doc:
            query_text = query_dict[line[0]].rstrip('\n')
            doc_text = corpus_dict[line[1]].rstrip('\n')
            # 从ne_ids中随机选择负样本
            negtive_text = corpus_dict[choice(neg_ids)].rstrip('\n')
            # 长度截断
            if len(negtive_text) > 64:
                negtive_text = negtive_text[:64]
                
            # 添加引号处理文本中自带逗号的问题
            if query_text.find(',') != -1:
                query_text = "\"" + query_text + "\""
            if doc_text.find(',') != -1:
                doc_text = "\"" + doc_text + "\""
            if negtive_text.find(',') != -1:
                negtive_text = "\"" + negtive_text + "\""
                
            validate_data.append(query_text + ',' + doc_text + ',' + negtive_text + '\n')
        f.writelines(validate_data)

def main():
    # 将数据转换为字典，方便后续使用id查找
    corpus_dict = to_dict('../data/processed_data/corpus.tsv')
    query_dict = to_dict('../data/processed_data/train.query.txt')

    # 将cpr数据转换为字典，方便后续使用id查找
    cpr_corpus_dict = to_dict('../data/processed_data_cpr/corpus.tsv')
    cpr_query_dict = to_dict('../data/processed_data_cpr/train.query.txt')

    # 构造含负样本的数据
    construct_negtive_sample(qrels_path='../data/original_data/qrels.train.tsv', 
                                cpr_qrels_path='../data/original_data_cpr/qrels.train.tsv',
                                corpus_dict=corpus_dict,
                                query_dict=query_dict,
                                cpr_corpus_dict=cpr_corpus_dict,
                                cpr_query_dict=cpr_query_dict,
                                train_data_path='../data/data_neg_with_cpr/train_data.csv',
                                validate_data_path='../data/data_neg_with_cpr/validate_data.csv')

    # 加cpr数据，负样本id仍然是原始比赛数据中的，全都放在train_data.csv
if __name__ == '__main__':
    main()

