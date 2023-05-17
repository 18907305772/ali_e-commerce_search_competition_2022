import csv
from random import shuffle
from tqdm import tqdm
import random

random.seed(2022)

cpr_corpus_files=['../data/processed_data_cpr/corpus.tsv', '../data/processed_data/corpus.tsv']
cpr_query_files=['../data/processed_data/train.query.txt', '../data/processed_data/dev.query.txt', '../data/processed_data_cpr/train.query.txt', '../data/processed_data_cpr/dev.query.txt']
train_file='../posttrain_data/posttrain_data.txt'

post_training_doc_list = []
post_training_query_list = []
#--------
# 把corpus载入post_training_data_list
for corpus_file in cpr_corpus_files:
    corpus_reader = csv.reader(open(corpus_file), delimiter='\t')
    for line in corpus_reader:
        corpus = line[1]
        post_training_doc_list.append(corpus)
# 随机抽取一万条doc作为测试数据的一部分
#-------------
# debug
# post_training_doc_list = post_training_doc_list[:1000]
#-------------
shuffle(post_training_doc_list)
train_lines = [line + '\n' for line in post_training_doc_list]

#---------
# 把训练集中的query载入到post_training_data_list
for query_file in cpr_query_files:
    train_query_reader = csv.reader(open(query_file), delimiter='\t')
    for line in train_query_reader:
        query = line[1]
        post_training_query_list.append(query)    
# 随机抽取一千条query作为测试数据的一部分  
#-------------
# debug
# post_training_query_list = post_training_query_list[:100]
#-------------
shuffle(post_training_query_list)
train_lines.extend([line + '\n' for line in post_training_query_list])

shuffle(train_lines)
train_set = set(train_lines)

print(len(train_set))

with open(train_file, 'w', encoding='utf-8') as train_writer:
    train_writer.writelines(train_set)