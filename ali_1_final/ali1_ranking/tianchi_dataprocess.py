from src.utils.tokenizer import Tokenizer
import numpy as np
import pandas as pd
import os
import csv
import faiss
from tqdm import tqdm


def read_corpus(file_path=None):
    reader = csv.reader(open(file_path), delimiter='\t')
    total_dict = dict()
    for line in reader:
        corpus_id = int(line[0])
        corpus = line[1]
        total_dict[corpus_id] = corpus
    return total_dict


# 前期处理部分
def data_process_pair():
    """
    构建训练验证数据
    输入：
    - 原始lowsimple数据: data_path
    - 初赛roberta-base单模各种embedding: chusai_result_path
    输出：
    - 处理成pair后的训练验证数据: data_process_path
    """
    data_path = "../ali_1/data/processed_data/"
    data_process_path = "data/tianchi_data_processed/"
    if not os.path.exists(data_process_path):
        os.mkdir(data_process_path)
    chusai_result_path = "../ali_1/result/baseline/checkpoint-1400/"

    train_data_path = data_path + "train.query.txt"

    # 训练部分
    doc_embedding_save_path = chusai_result_path + 'final_test_doc_emb.npy'
    doc_embedding_np = np.load(doc_embedding_save_path)

    # 构建doc查找索引
    doc_embedding_np = doc_embedding_np.astype('float32')
    index = faiss.IndexFlatL2(128)
    index.add(doc_embedding_np)

    train_dev_query_data = [line[1] for line in csv.reader(open(train_data_path), delimiter='\t')]
    train_query_data = train_dev_query_data[:-5000]

    train_query_embedding_save_path = chusai_result_path + 'train_query.npy'
    train_query_np = np.load(train_query_embedding_save_path)

    # 建立train_query对应的doc_label
    train_query_doc_match_path = data_path + "qrels.train.tsv"
    train_query_doc_match = pd.read_csv(train_query_doc_match_path, sep='\t', names=["query_ids", "doc_ids"])
    train_query_doc_match_list = list()
    for i in range(95000):
        train_query_doc_match_list.append(train_query_doc_match.iloc[i, 1])

    corpus_dict = read_corpus(data_path + "corpus.tsv")

    train_query_doc_final = csv.writer(open(data_process_path + "train_pairwise.csv", 'w'), delimiter='\t')
    train_query_doc_final.writerow(["query", "pos_doc", "neg_doc", "neg_doc_distance"])

    train_query_np = train_query_np.astype('float32')
    D, I = index.search(train_query_np, 36)  # (95000, 36)
    for x in range(95000):
        for y in range(36):
            I[x][y] += 1

    for i in tqdm(range(95000)):
        train_query_text = train_query_data[i]
        train_pos_doc_id = train_query_doc_match_list[i]
        train_pos_doc = corpus_dict[train_pos_doc_id]
        for x in range(35):
            if I[i][x] == train_pos_doc_id:
                I[i][x] = I[i][35]
                D[i][x] = D[i][35]
                break
        for j in range(35):
            train_neg_doc_id = I[i][j]
            train_neg_doc = corpus_dict[train_neg_doc_id]
            train_neg_doc_distance = D[i][j]
            train_query_doc_final.writerow([train_query_text, train_pos_doc, train_neg_doc, train_neg_doc_distance])
            if i < 3:
                print("query:{}, pos_doc:{}, neg_doc:{}, neg_doc_distance:{}".format(train_query_text, train_pos_doc,
                                                                                     train_neg_doc,
                                                                                     train_neg_doc_distance))

    # 验证部分
    dev_query_data = train_dev_query_data[-5000:]

    dev_query_embedding_save_path = chusai_result_path + 'dev_query.npy'
    dev_query_np = np.load(dev_query_embedding_save_path)

    # 建立dev_query对应的doc_label
    dev_query_doc_match_list = list()
    for i in range(95000, 100000):
        dev_query_doc_match_list.append(train_query_doc_match.iloc[i, 1])

    dev_query_doc_final = csv.writer(open(data_process_path + "dev_pairwise.csv", 'w'), delimiter='\t')
    dev_query_doc_final.writerow(["query", "doc", "label", "query_id"])

    dev_query_np = dev_query_np.astype('float32')
    _, I1 = index.search(dev_query_np, 36)  # (5000, 36)
    for x in range(5000):
        for y in range(36):
            I1[x][y] += 1

    for i in tqdm(range(5000)):
        dev_query_text = dev_query_data[i]
        dev_pos_doc_id = dev_query_doc_match_list[i]
        dev_pos_doc = corpus_dict[dev_pos_doc_id]
        for x in range(35):
            if I1[i][x] == dev_pos_doc_id:
                I1[i][x] = I1[i][35]
                break
        dev_query_doc_final.writerow([dev_query_text, dev_pos_doc, "1", str(95001 + i)])
        for j in range(35):
            dev_neg_doc_id = I1[i][j]
            dev_neg_doc = corpus_dict[dev_neg_doc_id]
            dev_query_doc_final.writerow([dev_query_text, dev_neg_doc, "0", str(95001 + i)])
            if i < 3:
                print("query:{}, pos_doc:{}, neg_doc:{}".format(dev_query_text, dev_pos_doc, dev_neg_doc))
    return data_process_path


def data_process_tokenize(data_pair_path):
    """
    训练验证数据tokenize
    输入：
    - 训练验证pair数据: data_process_path
    - posttrain-roberta-base tokenizer: tokenizer
    输出：
    - tokenize后的训练验证数据: data_tokenized_path
    """
    # 文件路径
    data_process_path = data_pair_path
    data_tokenized_path = "data/tianchi_data_tokenized/"
    if not os.path.exists(data_tokenized_path):
        os.mkdir(data_tokenized_path)

    # 读取文件
    train_data = csv.reader(open(data_process_path + "train_pairwise.csv"), delimiter='\t')
    dev_data = csv.reader(open(data_process_path + "dev_pairwise.csv"), delimiter='\t')

    # 处理训练数据
    tokenizer = Tokenizer(vocab_file='config/posttrain_roberta_wwm_ext/vocab.txt')
    train_pos_input_ids = []
    train_pos_token_type = []
    train_neg_input_ids = []
    train_neg_token_type = []
    train_neg_doc_distance = []

    next(train_data)
    train_data = [line for line in train_data]
    print(len(train_data))
    cnt = 0
    for line in tqdm(train_data):
        query, pos_doc, neg_doc, neg_doc_distance = line[0], line[1], line[2], line[3]
        pos_input_ids = tokenizer.text2ids(query + tokenizer.SEP_TOKEN + pos_doc, seq_length=128)
        pos_token_type = tokenizer.ids2token_type_ids(pos_input_ids)
        neg_input_ids = tokenizer.text2ids(query + tokenizer.SEP_TOKEN + neg_doc, seq_length=128)
        neg_token_type = tokenizer.ids2token_type_ids(neg_input_ids)
        train_pos_input_ids.append(pos_input_ids)
        train_pos_token_type.append(pos_token_type)
        train_neg_input_ids.append(neg_input_ids)
        train_neg_token_type.append(neg_token_type)
        train_neg_doc_distance.append(float(neg_doc_distance))
        cnt += 1
    print("train_cnt:", cnt)

    # 处理验证数据
    dev_query_doc_input_ids = []
    dev_query_doc_token_type = []
    dev_label = []
    dev_query_id = []

    next(dev_data)
    dev_data = [line for line in dev_data]
    print(len(dev_data))
    cnt = 0
    for line in tqdm(dev_data):
        dev_query, dev_doc, label, query_id = line[0], line[1], line[2], line[3]
        query_doc_input_ids = tokenizer.text2ids(dev_query + tokenizer.SEP_TOKEN + dev_doc, seq_length=128)
        query_doc_token_type = tokenizer.ids2token_type_ids(query_doc_input_ids)
        dev_query_doc_input_ids.append(query_doc_input_ids)
        dev_query_doc_token_type.append(query_doc_token_type)
        dev_label.append(label)
        dev_query_id.append(query_id)
        cnt += 1
    print("dev cnt:", cnt)

    # 保存结果
    train_pos_input_ids = np.array(train_pos_input_ids)
    train_pos_token_type = np.array(train_pos_token_type)
    train_neg_input_ids = np.array(train_neg_input_ids)
    train_neg_token_type = np.array(train_neg_token_type)
    train_neg_doc_distance = np.array(train_neg_doc_distance)

    np.save(data_tokenized_path + "train_pos_input_ids.npy", train_pos_input_ids)
    np.save(data_tokenized_path + "train_pos_token_type.npy", train_pos_token_type)
    np.save(data_tokenized_path + "train_neg_input_ids.npy", train_neg_input_ids)
    np.save(data_tokenized_path + "train_neg_token_type.npy", train_neg_token_type)
    np.save(data_tokenized_path + "train_neg_doc_distance.npy", train_neg_doc_distance)

    dev_query_doc_input_ids = np.array(dev_query_doc_input_ids)
    dev_query_doc_token_type = np.array(dev_query_doc_token_type)
    dev_label = np.array(dev_label)
    dev_query_id = np.array(dev_query_id)

    np.save(data_tokenized_path + "dev_query_doc_input_ids.npy", dev_query_doc_input_ids)
    np.save(data_tokenized_path + "dev_query_doc_token_type.npy", dev_query_doc_token_type)
    np.save(data_tokenized_path + "dev_label.npy", dev_label)
    np.save(data_tokenized_path + "dev_query_id.npy", dev_query_id)


def data_process_test():
    """
    测试数据构建
    输入：
    - 原始lowsimple数据: data_path
    - 初赛robeta-base单模结果数据: chusai_result_path
    输出：
    - 提交文件: submit_path
    """
    data_path = "../ali_1/data/processed_data/"
    chusai_result_path = "../ali_1/result/baseline/checkpoint-1400/"
    submit_path = "submit/"
    if not os.path.exists(submit_path):
        os.mkdir(submit_path)
    tokenizer = Tokenizer(vocab_file='config/posttrain_roberta_wwm_ext/vocab.txt')
    # 测试集query tokenize
    test_query = csv.reader(open(data_path + "dev.query.txt"), delimiter='\t')
    test_query = [line for line in test_query]

    test_query_tokenized = []
    for line in tqdm(test_query):
        id, query = line[0], line[1]
        test_query_tokenized.append(tokenizer.text2ids(query, seq_length=128))

    # query ids和embedding放到一个文件里
    test_query_embedding = csv.reader(open(chusai_result_path + 'query_embedding'), delimiter='\t')
    test_query_embedding = [line for line in test_query_embedding]

    query_embedding_processed = csv.writer(open(submit_path + 'query_embedding', 'w'), delimiter='\t')
    for i, line in enumerate(test_query_embedding):
        qid, emb = line[0], line[1]
        ids = test_query_tokenized[i]
        ids = [str(s) for s in ids]
        ids = ','.join(ids)
        query_embedding_processed.writerow([qid, emb, ids])

    # 对doc同理操作
    doc = csv.reader(open(data_path + "corpus.tsv"), delimiter='\t')
    doc = [line for line in doc]

    doc_tokenized = []
    for line in tqdm(doc):
        id, d = line[0], line[1]
        d_token = tokenizer.text2ids(d, seq_length=129)
        d_token = d_token[1:]
        doc_tokenized.append(d_token)

    doc_embedding = csv.reader(open(chusai_result_path + 'doc_embedding'), delimiter='\t')
    doc_embedding = [line for line in doc_embedding]

    doc_embedding_processed = csv.writer(open(submit_path + 'doc_embedding', 'w'), delimiter='\t')
    for i, line in enumerate(doc_embedding):
        did, emb = line[0], line[1]
        ids = doc_tokenized[i]
        ids = [str(s) for s in ids]
        ids = ','.join(ids)
        doc_embedding_processed.writerow([did, emb, ids])


data_pair_path = data_process_pair()
data_process_tokenize(data_pair_path)
data_process_test()
print("finished!")
