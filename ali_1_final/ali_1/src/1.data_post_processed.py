from multiprocessing import context
import pandas as pd
from utils import post_process


def main():
    # 处理比赛数据
    # 处理 train.query.txt
    train_query = []
    with open('../data/original_data/train.query.txt', 'r') as f:
        for line in f.readlines():
            content = line.split('\t')
            id = content[0]
            text = content[1]
            if text.find('\n') == -1:
                test = text + '\n'
            # 大写转小写 + 繁体转简体
            text = post_process(text)
            line = (id + '\t' + text).rstrip('\n') + '\n'
            train_query.append(line)
    with open('../data/processed_data/train.query.txt', 'w') as f:
        f.writelines(train_query)

    # 处理 dev.query.txt
    dev_query = []
    with open('../data/original_data/dev.query.txt', 'r') as f:
        for line in f.readlines():
            content = line.split('\t')
            id = content[0]
            text = content[1]
            if text.find('\n') == -1:
                test = text + '\n'
            # 大写转小写 + 繁体转简体
            text = post_process(text)
            line = (id + '\t' + text).rstrip('\n') + '\n'
            dev_query.append(line)
    with open('../data/processed_data/dev.query.txt', 'w') as f:
        f.writelines(dev_query)

    # 处理 corpus.tsv
    corpus = []
    with open('../data/original_data/corpus.tsv', 'r') as f:
        for line in f.readlines():
            content = line.split('\t')
            id = content[0]
            text = content[1]
            if text.find('\n') == -1:
                test = text + '\n'
            # 大写转小写 + 繁体转简体
            text = post_process(text)
            line = (id + '\t' + text).rstrip('\n') + '\n'
            corpus.append(line)
    with open('../data/processed_data/corpus.tsv', 'w') as f:
        f.writelines(corpus)

    # 处理cpr数据
    with open('../data/original_data_cpr/train.query.txt', 'r') as f:
        for line in f.readlines():
            content = line.split('\t')
            id = content[0]
            text = content[1]
            if text.find('\n') == -1:
                test = text + '\n'
            # 大写转小写 + 繁体转简体
            text = post_process(text)
            line = (id + '\t' + text).rstrip('\n') + '\n'
            train_query.append(line)
    with open('../data/processed_data_cpr/train.query.txt', 'w') as f:
        f.writelines(train_query)

    # 处理 dev.query.txt
    dev_query = []
    with open('../data/original_data_cpr/dev.query.txt', 'r') as f:
        for line in f.readlines():
            content = line.split('\t')
            id = content[0]
            text = content[1]
            if text.find('\n') == -1:
                test = text + '\n'
            # 大写转小写 + 繁体转简体
            text = post_process(text)
            line = (id + '\t' + text).rstrip('\n') + '\n'
            dev_query.append(line)
    with open('../data/processed_data_cpr/dev.query.txt', 'w') as f:
        f.writelines(dev_query)

    # 处理 corpus.tsv
    corpus = []
    with open('../data/original_data_cpr/corpus.tsv', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            content = line.split('\t')
            id = content[0]
            text = content[1]
            if text.find('\n') == -1:
                test = text + '\n'
            # 大写转小写 + 繁体转简体
            text = post_process(text)
            line = (id + '\t' + text).rstrip('\n') + '\n'
            corpus.append(line)
    with open('../data/processed_data_cpr/corpus.tsv', 'w') as f:
        f.writelines(corpus)
    
    # 写入qrels.train.tsv
    qrels_train = []
    with open('../data/original_data/qrels.train.tsv', 'r') as f:
        for line in f.readlines():
            line = line.rstrip('\n') + '\n'
            qrels_train.append(line)
    with open('../data/processed_data/qrels.train.tsv', 'w') as f:
        f.writelines(qrels_train)

if __name__ == "__main__":
    main()
