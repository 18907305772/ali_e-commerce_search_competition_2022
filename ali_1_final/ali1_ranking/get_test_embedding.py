import pandas as pd
import numpy as np

emb_path = "../ali_1/result/baseline/checkpoint-1400/"

doc_emb_path = emb_path + "/doc_embedding"

query_emb_path = emb_path + "/query_embedding"

def load_embedding_from_txt(read_path, save_path, n):
    embedding_df = pd.read_csv(read_path, sep="\t", index_col=0, names=["emb"])
    embedding_list = list()
    for i in range(n):
        emb = embedding_df.iloc[i, 0].split(',')
        new_emb = list()
        for x in emb:
            new_emb.append(float(x))
        embedding_list.append(np.array(new_emb))
    embedding_np = np.array(embedding_list)
    np.save(save_path, embedding_np)
    return embedding_np


doc_emb = load_embedding_from_txt(doc_emb_path, emb_path + "/final_test_doc_emb.npy", 1001500)
query_emb = load_embedding_from_txt(query_emb_path, emb_path + "/final_test_query_emb.npy", 1000)
