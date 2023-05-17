from tianchi_args import *
from tianchi_init import *


def get_train_dataloader():
    pos_input_ids = np.load(train_pos_input_ids_file)
    neg_input_ids = np.load(train_neg_input_ids_file)
    pos_token_type_ids = np.load(train_pos_segment_file)
    neg_token_type_ids = np.load(train_neg_segment_file)
    neg_doc_distance = np.load(train_neg_doc_distance_file)
    return Dataloader5(list(pos_input_ids),
                       list(pos_token_type_ids),
                       list(neg_input_ids),
                       list(neg_token_type_ids),
                       list(neg_doc_distance),
                       batch_size)


def get_dev_dataloader():
    input_ids = np.load(dev_input_ids_file)
    token_type_ids = np.load(dev_input_segment_file)
    labels = np.load(dev_labels_file)
    return Dataloader3(list(input_ids),
                       list(token_type_ids),
                       list(labels),
                       batch_size)
