import numpy as np

from src.optim.optimizer import Adam
from src.optim.scheduler import PolynomialScheduler
from tianchi_args import *
from tianchi_dataloader import get_train_dataloader
from tianchi_init import *


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_random_seed(seed)


def evaluate(ckpt_file):
    input_ids_list = np.reshape(np.load(dev_input_ids_file), [-1, 36, seq_length])
    token_type_ids_list = np.reshape(np.load(dev_input_segment_file), [-1, 36, seq_length])
    config = BertConfig(seq_length=seq_length).from_json(config_file)
    model = RobertaForPairwiseMatchingPrediction(config)
    model.compile(training=False,
                  input_ids=tf.placeholder(tf.int32, [None, seq_length]),
                  token_type_ids=tf.placeholder(tf.int32, [None, seq_length]))
    model.load(ckpt_file)
    mrr = 0.0
    mrr_top10 = 0.0
    step = 0
    total_steps = len(input_ids_list)
    print("Evaluating......")
    for input_ids, token_type_ids in zip(input_ids_list, token_type_ids_list):
        outputs = model.evaluate(input_ids=input_ids, token_type_ids=token_type_ids)
        score = outputs.logits
        score = np.array(score)[:, 0]  # (b,)
        score_top10 = score[:11]
        s = np.argsort(-score)  # (b,)
        s_top10 = np.argsort(-score_top10)
        rank = np.where(s == 0)[0] + 1
        rank_top10 = np.where(s_top10 == 0)[0] + 1
        if rank <= 10:
            batch_mrr = 1.0 / rank
        else:
            batch_mrr = 0.0
        batch_mrr_top10 = 1.0 / rank_top10 if rank_top10 <= 10 else 0.0
        mrr += batch_mrr
        mrr_top10 += batch_mrr_top10
        if step % int(show_steps * 10) == 0:
            print(f'Evaluating - step {step} of {total_steps}')
        step += 1
    mrr = mrr / total_steps
    mrr_top10 = mrr_top10 / total_steps
    print("mrr10: top10-{}, top36-{}".format(mrr_top10, mrr))
    print("mrr10_true: top10-{}, top36-{}".format(mrr_top10*0.5416, mrr*0.713))  # 0.713 36


def train():
    dataloader = get_train_dataloader()
    total_steps = int(epochs * dataloader.maximum_steps)
    config = BertConfig(seq_length=seq_length).from_json(config_file)
    model = RobertaForPairwiseMatchingWithRDrop(config, margin=margin)
    optimizer = PolynomialScheduler(Adam(lr=lr), total_steps).apply()
    model.compile(optimizer=optimizer,
                  pos_input_ids=tf.placeholder(tf.int32, [None, seq_length]),
                  neg_input_ids=tf.placeholder(tf.int32, [None, seq_length]),
                  pos_token_type_ids=tf.placeholder(tf.int32, [None, seq_length]),
                  neg_token_type_ids=tf.placeholder(tf.int32, [None, seq_length]),
                  neg_doc_distance=tf.placeholder(tf.float32, [None]))
    model.load(checkpoint)

    step = 0
    for epoch in range(epochs):
        for pos_input_ids, pos_token_type_ids, neg_input_ids, neg_token_type_ids, neg_doc_distance in dataloader:
            outputs = model.train(
                pos_input_ids=pos_input_ids,
                neg_input_ids=neg_input_ids,
                pos_token_type_ids=pos_token_type_ids,
                neg_token_type_ids=neg_token_type_ids,
                neg_doc_distance=neg_doc_distance
            )
            predicts = np.argmax(outputs.logits, axis=-1)
            if step % show_steps == 0:
                print(f'{epoch} epoch of {epochs} - step {step} of {total_steps} - loss {outputs.loss}')
            if step % eval_steps == 0:
                print('predicts: ', predicts)
            if step % save_steps == 0:
                ckpt_file = model.save(save_path, step=step)
                evaluate(ckpt_file)
            step += 1
    model.save(save_path)


if __name__ == '__main__':
    set_seed(1000)
    train()
