import os

from src.utils.args_util import get_pairwise_args

args = get_pairwise_args()

task = args.task
lr = args.lr if args.lr else 1e-5
epochs = args.epochs if args.epochs else 10
batch_size = args.batch_size if args.batch_size else 64
global_batch_size = 1 * batch_size
seq_length = args.seq_length if args.seq_length else 128
eval_steps = args.eval_steps if args.eval_steps else 500
save_steps = args.save_steps if args.save_steps else 5000
show_steps = args.show_steps if args.show_steps else 100
hidden_size = args.hidden_size
hidden_dropout_prob = args.hidden_dropout_prob
attention_probs_dropout_prob = args.attention_probs_dropout_prob
num_attention_heads = args.num_attention_heads
intermediate_size = args.intermediate_size
vocab_size = args.vocab_size
num_classes = args.num_classes if args.num_classes else 2
model_type = args.model_type if args.model_type else 'roberta-base-chinese'
config_path = os.path.join('config', model_type)
config_file = os.path.join(config_path, 'config.json')
vocab_file = os.path.join(config_path, 'vocab.txt')
checkpoint = args.checkpoint if args.checkpoint else os.path.join(config_path, f'{model_type}.ckpt')
result_path = os.path.join('result', 'tianchi')
save_path = os.path.join(result_path, task)
wrapper_save_path = os.path.join('submit', 'model')

margin = args.margin if args.margin else 0.1

data_file_name = 'tianchi_data_tokenized'
train_pos_segment_file = os.path.join('data', data_file_name, 'train_pos_token_type.npy')
train_pos_input_ids_file = os.path.join('data', data_file_name, 'train_pos_input_ids.npy')
train_neg_segment_file = os.path.join('data', data_file_name, 'train_neg_token_type.npy')
train_neg_input_ids_file = os.path.join('data', data_file_name, 'train_neg_input_ids.npy')
train_neg_doc_distance_file = os.path.join('data', data_file_name, 'train_neg_doc_distance.npy')
dev_input_ids_file = os.path.join('data', data_file_name, 'dev_query_doc_input_ids.npy')
dev_input_segment_file = os.path.join('data', data_file_name, 'dev_query_doc_token_type.npy')
dev_labels_file = os.path.join('data', data_file_name, 'dev_label.npy')
