import argparse


class BasedArgumentParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__()
        self.add_argument('--task', type=str, default='', help="Type of different jobs.")
        self.add_argument('--vocab_file', type=str, default=None, help="")
        self.add_argument('--data_path', type=str, default=None, help="")
        self.add_argument('--checkpoint', type=str, default=None, help="")
        self.add_argument('--lr', type=float, default=None, help="")
        self.add_argument('--epochs', type=int, default=None, help="")
        self.add_argument('--seq_length', type=int, default=None, help="")
        self.add_argument('--num_classes', type=int, default=None, help="")
        self.add_argument('--vocab_size', type=int, default=None, help="")
        self.add_argument('--batch_size', type=int, default=None, help="")
        self.add_argument('--eval_steps', type=int, default=None, help="")
        self.add_argument('--save_steps', type=int, default=None, help="")
        self.add_argument('--show_steps', type=int, default=None, help="")
        self.add_argument('--hidden_dropout_prob', type=float, default=None, help="")
        self.add_argument('--attention_probs_dropout_prob', type=float, default=None, help="")
        self.add_argument('--num_hidden_layers', type=int, default=None, help="")
        self.add_argument('--hidden_size', type=int, default=None, help="")
        self.add_argument('--num_attention_heads', type=int, default=None, help="")
        self.add_argument('--intermediate_size', type=int, default=None, help="")
        self.add_argument('--use_positional_embeddings', type=str, default='true')
        self.add_argument('--use_token_type_embeddings', type=str, default='true')
        self.add_argument('--model_type', type=str, default=None)


def get_args():
    parser = BasedArgumentParser()
    return parser.parse_args()


def get_pairwise_args():
    parser = BasedArgumentParser()
    parser.add_argument('--margin', type=float, default=None)
    return parser.parse_args()
