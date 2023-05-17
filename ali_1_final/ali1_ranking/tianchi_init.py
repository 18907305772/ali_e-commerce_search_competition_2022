import numpy as np
import tensorflow as tf
import random
# from transformers import set_seed

from src.utils.ckpt_util import print_checkpoint_variables
from src.models.roberta import RobertaForPairwiseMatching, RobertaForPairwiseMatchingPrediction, RobertaForPairwiseMatchingWithRDrop
from src.utils.config_util import BertConfig
from src.preprocess.dataloader import Dataloader4, Dataloader3, Dataloader5
from src.preprocess.tokenizer import Tokenizer

# set_seed(5)