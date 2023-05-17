# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, CTRL, BERT, RoBERTa, XLNet).
GPT, GPT-2 and CTRL are fine-tuned using a causal language modeling (CLM) loss. BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss. XLNet is fine-tuned using a permutation language modeling (PLM) loss.
"""


from ast import Raise
from distutils.log import error
import logging
import math
import os
import json
from dataclasses import dataclass, field
from datasets import Dataset
from typing import Optional
from numpy import str0
from transformers import BertTokenizer, BertForMaskedLM, BertConfig, BertLMHeadModel
from transformers import AutoTokenizer, AutoModel

from transformers import (
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorForWholeWordMask,
    HfArgumentParser,
    LineByLineTextDataset,
    PreTrainedTokenizer,
    TextDataset,
    Trainer,
    TrainingArguments,
    set_seed,
)


logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

# transformers 相关
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

from numpy import mask_indices
from transformers.file_utils import PaddingStrategy
from transformers.models.bert import BertTokenizer, BertTokenizerFast
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase



@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_data_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    eval_data_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    line_by_line: bool = field(
        default=True,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )

    mlm: Optional[str] = field(
        default=None, metadata={"help": "Train with masked-language modeling loss instead of language modeling."}
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    plm_probability: float = field(
        default=1 / 6,
        metadata={
            "help": "Ratio of length of a span of masked tokens to surrounding context length for permutation language modeling."
        },
    )
    max_span_length: int = field(
        default=5, metadata={"help": "Maximum length of a span of masked tokens for permutation language modeling."}
    )

    block_size: int = field(
        default=512,
        metadata={
            "help": "Optional input sequence length after tokenization."
            "The training dataset will be truncated in block of this size for training."
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    train_ref_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input train ref data file for whole word masking in Chinese."},
    )
    validation_ref_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input validation ref data file for whole word masking in Chinese."},
    )


def get_dataset(args: DataTrainingArguments, tokenizer: PreTrainedTokenizer, evaluate=False):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    if args.line_by_line:
        return LineByLineTextDataset(tokenizer=tokenizer, file_path=file_path, block_size=args.block_size)
    else:
        return TextDataset(
            tokenizer=tokenizer, file_path=file_path, block_size=args.block_size, overwrite_cache=args.overwrite_cache
        )

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # model_args.config_name =
    if data_args.eval_data_file is None and training_args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        config = BertConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
        # config = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        tokenizer = BertTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
        # tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name"
        )

    if model_args.model_name_or_path:
        # model = AutoModelWithLMHead.from_pretrained(
        #     model_args.model_name_or_path,
        #     from_tf=bool(".ckpt" in model_args.model_name_or_path),
        #     config=config,
        #     cache_dir=model_args.cache_dir,
        # )
        model = BertForMaskedLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelWithLMHead.from_config(config)

    model.resize_token_embeddings(len(tokenizer))

    if config.model_type in ["bert", "roberta", "distilbert", "camembert"] and not data_args.mlm:
        raise ValueError(
            "BERT and RoBERTa-like models do not have LM heads but masked LM heads. They must be run using the"
            "--mlm flag (masked language modeling)."
        )

    if data_args.block_size <= 0:
        data_args.block_size = tokenizer.max_len
        # Our input block size will be the max possible for the model
    else:
        data_args.block_size = min(data_args.block_size, tokenizer.model_max_length)

    # Get datasets
    print('data args--------------------', data_args)
    train_dataset = get_dataset(data_args, tokenizer=tokenizer) if training_args.do_train else None
    eval_dataset = get_dataset(data_args, tokenizer=tokenizer, evaluate=True) if training_args.do_eval else None
    
    # ngram data collector
    #--------------------------------------
    class DataCollatorMixin:
        def __call__(self, features, return_tensors=None):
            if return_tensors is None:
                return_tensors = self.return_tensors
            if return_tensors == "tf":
                return self.tf_call(features)
            elif return_tensors == "pt":
                return self.torch_call(features)
            elif return_tensors == "np":
                return self.numpy_call(features)
            else:
                raise ValueError(f"Framework '{return_tensors}' not recognized!")
                import random

    @dataclass
    class DataCollatorForNgramMasking(DataCollatorMixin):
        """
        Data collator used for language modeling. Inputs are dynamically padded to the maximum length of a batch if they
        are not all of the same length.

        Args:
            tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
                The tokenizer used for encoding the data.
            mlm (`bool`, *optional*, defaults to `True`):
                Whether or not to use masked language modeling. If set to `False`, the labels are the same as the inputs
                with the padding tokens ignored (by setting them to -100). Otherwise, the labels are -100 for non-masked
                tokens and the value to predict for the masked token.
            mlm_probability (`float`, *optional*, defaults to 0.15):
                The probability with which to (randomly) mask tokens in the input, when `mlm` is set to `True`.
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the sequence to a multiple of the provided value.
            return_tensors (`str`):
                The type of Tensor to return. Allowable values are "np", "pt" and "tf".

        <Tip>

        For best performance, this data collator should be used with a dataset having items that are dictionaries or
        BatchEncoding, with the `"special_tokens_mask"` key, as returned by a [`PreTrainedTokenizer`] or a
        [`PreTrainedTokenizerFast`] with the argument `return_special_tokens_mask=True`.

        </Tip>"""

        tokenizer: PreTrainedTokenizerBase
        mlm: bool = True
        mlm_probability: float = 0.15
        pad_to_multiple_of: Optional[int] = None
        tf_experimental_compile: bool = False
        return_tensors: str = "pt"

        def __post_init__(self):
            if self.mlm and self.tokenizer.mask_token is None:
                raise ValueError(
                    "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                    "You should pass `mlm=False` to train on causal language modeling instead."
                )
            if self.tf_experimental_compile:
                import tensorflow as tf

                self.tf_mask_tokens = tf.function(self.tf_mask_tokens, jit_compile=True)

        @staticmethod
        def tf_bernoulli(shape, probability):
            import tensorflow as tf

            prob_matrix = tf.fill(shape, probability)
            return tf.cast(prob_matrix - tf.random.uniform(shape, 0, 1) >= 0, tf.bool)

        def tf_mask_tokens(
            self, inputs: Any, vocab_size, mask_token_id, special_tokens_mask: Optional[Any] = None
        ) -> Tuple[Any, Any]:
            """
            Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
            """
            import tensorflow as tf

            input_shape = tf.shape(inputs)
            # 1 for a special token, 0 for a normal token in the special tokens mask
            # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
            masked_indices = self.tf_bernoulli(input_shape, self.mlm_probability) & ~special_tokens_mask
            # Replace unmasked indices with -100 in the labels since we only compute loss on masked tokens
            labels = tf.where(masked_indices, inputs, -100)

            # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
            indices_replaced = self.tf_bernoulli(input_shape, 0.8) & masked_indices

            inputs = tf.where(indices_replaced, mask_token_id, inputs)

            # 10% of the time, we replace masked input tokens with random word
            indices_random = self.tf_bernoulli(input_shape, 0.1) & masked_indices & ~indices_replaced
            random_words = tf.random.uniform(input_shape, maxval=vocab_size, dtype=tf.int64)
            inputs = tf.where(indices_random, random_words, inputs)

            # The rest of the time (10% of the time) we keep the masked input tokens unchanged
            return inputs, labels

        def tf_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
            import tensorflow as tf

            # Handle dict or lists with proper padding and conversion to tensor.
            if isinstance(examples[0], (dict, BatchEncoding)):
                batch = self.tokenizer.pad(examples, return_tensors="tf", pad_to_multiple_of=self.pad_to_multiple_of)
            else:
                batch = {
                    "input_ids": _tf_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
                }

            # If special token mask has been preprocessed, pop it from the dict.
            special_tokens_mask = batch.pop("special_tokens_mask", None)
            if self.mlm:
                if special_tokens_mask is None:
                    special_tokens_mask = [
                        self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
                        for val in batch["input_ids"].numpy().tolist()
                    ]
                    # Cannot directly create as bool
                    special_tokens_mask = tf.cast(tf.convert_to_tensor(special_tokens_mask, dtype=tf.int64), tf.bool)
                else:
                    special_tokens_mask = tf.cast(special_tokens_mask, tf.bool)
                batch["input_ids"], batch["labels"] = self.tf_mask_tokens(
                    tf.cast(batch["input_ids"], tf.int64),
                    special_tokens_mask=special_tokens_mask,
                    mask_token_id=self.tokenizer.mask_token_id,
                    vocab_size=len(self.tokenizer),
                )
            else:
                labels = batch["input_ids"]
                if self.tokenizer.pad_token_id is not None:
                    # Replace self.tokenizer.pad_token_id with -100
                    labels = tf.where(labels == self.tokenizer.pad_token_id, -100, labels)
                else:
                    labels = tf.identity(labels)  # Makes a copy, just in case
                batch["labels"] = labels
            return batch

        def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
            # Handle dict or lists with proper padding and conversion to tensor.
            if isinstance(examples[0], (dict, BatchEncoding)):
                batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
            else:
                batch = {
                    "input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
                }

            # If special token mask has been preprocessed, pop it from the dict.
            special_tokens_mask = batch.pop("special_tokens_mask", None)
            if self.mlm == 'ngram':
                # ngram mask
                batch["input_ids"], batch["labels"] = self.torch_ngram_mask_tokens(
                    batch["input_ids"], special_tokens_mask=special_tokens_mask
                )
            else:
                labels = batch["input_ids"].clone()
                if self.tokenizer.pad_token_id is not None:
                    labels[labels == self.tokenizer.pad_token_id] = -100
                batch["labels"] = labels
            return batch

        #------------------
        # ngram mask token
        def torch_ngram_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
            """
            Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
            """
            import torch
            import numpy as np

            labels = inputs.clone()
            # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
            probability_matrix = torch.full(labels.shape, self.mlm_probability)
            if special_tokens_mask is None:
                special_tokens_mask = [
                    self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
                ]
                special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
            else:
                special_tokens_mask = special_tokens_mask.bool()

            probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
            masked_indices = torch.bernoulli(probability_matrix).bool()
            # ngram mask设定下的masked_indices
            batch_size, time_steps = masked_indices.shape
            for b in range(batch_size):
                # [SEP] token 的index
                sep_index = torch.nonzero(probability_matrix[b] == 0.0)[1]
                for text_index in range(1, sep_index):
                    if torch.bernoulli(probability_matrix[b, text_index]).bool():
                        seq_len = sep_index - 1
                        ngram=np.random.choice([1,2,3], p=[0.7,0.2,0.1])#若要mask，进行x_gram mask的概率
                        if ngram==3 and seq_len<7:#太大的gram不要应用于过短文本
                            ngram=2
                        if ngram==2 and seq_len<4:
                            ngram=1
                        left_boundary = text_index
                        right_boundary = min(left_boundary + ngram, sep_index)
                        masked_indices[b, left_boundary : right_boundary] = True
                        # 禁止mask片段的下一个token被mask，防止一大片连续mask
                        probability_matrix[b, right_boundary] = 0.0

            labels[~masked_indices] = -100  # We only compute loss on masked tokens

            # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
            indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
            inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

            # 10% of the time, we replace masked input tokens with random word
            indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
            random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
            inputs[indices_random] = random_words[indices_random]

            # The rest of the time (10% of the time) we keep the masked input tokens unchanged
            return inputs, labels
        #----------------------------

        def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
            """
            Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
            """
            import torch

            labels = inputs.clone()
            # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
            probability_matrix = torch.full(labels.shape, self.mlm_probability)
            if special_tokens_mask is None:
                special_tokens_mask = [
                    self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
                ]
                special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
            else:
                special_tokens_mask = special_tokens_mask.bool()

            probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
            masked_indices = torch.bernoulli(probability_matrix).bool()
            labels[~masked_indices] = -100  # We only compute loss on masked tokens

            # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
            indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
            inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

            # 10% of the time, we replace masked input tokens with random word
            indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
            random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
            inputs[indices_random] = random_words[indices_random]

            # The rest of the time (10% of the time) we keep the masked input tokens unchanged
            return inputs, labels

        def numpy_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
            import numpy as np

            # Handle dict or lists with proper padding and conversion to tensor.
            if isinstance(examples[0], (dict, BatchEncoding)):
                batch = self.tokenizer.pad(examples, return_tensors="np", pad_to_multiple_of=self.pad_to_multiple_of)
            else:
                batch = {
                    "input_ids": _numpy_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
                }

            # If special token mask has been preprocessed, pop it from the dict.
            special_tokens_mask = batch.pop("special_tokens_mask", None)
            if self.mlm:
                batch["input_ids"], batch["labels"] = self.numpy_mask_tokens(
                    batch["input_ids"], special_tokens_mask=special_tokens_mask
                )
            else:
                labels = np.copy(batch["input_ids"])
                if self.tokenizer.pad_token_id is not None:
                    labels[labels == self.tokenizer.pad_token_id] = -100
                batch["labels"] = labels
            return batch

        def numpy_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
            """
            Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
            """
            import numpy as np

            labels = np.copy(inputs)
            # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
            probability_matrix = np.full(labels.shape, self.mlm_probability)
            if special_tokens_mask is None:
                special_tokens_mask = [
                    self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
                ]
                special_tokens_mask = np.array(special_tokens_mask, dtype=np.bool)
            else:
                special_tokens_mask = special_tokens_mask.astype(np.bool)

            probability_matrix[special_tokens_mask] = 0
            # Numpy doesn't have bernoulli, so we use a binomial with 1 trial
            masked_indices = np.random.binomial(1, probability_matrix, size=probability_matrix.shape).astype(np.bool)
            labels[~masked_indices] = -100  # We only compute loss on masked tokens

            # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
            indices_replaced = np.random.binomial(1, 0.8, size=labels.shape).astype(np.bool) & masked_indices
            inputs[indices_replaced] = self.tokenizer.mask_token_id

            # 10% of the time, we replace masked input tokens with random word
            # indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
            indices_random = (
                np.random.binomial(1, 0.5, size=labels.shape).astype(np.bool) & masked_indices & ~indices_replaced
            )
            random_words = np.random.randint(
                low=0, high=len(self.tokenizer), size=np.count_nonzero(indices_random), dtype=np.int64
            )
            inputs[indices_random] = random_words

            # The rest of the time (10% of the time) we keep the masked input tokens unchanged
            return inputs, labels
    #--------------------------------------
    #--------------------------------------
    # 选择合适的 data collector
    # mlm
    if data_args.mlm == "ngram":
        print("----------NGRAM MASK----------")
        # data_collator = DataCollatorForNgramMasking(
        #     tokenizer=tokenizer, mlm=data_args.mlm, mlm_probability=data_args.mlm_probability
        # )
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability=data_args.mlm_probability
        )
    else:
        error("Error: wrong masking methods")

    # Initialize our Trainer
    training_args.prediction_loss_only = True
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Training
    if training_args.do_train:
        model_path = (
            model_args.model_name_or_path
            if model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path)
            else None
        )
        trainer.train(model_path=model_path)
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        eval_output = trainer.evaluate()

        perplexity = math.exp(eval_output["eval_loss"])
        result = {"perplexity": perplexity}

        output_eval_file = os.path.join(training_args.output_dir, "eval_results_lm.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

        results.update(result)

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()