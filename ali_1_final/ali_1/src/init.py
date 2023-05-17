# 普通
import os
import sys
import math
import time
import logging
import numpy as np

from sklearn.metrics import accuracy_score

# pytorch
import torch
import torch.nn.functional as F
import torch.distributed as dist

from torch import nn

import transformers

# 预设参数
from dataclasses import (
    dataclass,
    field,
)
from typing import (
    Optional,
    Tuple,
)
from transformers import (
    set_seed,
    HfArgumentParser,
    TrainingArguments,
)
from transformers.file_utils import (
    cached_property,  
    is_torch_tpu_available,
    ModelOutput,
    torch_required,
)

# 日志
from transformers.trainer_utils import is_main_process

# 数据集
from datasets import load_dataset

# 模型
from transformers import (
    AutoConfig,
    AutoTokenizer,
    CONFIG_MAPPING,     
    
    AutoModelForMaskedLM,
    BertPreTrainedModel,
)

from transformers.models.bert.modeling_bert import (
    BertModel,
)

from transformers.trainer import Trainer

from transformers.trainer_pt_utils import (
    find_batch_size,
    nested_concat,
    nested_numpify,
)

from transformers.trainer_utils import (
    denumpify_detensorize,
    EvalLoopOutput,
    speed_metrics,
)

from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions,
)

#----------------------------------------
# 常量
sep = os.sep

version_torch = torch.__version__
version_cuda = torch.version.cuda
version_cudnn = torch.backends.cudnn.version()
cuda_available = torch.cuda.is_available()
num_device = torch.cuda.device_count()

#----------------------------------------
# 创建文件夹
def initDir(dir_tgt):
    if not os.path.exists(dir_tgt):
        os.mkdir(dir_tgt)
    return