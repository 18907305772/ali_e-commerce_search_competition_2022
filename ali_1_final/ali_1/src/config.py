# 导入库
from init import *

#----------------------------------------
# 参数：模型
@dataclass
class ModelArguments:

    # Huggingface 参数
    cache_dir: Optional[str] = field(default='../model_cache/')
    config_name: Optional[str] = field(default=None)
    model_name_or_path: Optional[str] = field(default='../posttrain_result/best_model/checkpoint-113960') 
    model_revision: str = field(default='main')
    model_type: Optional[str] = field(default=None)
    tokenizer_name: Optional[str] = field(default=None)
    use_auth_token: bool = field(default=False)
    use_fast_tokenizer: bool = field(default=True)
    
    # SimCSE 参数
    hard_negative_weight: float = field(default=0)
    mlp_only_train: bool = field(default=False)
    pooler_type: str = field(default='cls')
    temperature: float = field(default=0.05)

#----------------------------------------
# 参数：训练数据
@dataclass
class DataTrainingArguments:
    
    # Huggingface 参数
    dataset_name: Optional[str] = field(default=None)
    dataset_config_name: Optional[str] = field(default=None)
    overwrite_cache: bool = field(default=True)
    preprocessing_num_workers: Optional[int] = field(default=None)
    validation_split_percentage: Optional[int] = field(default=5)

    # SimCSE 参数
    # 路径
    dir_data: Optional[str] = field(default='../data')
    dir_data_cache: Optional[str] = field(default='../data_cache')
    path_file_train: Optional[str] = field(default='../data/data_neg_with_cpr/train_data.csv')
    path_file_val: Optional[str] = field(default='../data/data_neg_with_cpr/validate_data.csv')
    
    max_num_train: Optional[int] = field(default=0)
    max_num_val: Optional[int] = field(default=0)
    max_seq_length: Optional[int] = field(default=108)
    pad_to_max_length: bool = field(default=False)

    def __post_init__(self):
        if self.dataset_name is None and self.path_file_train is None and self.path_file_train is None:
            raise ValueError('Need either a dataset name or a training/val file.')
        else:
            if self.path_file_train:
                extension = self.path_file_train.split('.')[-1]
                assert extension in ['csv', 'json', 'txt'], '`path_file_train` should be a csv, a json or a txt file.'

#----------------------------------------
# 参数：自定义
@dataclass
class OurTrainingArguments(TrainingArguments):

    # 路径
    output_dir: Optional[str] = field(default='../result/baseline')
    resume_from_checkpoint: Optional[str] = field(default=None)

    do_fgm: int = field(default=1)
    do_pgd: int = field(default=0)
    evaluation_strategy: str = field(default='steps')
    greater_is_better: Optional[bool] = field(default=True)
    load_best_model_at_end: bool = field(default=True)
    num_train_epochs: int = field(default=7)
    per_device_train_batch_size: int = field(default=96)
    per_device_eval_batch_size: int = field(default=96)
    learning_rate: float = field(default=3e-5)

    do_eval: int = field(default=1)
    do_train: int = field(default=1)
    eval_steps: int = field(default=100)
    save_steps: int = field(default=100)
    log_steps: int = field(default=100)

    @cached_property
    @torch_required
    def _setup_devices(self) -> "torch.device":
        
        if self.no_cuda:
            device = torch.device('cpu')
            self._n_gpu = 0
        elif self.local_rank == -1:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            self._n_gpu = torch.cuda.device_count()
        else:
            torch.distributed.init_process_group(backend='nccl')
            device = torch.device('cuda', self.local_rank)
            self._n_gpu = 1

        if device.type == 'cuda':
            torch.cuda.set_device(device)

        return device
