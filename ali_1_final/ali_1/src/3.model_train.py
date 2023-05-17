# 导入库
from init import *
from config import *

from model_cl import BertForCL
from collate_data import PadDataCollator
#----------------------------------------
# 主函数
def main():

    # 参数
    # 解析输入参数
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, OurTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        args_model, args_data, args_training = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        args_model, args_data, args_training = parser.parse_args_into_dataclasses()
        
    #----------
    # 局部所需参数
    # 模型参数
    cache_dir = args_model.cache_dir
    config_name = args_model.config_name
    model_name_or_path = args_model.model_name_or_path
    model_revision = args_model.model_revision
    model_type = args_model.model_type
    tokenizer_name = args_model.tokenizer_name
    use_auth_token = args_model.use_auth_token
    use_fast_tokenizer = args_model.use_fast_tokenizer

    pooler_type = args_model.pooler_type
    temperature = args_model.temperature
    
    # 数据参数
    dir_data = args_data.dir_data
    dir_data_cache = args_data.dir_data_cache
    max_seq_length = args_data.max_seq_length
    overwrite_cache = args_data.overwrite_cache
    pad_to_max_length = args_data.pad_to_max_length
    path_file_train = args_data.path_file_train
    path_file_val = args_data.path_file_val
    preprocessing_num_workers = args_data.preprocessing_num_workers
    
    max_num_train = args_data.max_num_train
    max_num_val = args_data.max_num_val
        
    # 训练参数
    do_eval = args_training.do_eval
    do_train = args_training.do_train
    do_fgm = args_training.do_fgm
    local_rank = args_training.local_rank
    main_process_first = args_training.main_process_first
    output_dir = args_training.output_dir
    resume_from_checkpoint = args_training.resume_from_checkpoint
    seed = args_training.seed
    
    # 补充参数
    flag_master = is_main_process(local_rank)
    args_training.flag_master = flag_master
    set_seed(seed)
    
    #--------------------
    # 日志
    logging.basicConfig(
        format="%(asctime)s | %(name)s: %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO if flag_master else logging.WARN,
    )
    
    if flag_master:
        # 避免加载模型时的底层 logging.info() 显示
        transformers.utils.logging.set_verbosity_warning()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    
    #--------------------
    # 运行环境
    logger.info(f'torch: {version_torch} | cuda available: {cuda_available}')
    logger.info(f'cuda: {version_cuda} | cudnn: {version_cudnn}')
    
    lst_name_arg = [
        'do_train', 'do_eval', 'do_fgm', 
        'n_gpu', 'local_rank', 'seed', 
        'num_train_epochs','learning_rate', 'weight_decay',
        'warmup_ratio', 'warmup_steps', 'greater_is_better', 
        'eval_steps', 'save_steps', 'log_steps', 
        'overwrite_output_dir', 'load_best_model_at_end', 'output_dir',
        'per_device_train_batch_size', 'temperature', 'resume_from_checkpoint',
        'per_device_eval_batch_size', 'fp16', 'max_seq_length', 
    ]
    str_args = ''
    for i_arg, name_arg in enumerate(lst_name_arg, 1):
        for args in [args_model, args_data, args_training]:
            if hasattr(args, name_arg):
                arg = getattr(args, name_arg)
                arg = int(arg) if type(arg) == bool else arg
                arg = str(arg)
                str_args += f'{name_arg:16s}: {arg:10s} | '
                if i_arg % 3 == 0:
                    logger.info(str_args)
                    str_args = ''
                break
    if len(str_args):
        logger.info(str_args)
        
    #--------------------
    # 加载数据集
    # 名字: 路径 组成字典
    data_files = {}
    if path_file_train:
        data_files['train'] = path_file_train
    if path_file_val:
        data_files['val'] = path_file_val
    extension = path_file_train.split('.')[-1]
    if extension == 'txt':
        extension = 'text'
        
    #----------
    # 部分数据
    dic_split = {}
    dic_split['train'] = f'train[:{max_num_train}]' if max_num_train else 'train[:]'
    dic_split['val'] = f'val[:{max_num_val}]' if max_num_val else 'val[:]'
    
    #----------
    # 加载数据集所需参数
    kwargs_dataset = {
        'path': extension,
        'data_files': data_files,
        'cache_dir': dir_data_cache,
        'split': dic_split
    }
    if extension == 'csv':
        kwargs_dataset['delimiter'] = '\t' if 'tsv' in path_file_train else ','
        
    # 加载数据集
    datasets = load_dataset(**kwargs_dataset)
    
    #--------------------
    # 加载预训练
    # 加载配置所需参数
    kwargs_config = {
        'cache_dir': cache_dir,
        'revision': model_revision,
        'use_auth_token': True if use_auth_token else None
    }
    
    # 加载配置
    if config_name:
        config = AutoConfig.from_pretrained(config_name, **kwargs_config)
        logger.info(f'init model config from: {config_name}')
    elif model_name_or_path:
        config = AutoConfig.from_pretrained(model_name_or_path, **kwargs_config)
        logger.info(f'init model config from: {model_name_or_path}')
    else:
        config = CONFIG_MAPPING[model_type]()
        logger.info(f'init model config from scratch')
        
    #----------
    # 加载 tokenizer 所需参数
    kwargs_tokenizer = {
        'cache_dir': cache_dir,
        'revision': model_revision,
        'use_auth_token': True if use_auth_token else None,
        'use_fast': use_fast_tokenizer
    }
    
    # 加载 tokenizer
    if tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **kwargs_tokenizer)
        logger.info(f'init model tokenizer from: {tokenizer_name}')
    elif model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **kwargs_tokenizer)
        logger.info(f'init model tokenizer from: {model_name_or_path}')
    else:
        raise ValueError('init model tokenizer from scratch [wrong], please use `--tokenizer_name`')

    #----------
    # 加载模型所需参数
    kwargs_model = {
        'pretrained_model_name_or_path': model_name_or_path,
        'config': config,
        'cache_dir': cache_dir,
        'from_tf': '.ckpt' in model_name_or_path,
        'revision': model_revision,
        'use_auth_token': True if use_auth_token else None,

        'pooler_num': 128,
        'pooler_type': pooler_type,
        'temp': temperature
    }
    
    # 加载模型
    if model_name_or_path:
        model = BertForCL.from_pretrained(**kwargs_model)
        logger.info(f'init model from: {model_name_or_path}')
    else:
        model = AutoModelForMaskedLM.from_config(config)
        raise NotImplementedError('init model from scratch')
    model.resize_token_embeddings(len(tokenizer))

    #----------
    # 加载训练集特征
    column_names = datasets["train"].column_names
    sent0_cname, sent1_cname, sent2_cname = column_names
    # column_names: List[str] | ex. ['query', 'doc', 'doc_neg']

    def mapDataset(batch_data):

        num_sample = len(batch_data[sent0_cname])

        #----------
        # 替换 None 为 ' '
        for i_sample in range(num_sample):
            if batch_data[sent0_cname][i_sample] is None:
                batch_data[sent0_cname][i_sample] = ' '
            if batch_data[sent1_cname][i_sample] is None:
                batch_data[sent1_cname][i_sample] = ' '
            if batch_data[sent2_cname][i_sample] is None:
                batch_data[sent2_cname][i_sample] = ' '

        #----------
        sentences = batch_data[sent0_cname] + batch_data[sent1_cname] + batch_data[sent2_cname]
        # sentences: len = 2 * num_sample

        # 句子转为 id 所需参数
        kwargs_sent_tokenizer = {
            'max_length': max_seq_length,
            'padding': 'max_length' if pad_to_max_length else False,
            'truncation': True,
        }
        sent_features = tokenizer(sentences, **kwargs_sent_tokenizer)
        # sent_features key: input_ids, token_type_ids, attention_mask
        # sent_features.input_ids: List[List[int]] | 第一层是 2 * num_sample 个，第二层是“该句子的长度”个

        features = {}
        for key in sent_features:
            features[key] = [
                [sent_features[key][i], sent_features[key][i + num_sample], sent_features[key][i + num_sample * 2]]
                    for i in range(num_sample)
            ]

        return features

    # 处理数据集所需参数
    kwargs_map_dataset = {
        'function': mapDataset,
        'batched': True,
        'load_from_cache_file': not(overwrite_cache),
        'num_proc': preprocessing_num_workers,
        'remove_columns': column_names
    }

    if do_train:
        dataset_train = datasets['train']
        dataset_train = dataset_train.map(**kwargs_map_dataset)
    if do_eval:
        dataset_val = datasets['val']
        with main_process_first():
            dataset_val = dataset_val.map(**kwargs_map_dataset)

    #----------
    # 处理数据
    data_collator = PadDataCollator(tokenizer)

    #----------
    # 训练器所需参数
    from trainer_cl import CLTrainer
    kwargs_trainer = {
        'args_training': args_training,
        'model': model,
        'tokenizer': tokenizer,
        'dataset_train': dataset_train,
        'dataset_val': dataset_val,
        'data_collator': data_collator,
        'do_fgm': do_fgm
    }
    trainer = CLTrainer(**kwargs_trainer)
    trainer.args_model = args_model

    #----------
    if do_train:
        checkpoint = resume_from_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()


    # from trainers import CLTrainer
    # trainer = CLTrainer(
    #     model=model,
    #     args=args_training,
    #     train_dataset=dataset_train if do_train else None,
    #     eval_dataset=dataset_val if do_eval else None,
    #     tokenizer=tokenizer,
    #     data_collator=data_collator,
    #     do_fgm=do_fgm,
    # )
    # trainer.model_args = args_model
    #
    # # ----------
    # if do_train:
    #     checkpoint = resume_from_checkpoint
    #     train_result = trainer.train(resume_from_checkpoint=checkpoint)
    #     trainer.save_model()
    #
    #     if trainer.is_world_process_zero():
    #         path_result_train = sep.join([output_dir, 'result_train.txt'])
    #         with open(path_result_train, 'w') as f:
    #             logger.info('--- train result: ---')
    #             for key, value in sorted(train_result.metrics.items()):
    #                 logger.info(f'  {key} = {value}')
    #                 f.write(f'  {key} = {value}')
    #
    #         path_trainer_state = sep.join([output_dir, 'trainer_state.json'])
    #         trainer.state.save_to_json(path_trainer_state)

    return

#----------------------------------------
# 入口
if __name__ == '__main__':
    
    logger = logging.getLogger(__name__)
    main()
    