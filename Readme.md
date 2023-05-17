# “阿里灵杰”问天引擎电商搜索算法赛

团队：gpushare.com-比赛

最终排名： 

- 初赛：11 / 2771 （0.337）；

- 复赛：5 / 2771 （0.3789）；

- 决赛：4 / 2771 （**季军 + 技术创新奖**）；

答辩 PPT：[PPT](答辩PPT_gpushare.com-比赛.pptx)

知乎 Link：[Link](https://zhuanlan.zhihu.com/p/535910140)

这个仓库是我们团队在 [“阿里灵杰”问天引擎电商搜索算法赛](https://tianchi.aliyun.com/specials/promotion/opensearch) 比赛中取得季军结果的相关文件及代码开源，包含我们完成本次比赛的完整思路和复现代码。

## 方案简介

### 召回阶段

召回阶段主要分为两步，分别是**领域数据后训练**和**任务数据微调**。

#### 领域数据后训练

##### 数据处理

使用原始比赛数据和官方提供的CPR数据中的train.query、test.query、corpus数据，经过中文简写英文小写转换（follow ernie预训练时的数据处理方式）作为训练和验证数据。

##### 训练

模型使用chinese-roberta-wwm-ext；后训练方式为N-gram mlm。

#### 召回任务微调

##### 数据处理

数据来源为原始比赛数据和官方提供的CPR数据。使用train.query和corpus中对应的doc构建query和正样本doc，并且随机采样（借鉴DPR中的部分负样本构建）corpus中不包含对应query的doc，作为负样本doc。

##### 训练

follow有监督simcse的训练。对于每个query，正样本为对应的doc，负样本为in-batch其它query的对应doc和in-batch随机采样的所有doc，将query和正负doc分别输入模型，使用模型最后一层隐状态的平均值作为对应表示，然后将query表示分别和正负doc表示计算余弦相似度后，计算infoNCE loss。

并且训练时使用FGM对抗训练。

### 精排阶段

精排阶段使用召回阶段后训练的模型，然后使用精排任务微调。

#### 精排任务微调

##### 数据处理

数据来源为原始比赛数据。使用初赛训练好的召回模型，为每个query模拟召回top-k个doc。对于每个query，其正样本为corpus中对应的doc，负样本为模拟召回的top-k中排除正样本后的doc。

##### 训练

follow百度开源的neural_search中的ranking模块。每条训练样本为query，正样本doc，负样本doc，分别将query和正负样本doc凭借后输入模型，取[CLS]向量经过映射后降为1维，然后过sigmoid激活函数作为排序得分，然后使用margin ranking loss计算损失。

并且为了使得训练时降为1维的logits经过sigmoid激活函数后更有区分度，我们将正例pair的logits和负例pair的logits取平均值获得mean logits，然后使用正例pair的logits - mean logits获得新的正例pair的logits，负例pair的logits - mean logits获得新的负例pair的logits。

同时对于margin ranking loss，我们提出了动态margin的策略，即使用模拟召回时负样本对应的召回距离控制margin大小，margin的大小和召回距离的平方成正比。通过这种方式，不相关的负样本由于召回距离大，所以margin更大，使得召回的所有的负样本都能被更全面的优化。

最后使用了R-Drop的方式优化训练过程。

##### 验证

对于验证，考虑到线下验证与线上测试的一致性，线下验证时直接采用和线上相同的排序方式进行验证。每条验证样本为query，正样本（或负样本）doc，同一query的top-k个负样本doc + 1个正样本样本doc组成一个验证集batch，每次计算batch中所有样本对的排序得分，然后进行排序，找到正样本所在的排名计算MRR@10。同时考虑到线上测试时可能无法召回正样本doc，所以我们计算了验证集的召回top-k+1中包含正样本的概率，用这个概率乘MRR@10就可以大致模拟线上的测试集得分。

##### 确定召回top-k以及提交的Rerank_size

首先，通过测试，我们可知base模型的测试Rerank_size大概在35-40之间，所以我们召回的top-k为35，但在提交时，却不是越大越好，而是在中间的某个值时达到最优，经过尝试，我们确定提交的Rerank_size为24。

## 代码说明

### 1. 代码文件结构

```
.
├── ali_1
│   ├── data
│   │   ├── data_neg
│   │   ├── data_neg_with_cpr
│   │   ├── original_data
│   │   │   ├── corpus.tsv
│   │   │   ├── dev.query.txt
│   │   │   ├── qrels.train.tsv
│   │   │   └── train.query.txt
│   │   ├── original_data_cpr
│   │   │   ├── corpus.tsv
│   │   │   ├── dev.query.txt
│   │   │   ├── qrels.dev.tsv
│   │   │   ├── qrels.train.tsv
│   │   │   └── train.query.txt
│   │   ├── processed_data
│   │   └── processed_data_cpr
│   ├── model
│   │   ├── added_tokens.json
│   │   ├── config.json
│   │   ├── gitattributes
│   │   ├── pytorch_model.bin
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer_config.json
│   │   ├── tokenizer.json
│   │   └── vocab.txt
│   ├── posttrain_data
│   ├── posttrain_result
│   ├── requirements.txt
│   ├── result
│   └── src
│       ├── 1.data_post_processed.py
│       ├── 2.construct_dataset.py
│       ├── 3.model_train.py
│       ├── 4.get_data_embeddings.py
│       ├── collate_data.py
│       ├── config.py
│       ├── init.py
│       ├── main_recall.sh
│       ├── model_cl.py
│       ├── posttrain_data_post_processed.py
│       ├── posttrain.py
│       ├── shell_1.run_data_process.sh
│       ├── shell_2.run_posttrain.sh
│       ├── shell_3.run_train.sh
│       ├── shell_4.run_get_embeding.sh
│       ├── trainer_cl.py
│       ├── utils.py
│       └── zh_wiki.py
└── ali1_ranking
    ├── config
    │   └── posttrain_roberta_wwm_ext
    ├── convert_bert_pytorch_checkpoint_to_original_tf.py
    ├── data
    ├── get_test_embedding.py
    ├── requirements.txt
    ├── result
    │   └── tianchi
    ├── run.sh
    ├── src
    │   ├── models
    │   │   ├── adversarial.py
    │   │   ├── bert.py
    │   │   ├── ernie.py
    │   │   ├── module.py
    │   │   ├── roberta.py
    │   │   └── transformer.py
    │   ├── optim
    │   │   ├── adam.py
    │   │   ├── optimizer.py
    │   │   └── scheduler.py
    │   ├── preprocess
    │   │   ├── dataloader.py
    │   │   └── tokenizer.py
    │   ├── train
    │   │   └── trainer.py
    │   └── utils
    │       ├── args_util.py
    │       ├── ckpt_util.py
    │       ├── config_util.py
    │       ├── debug_util.py
    │       ├── mlm_util.py
    │       ├── random_util.py
    │       ├── tensor_util.py
    │       ├── tokenizer.py
    │       └── visual_util.py
    ├── submit
    ├── tianchi_args.py
    ├── tianchi_dataloader.py
    ├── tianchi_dataprocess.py
    ├── tianchi_init.py
    ├── tianchi_r_drop.py
    ├── tianchi_r_drop.sh
    ├── tianchi_wrapper.py
    └── tianchi_wrapper.sh
```



### 2. 说明

#### 2.1 召回

##### 数据简介 (data)

```
original_data			# 原始数据文件夹
	corpus.tsv			# 原始数据文档集
	dev.query.txt		# 原始数据验证查询集
	qrels.train.tsv		# 原始数据训练对照关系集
	train.query.txt		# 原始数据训练查询集

original_data_cpr		# CPR 数据文件夹
	corpus.tsv			# CPR 数据文档集
	dev.query.txt		# CPR 数据验证查询集
	qrels.dev.tsv		# CPR 数据验证对照关系集
	qrels.train.tsv		# CPR 数据训练对照关系集
	train.query.txt		# CPR 数据训练查询集

processed_data			# 存放预处理后的原始数据

processed_data_cpr		# 存放预处理后的 CPR 数据

data_neg_with_cpr		# 存放混合并采样负样本后的数据
```

##### 后训练数据

```
posttrain_data		# 保存后训练输入的数据
```

##### 模型

```
model				# 存放预训练模型
posttrain_result	# 存放后训练模型
result				# 存放召回模型
```

##### 代码简介

```
1.data_post_processed.py			# 数据处理
2.construct_dataset.py				# 后训练
3.model_train.py					# 召回模型训练
4.get_data_embeddings.py			# 获取 embedding

zh_wiki.py							# 繁体字与简体字对照关系
utils.py							# 处理小写和简体字

posstrain.py						# 后训练细节
posstrain_data_post_processed.py	# 混合原始数据与 CPR 数据

collate_data.py						# 召回模型所需构造 batch
config.py							# 召回模型所需超参数
init.py								# 召回模型所需库
model_cl.py							# 召回模型细节
trainer_cl.py						# 召回模型训练器

shell_1.run_data_process.sh			# 数据处理
shell_2.run_posttrain.sh			# 后训练
shell_3.run_train.sh				# 召回模型训练
shell_4.run_get_embedding.sh		# 获取 embedding
main_recall.sh						# 依次运行上述脚本
```

#### 2.2 精排

- 后训练模型：config

- 数据：data

- 模型结果：result

- 源代码：src

- 提交结果：submit

### 3. 运行环境

#### 3.1 召回

##### 后训练

```
cuda==10.1
python==3.8.12
GPU型号==GeForce GTX 1080Ti*4张
apex==0.9.10dev
datasets==2.0.0
deepspeed==0.6.0
fairscale==0.4.6
filelock==3.6.0
numpy==1.22.3
packaging==21.3
scikit_learn==1.0.2
torch==1.11.0
tqdm==4.63.0
transformers==4.17.0
```

tips：

1. pytorch请安装GPU版本。

2. 如果apex安装失败会导致运行报错，因为pip直接安装的apex和NVIDIA的apex库不是同一个库，我们需要的是NVIDIA的apex库。
3. 解决方法：安装NVIDIA的apex库，命令如下：

```
git clone https://github.com/NVIDIA/apex
cd apex
pip3 install -v --no-cache-dir ./
```

##### 微调

```
cuda==11.3
python==3.8.12
GPU型号==GeForce RTX 3090*8张
apex==0.9.10dev
datasets==2.0.0
deepspeed==0.6.0
fairscale==0.4.6
filelock==3.6.0
numpy==1.22.3
packaging==21.3
scikit_learn==1.0.2
torch==1.11.0
tqdm==4.63.0
transformers==4.17.0
pandas==1.4.1
faiss-gpu==1.7.2
```

tips：同上。

#### 3.2 精排

##### 微调

```
cuda==10.1
python==3.7.0
GPU型号==GeForce RTX 2080Ti*1张
tensorflow-gpu==1.13.1
keras==2.2.4
torch==1.8.1
numpy==1.21.6
pandas==0.23.4
faiss-gpu==1.7.2
tqdm==4.26.0
```

tips：同上。



### 4. 运行说明

#### 4.1 召回
全流程一键运行
```
conda activate recall_env
cd ./ali_1_final/ali_1/src/
bash main_recall.sh
```

tips:

#### 4.2 精排
全流程一键运行
```
conda activate rank_env
cd ./ali_1_final/ali_1_ranking/
bash run.sh
```

tips：

1. run.sh包含数据处理、模型训练、模型转换为提交格式、打包流程。
2. 最终输出的提交文件位于**submit/foo.tar.gz**。

3. 如果要复现之前的结果，请按照**4.1及4.2部分**分别在**不同机器**上按顺序执行（每一台机器执行完毕后请复制执行后生成的文件到新机器）。