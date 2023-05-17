#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3

python posttrain_data_post_processed.py


# 初始化模型路径
model_name_or_path='../model'
# 后训练模型保存路径
output_dir='../posttrain_result/best_model'
# 后训练训练数据路径
train_data_file='../posttrain_data/posttrain_data.txt'

python posttrain.py --mlm ngram --model_type bert --model_name_or_path=$model_name_or_path --output_dir=$output_dir --do_train --train_data_file=$train_data_file --per_device_train_batch_size 48 --per_device_eval_batch_size 48 --logging_strategy "epoch" --save_strategy "epoch" --num_train_epochs 10 --learning_rate 5e-05 && echo done

# 最佳模型：../posttrain_result/best_model/checkpoint-113960
         
                



