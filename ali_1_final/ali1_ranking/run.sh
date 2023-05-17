#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python convert_bert_pytorch_checkpoint_to_original_tf.py --model_name="../ali_1/posttrain_result/best_model/checkpoint-113960" --pytorch_model_path="../ali_1/posttrain_result/best_model/checkpoint-113960/pytorch_model.bin" --tf_cache_dir="../ali_1/posttrain_result/best_model/checkpoint-113960"
cp -r "../ali_1/posttrain_result/best_model/checkpoint-113960/*" "config/posttrain_roberta_wwm_ext/"
python tianchi_dataprocess.py
python get_test_embedding.py
python tianchi_r_drop.py --task=final_result --model_type=posttrain_roberta_wwm_ext --lr=1e-5 --epochs=1 --batch_size=12 --save_steps=10000 --margin=0.4
python tianchi_wrapper.py --model_type=posttrain_roberta_wwm_ext --checkpoint=result/tianchi/final_result/model.ckpt-70000
printf "24"> submit/rerank_size
tar zcvf submit/foo.tar.gz submit/doc_embedding submit/query_embedding submit/model submit/rerank_size