CUDA_VISIBLE_DEVICES=3 python tianchi_r_drop.py --task=final_roberta-wo-test-emb_rerank35_posttrain_roberta_wwm_ext_seed1000_epoch1_lr1e-5_margin0.4_scale-sim_dynamic-margin_rdrop16 --model_type=posttrain_roberta_wwm_ext --lr=1e-5 --epochs=1 --batch_size=12 --save_steps=10000 --margin=0.4