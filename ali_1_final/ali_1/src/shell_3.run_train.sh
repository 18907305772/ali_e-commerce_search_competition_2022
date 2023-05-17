export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPU=8
PORT_ID=$(expr $RANDOM + 1000)
export OMP_NUM_THREADS=16
python -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port $PORT_ID 3.model_train.py