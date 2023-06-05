export CUDA_VISIBLE_DEVICES=6,7
python -m torch.distributed.run \
--nproc_per_node=2 --master_port 16853 tools/train_ltbgnn_only.py \
