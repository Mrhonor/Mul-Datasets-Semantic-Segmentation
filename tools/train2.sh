export CUDA_VISIBLE_DEVICES=5,6
python -m torch.distributed.run \
--nproc_per_node=2 --master_port 16854 tools/train_ltgnn_only.py \
