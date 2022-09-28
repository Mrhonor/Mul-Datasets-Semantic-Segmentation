export CUDA_VISIBLE_DEVICES=5,6,7
python -m torch.distributed.run \
--nproc_per_node=3 --master_port 16748 tools/train_amp_contrast.py \
