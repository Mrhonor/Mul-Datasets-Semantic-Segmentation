export CUDA_VISIBLE_DEVICES=4,5
python -m torch.distributed.run \
--nproc_per_node=2 --master_port 16855 tools/train_amp_contrast_single.py \
