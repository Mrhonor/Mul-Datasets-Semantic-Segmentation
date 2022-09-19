export CUDA_VISIBLE_DEVICES=1,2,3
python -m torch.distributed.launch \
--nproc_per_node=3 --master_port 16745 tools/train_amp_contrast.py \
