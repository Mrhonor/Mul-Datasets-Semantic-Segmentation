export CUDA_VISIBLE_DEVICES=1,2
python -m torch.distributed.run \
--nproc_per_node=2 --master_port 16853 tools/train_clip_3datasets.py \