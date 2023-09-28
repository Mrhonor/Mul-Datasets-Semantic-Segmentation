export CUDA_VISIBLE_DEVICES=4
python -m torch.distributed.run \
--nproc_per_node=3 --master_port 16856 tools/train_clip_5datasets.py --config configs/clip_7_datasets.json --port 16856 \
