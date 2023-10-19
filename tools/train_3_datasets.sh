export CUDA_VISIBLE_DEVICES=5,7
python -m torch.distributed.run \
--nproc_per_node=2 --master_port 16910 tools/train_ltbgnn_all_datasets.py --config configs/ltbgnn_3_datasets.json --port 16910 \
