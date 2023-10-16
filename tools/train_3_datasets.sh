export CUDA_VISIBLE_DEVICES=4,5,6,7
python -m torch.distributed.run \
--nproc_per_node=4 --master_port 14856 tools/train_ltbgnn_all_datasets.py --config configs/ltbgnn_3_datasets.json --port 14856 \
