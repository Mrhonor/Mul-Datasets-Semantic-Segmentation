export CUDA_VISIBLE_DEVICES=2,3
python -m torch.distributed.run \
--nproc_per_node=2 --master_port 13956 tools/train_ltbgnn_all_datasets.py --config configs/ltbgnn_7_datasets.json --port 13956 \
