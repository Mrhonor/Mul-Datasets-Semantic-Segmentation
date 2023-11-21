export CUDA_VISIBLE_DEVICES=4,5,7
python -m torch.distributed.run \
--nproc_per_node=3 --master_port 32956 tools/train_ltbgnn_all_datasets.py --config configs/ltbgnn_7_datasets.json --port 32956 \
