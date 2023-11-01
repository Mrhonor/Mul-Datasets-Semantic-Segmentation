export CUDA_VISIBLE_DEVICES=4,5,6,7
python -m torch.distributed.run \
--nproc_per_node=4 --master_port 10956 tools/train_temp.py --config configs/ltbgnn_7_datasets.json --port 10956 \
