export CUDA_VISIBLE_DEVICES=0,1
python -m torch.distributed.run \
--nproc_per_node=2 --master_port 14856 tools/train_ltbgnn_all_datasets.py --config configs/ltbgnn_3_datasets.json --port 14856 \
