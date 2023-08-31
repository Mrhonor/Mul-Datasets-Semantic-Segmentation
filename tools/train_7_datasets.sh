export CUDA_VISIBLE_DEVICES=0,1,2
python -m torch.distributed.run \
--nproc_per_node=3 --master_port 16854 tools/train_ltbgnn_all_datasets.py --config configs/ltbgnn_7_datasets.json --port 16854 \
