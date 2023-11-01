export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m torch.distributed.run \
--nproc_per_node=8 --master_port 13910 tools/train_ltbgnn_all_datasets.py --config configs/ltbgnn_3_datasets.json --port 13910 \
