export CUDA_VISIBLE_DEVICES=0,1
python -m torch.distributed.run \
--nproc_per_node=2 --master_port 16855 tools/train_ltbgnn_all_datasets.py --config configs/ltbgnn_7_datasets_2.json --port 16855 \
