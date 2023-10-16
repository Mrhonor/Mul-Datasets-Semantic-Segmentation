export CUDA_VISIBLE_DEVICES=0,1
python -m torch.distributed.run \
--nproc_per_node=2 --master_port 16853 tools/train_ltbgnn_all_datasets_segonly.py --config configs/ltbgnn_7_datasets_segonly.json --port 16853
