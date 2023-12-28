export CUDA_VISIBLE_DEVICES=2,3
python -m torch.distributed.run \
--nproc_per_node=2 --master_port 15643 tools/train_ltbgnn_all_datasets_segonly.py --config configs/ltbgnn_5_datasets_ade.json --port 15643 \
