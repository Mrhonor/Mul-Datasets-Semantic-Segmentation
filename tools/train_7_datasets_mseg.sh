export CUDA_VISIBLE_DEVICES=1,2
python -m torch.distributed.run \
--nproc_per_node=2 --master_port 25956 tools/train_ltbgnn_all_datasets.py --config configs/ltbgnn_7_datasets_mseg.json --port 25956 \
