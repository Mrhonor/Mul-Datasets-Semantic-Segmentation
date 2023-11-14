export CUDA_VISIBLE_DEVICES=5,7
python -m torch.distributed.run \
--nproc_per_node=2 --master_port 55956 tools/train_ltbgnn_all_datasets.py --config configs/ltbgnn_7_datasets_snp.json --port 55956 \
