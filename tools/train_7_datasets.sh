export CUDA_VISIBLE_DEVICES=0,1
# python tools/train_ltbgnn_all_datasets.py --config configs/ltbgnn_7_datasets.json
python3.8 -m torch.distributed.run \
--nproc_per_node=2 --master_port 32956 tools/train_ltbgnn_all_datasets_cvcuda_uot_tg.py --config configs/ltbgnn_7_datasets_snp_train_tg_dist.json --port 32956 \
