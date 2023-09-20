export CUDA_VISIBLE_DEVICES=0,1
python -m torch.distributed.run \
--nproc_per_node=2 --master_port 16854 tools/train_ltbgnn_all_datasets.py \
