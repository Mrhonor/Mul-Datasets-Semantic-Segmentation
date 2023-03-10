export CUDA_VISIBLE_DEVICES=3,7
python -m torch.distributed.run \
--nproc_per_node=2 --master_port 16854 tools/train_celoss_3datasets.py \
