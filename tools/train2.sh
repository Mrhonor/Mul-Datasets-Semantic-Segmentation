export CUDA_VISIBLE_DEVICES=4,5,6
python -m torch.distributed.run \
--nproc_per_node=3 --master_port 16854 tools/train_celoss_3datasets.py \
