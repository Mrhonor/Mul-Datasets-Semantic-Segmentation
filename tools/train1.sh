export CUDA_VISIBLE_DEVICES=0,3
python -m torch.distributed.launch \
--nproc_per_node=2 --master_port 12345 train_amp_city.py \
--config /root/autodl-tmp/project/BiSeNet/configs/bisenetv2_city.py \
--finetune_from /root/autodl-tmp/project/BiSeNet/pth/backbone_v2.pth