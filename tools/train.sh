export CUDA_VISIBLE_DEVICES=0,3,5
python -m torch.distributed.launch \
--nproc_per_node=3 --master_port 12345 train_amp.py \
--config /root/autodl-tmp/project/BiSeNet/configs/bisenetv2_city.py \
--finetune_from /root/autodl-tmp/project/BiSeNet/res/model_20000.pth