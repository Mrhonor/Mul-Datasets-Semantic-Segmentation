export CUDA_VISIBLE_DEVICES=0
python evaluate.py \
--config configs/bisenetv2_city.py \
--weight-path res/same_affine_layer.pth