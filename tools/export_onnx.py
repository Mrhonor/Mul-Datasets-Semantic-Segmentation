import argparse
import os.path as osp
import sys
from turtle import forward
sys.path.insert(0, '.')

import torch

from lib.models import model_factory
from configs import set_cfg_from_file

torch.set_grad_enabled(False)


parse = argparse.ArgumentParser()
parse.add_argument('--config', dest='config', type=str,
        default='configs/bisenetv2.py',)
parse.add_argument('--weight-path', dest='weight_pth', type=str,
        default='model_final.pth')
parse.add_argument('--outpath', dest='out_pth', type=str,
        default='model.onnx')
parse.add_argument('--aux-mode', dest='aux_mode', type=str,
        default='pred')
args = parse.parse_args()


cfg = set_cfg_from_file(args.config)
if cfg.use_sync_bn: cfg.use_sync_bn = False


class E2EModel(torch.nn.Module):
        
    def __init__(self, cfg, args) -> None:
        super().__init__()
        
        self.mean = torch.tensor([0.3257, 0.3690, 0.3223])[:, None, None]
        self.std = torch.tensor([0.2112, 0.2148, 0.2115])[:, None, None]
        
        self.net = model_factory[cfg.model_type](cfg.n_cats, aux_mode="pred")
        self.net.load_state_dict(torch.load(args.weight_pth, map_location='cpu'), strict=False)
        
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = x.div_(255.)
        x = x.sub_(self.mean).div_(self.std).clone()
        out = self.net(x)
        return out

# net = model_factory[cfg.model_type](cfg.n_cats, aux_mode=args.aux_mode)
# net.load_state_dict(torch.load(args.weight_pth, map_location='cpu'), strict=False)
# net.eval()
net = E2EModel(cfg, args)


#  dummy_input = torch.randn(1, 3, *cfg.crop_size)
dummy_input = torch.randn(1, 1024, 2048, 3)
input_names = ['input_image']
output_names = ['preds',]

torch.onnx.export(net, dummy_input, args.out_pth,
    input_names=input_names, output_names=output_names,
    verbose=False, opset_version=11)

