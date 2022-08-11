
import sys
sys.path.insert(0, '.')
import argparse
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import cv2
from time import time

import lib.transform_cv2 as T
from lib.models import model_factory
from configs import set_cfg_from_file

torch.set_grad_enabled(False)
np.random.seed(123)


# args
parse = argparse.ArgumentParser()
parse.add_argument('--config', dest='config', type=str, default='configs/bisenetv2_city.py',)
parse.add_argument('--weight-path', type=str, default='./res/model_final_1_1.pth',)
parse.add_argument('--img-path', dest='img_path', type=str, default='./Seq05VD_f02070.png',)
args = parse.parse_args()
cfg = set_cfg_from_file(args.config)


palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8)

class E2EModel(torch.nn.Module):
        
    def __init__(self, cfg, weight_path) -> None:
        super().__init__()
        
        self.mean = torch.tensor([0.3257, 0.3690, 0.3223])[:, None, None] #.cuda()
        self.std = torch.tensor([0.2112, 0.2148, 0.2115])[:, None, None] #.cuda()
        
        # self.net = model_factory[cfg.model_type](cfg.n_cats, aux_mode="pred")
        self.net = model_factory[cfg.model_type](38, 'pred', 2, 19)
        self.net.load_state_dict(torch.load(weight_path, map_location='cpu'), strict=False)
        self.net.eval()
        # self.net.cuda()
        
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = x.div_(255.)
        x = x.sub_(self.mean).div_(self.std).clone()
        out = self.net(x, 0)[0]
        return out
    
net1 = E2EModel(cfg, args.weight_path)
net2 = E2EModel(cfg, args.weight_path)
net2.net.segment.load_state_dict(torch.load('res/segment_precise_bn_40.pth', map_location='cpu'), strict=False)
net2.net.detail.load_state_dict(torch.load('res/detail_precise_bn.pth', map_location='cpu'), strict=False)

# # define model
# net = model_factory[cfg.model_type](cfg.n_cats, aux_mode='pred')
# net.load_state_dict(torch.load(args.weight_path, map_location='cpu'), strict=False)
# net.eval()
# net.cuda()

# # prepare data
# to_tensor = T.ToTensor(
#     mean=(0.3257, 0.3690, 0.3223), # city, rgb
#     std=(0.2112, 0.2148, 0.2115),
# )

print(args.img_path)
im = cv2.imread(args.img_path)[:, :, ::-1]

for i in range(1):
    t0 = time()
    # input_im = to_tensor(dict(im=im, lb=None))['im'].unsqueeze(0).cuda()
    input_im = cv2.resize(im, (960, 768))
    # input_im = im
    input_im = torch.tensor(input_im.astype(np.float32).copy()).unsqueeze(0) #.cuda()
    # print(input_im)
    # inference
    # out1 = net1(input_im).squeeze().detach().cpu().numpy()
    out2 = net2(input_im).squeeze().detach().cpu().numpy()
    # print(out.shape)
    # pred1 = palette[out1]
    pred2 = palette[out2]
    # print((time() - t0) * 1000)

# cv2.imwrite('./res1.jpg', pred1)
cv2.imwrite('./res2.jpg', pred2)
