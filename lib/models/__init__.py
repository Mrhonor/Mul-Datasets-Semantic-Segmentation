

from .bisenetv1 import BiSeNetV1
from .bisenetv2 import BiSeNetV2
from .bisenetv1_swin import BiSeNetV1_Swin
from .bisenetv2_contrast import BiSeNetV2_Contrast, BiSeNetV2_Contrast_Teacher
from .bisenetv2_contrast_wn import BiSeNetV2_Contrast_WN
from .bisenetv2_contrast_bn import BiSeNetV2_Contrast_BN
from .HRNetv2 import HRNet_W48_CONTRAST, HRNet_W48, HRNet_W48_CLIP, HRNet_W48_GNN
from .graph_attention_network import GraphAttentionNetwork

model_factory = {
    'bisenetv1': BiSeNetV1,
    'bisenetv2': BiSeNetV2,
    'bisenetv1_swin': BiSeNetV1_Swin,
    'bisenetv2_contrast': BiSeNetV2_Contrast,
    'bisenetv2_contrast_wn': BiSeNetV2_Contrast_WN,
    'bisenetv2_contrast_bn': BiSeNetV2_Contrast_BN,
    'bisenetv2_contrast_ema': BiSeNetV2_Contrast_Teacher,
    'hrnet_w48_contrast': HRNet_W48_CONTRAST,
    'hrnet_w48': HRNet_W48, 
    'hrnet_w48_clip': HRNet_W48_CLIP, 
    'graph_attention_network': GraphAttentionNetwork,
    'hrnet_w48_gnn': HRNet_W48_GNN
}
