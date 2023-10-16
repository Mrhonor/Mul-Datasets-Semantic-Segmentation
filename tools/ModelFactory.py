import torch

from lib.models.bisenetv1 import BiSeNetV1
from lib.models.bisenetv2 import BiSeNetV2
from lib.models.bisenetv1_swin import BiSeNetV1_Swin
from lib.models.bisenetv2_contrast import BiSeNetV2_Contrast

# wait for imp
class ModelFactory():
    def Assemble(configer):
        logger = logging.getLogger()

        net = eval(configer.get('model_name'))(configer)

        if configer.get('train', 'finetune'):
            logger.info(f"load pretrained weights from {configer.get('train', 'finetune_from')}")
            net.load_state_dict(torch.load(configer.get('train', 'finetune_from'), map_location='cpu'), strict=False)

            
        if configer.get('use_sync_bn'): 
            net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
        net.cuda()
        net.train()
        
        if hasattr(net, 'get_params'):
            wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = net.get_params()
            #  wd_val = cfg.weight_decay
            wd_val = 0
            params_list = [
                {'params': wd_params, },
                {'params': nowd_params, 'weight_decay': wd_val},
                {'params': lr_mul_wd_params, 'lr': configer.get('lr', 'lr_start')},
                {'params': lr_mul_nowd_params, 'weight_decay': wd_val, 'lr': configer.get('lr', 'lr_start')},
            ]
        else:
            wd_params, non_wd_params = [], []
            for name, param in net.named_parameters():
                if param.requires_grad == False:
                    continue
                
                if param.dim() == 1:
                    non_wd_params.append(param)
                elif param.dim() == 2 or param.dim() == 4:
                    wd_params.append(param)
            params_list = [
                {'params': wd_params, },
                {'params': non_wd_params, 'weight_decay': 0},
            ]
        
        if configer.get('optim') == 'SGD':
            optim = torch.optim.SGD(
                params_list,
                lr=configer.get('lr', 'lr_start'),
                momentum=0.9,
                weight_decay=configer.get('lr', 'weight_decay'),
            )
        elif configer.get('optim') == 'AdamW':
            optim = torch.optim.AdamW(
                params_list,
                lr=configer.get('lr', 'lr_start'),
            )
        
        
        return net, optim