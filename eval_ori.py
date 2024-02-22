/cpfs01/projects-HDD/pujianxiangmuzu_HDD/mr_22210240239/Mul-Datasets-Semantic-Segmentation/./lib/module/util.py:50: UserWarning: Using conv type 3x3: <class 'torch.nn.modules.conv.Conv2d'>
  warnings.warn(f'Using conv type {k}x{k}: {conv_class}')
/cpfs01/projects-HDD/pujianxiangmuzu_HDD/mr_22210240239/Mul-Datasets-Semantic-Segmentation/./lib/module/util.py:127: UserWarning: Using skip connections: True
  warnings.warn(f'Using skip connections: {self.use_skip}', UserWarning)
/cpfs01/projects-HDD/pujianxiangmuzu_HDD/mr_22210240239/Mul-Datasets-Semantic-Segmentation/./lib/module/util.py:50: UserWarning: Using conv type 1x1: <class 'torch.nn.modules.conv.Conv2d'>
  warnings.warn(f'Using conv type {k}x{k}: {conv_class}')
load pretrained weights from res/celoss/pretrain_model_60000.pth
Using CVCUDA as preprocessor.
tensor([0.9691, 0.7752, 0.8988, 0.5033, 0.4784, 0.4943, 0.5442, 0.6632, 0.9000,
        0.5807, 0.9300, 0.7067, 0.4994, 0.9137, 0.6615, 0.7492, 0.6054, 0.4582,
        0.6588], device='cuda:0')
Traceback (most recent call last):
  File "tools/eval_snp_cvcuda.py", line 409, in <module>
    main()
  File "tools/eval_snp_cvcuda.py", line 384, in main
    train()
  File "tools/eval_snp_cvcuda.py", line 354, in train
    heads, mious = eval_model_cvcuda(configer, net, device_id, cuda_ctx)
  File "/usr/local/lib/python3.8/dist-packages/torch/utils/_contextlib.py", line 115, in decorate_context
    return func(*args, **kwargs)
  File "/cpfs01/projects-HDD/pujianxiangmuzu_HDD/mr_22210240239/Mul-Datasets-Semantic-Segmentation/./evaluate.py", line 1088, in eval_model_cvcuda
    del dls
UnboundLocalError: local variable 'dls' referenced before assignment
-------------------------------------------------------------------
PyCUDA ERROR: The context stack was not empty upon module cleanup.
-------------------------------------------------------------------
A context was still active when the context stack was being
cleaned up. At this point in our execution, CUDA may already
have been deinitialized, so there is no way we can finish
cleanly. The program will be aborted now.
Use Context.pop() to avoid this problem.
-------------------------------------------------------------------
