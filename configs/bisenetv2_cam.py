
## bisenetv2
cfg = dict(
    model_type='bisenetv2',
    n_cats=12,
    num_aux_heads=4,
    lr_start=2.5e-3, # 5e-3
    weight_decay=5e-4,
    warmup_iters=1000,
    max_iter=20,
    last_iter=1000,
    dataset='CamVid',
    im_root='/home/cxh/datasets/CamVid',
    train_im_anns='datasets/CamVid/train.txt',
    val_im_anns='datasets/CamVid/test.txt',
    scales=[0.5,1,2], #[0.25, 2.],
    cropsize=[512, 1024],
    eval_crop=[512, 1024],
    eval_scales=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
    ims_per_gpu=2,
    eval_ims_per_gpu=4,
    use_fp16=True,
    use_sync_bn=False, # 单卡训练，多卡改为True
    respth='res',
)

