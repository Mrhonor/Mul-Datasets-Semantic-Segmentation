
## bisenetv2
cfg = dict(
    model_type='bisenetv2',
    n_cats=38,
    num_aux_heads=4,
    lr_start=5e-3,
    weight_decay=5e-4,
    warmup_iters=1000,
    max_iter=100000,
    dataset='A2D2Data',
    im_root='/nfs/data/A2D2/split',
    train_im_anns='./datasets/cityscapes/train.txt',
    val_im_anns='./datasets/cityscapes/val.txt',
    scales=[0.25, 2.],
    cropsize=[512, 1024],
    eval_crop=[1024, 1024],
    eval_scales=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
    ims_per_gpu=6,
    eval_ims_per_gpu=2,
    use_fp16=True,
    use_sync_bn=True,
    respth='./res',
)
