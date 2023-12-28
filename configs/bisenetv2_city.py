
## bisenetv2
cfg = dict(
    model_type='bisenetv2',
    n_cats=18,
    num_aux_heads=4,
    lr_start=5e-3,
    weight_decay=5e-4,
    warmup_iters=0,
    max_iter=130000,
    dataset='CityScapes',
    im_root='/home/cxh/datasets/',
    train_im_anns='datasets/Cityscapes/train.txt',
    val_im_anns='datasets/Cityscapes/val.txt',  ## for precise bn
    scales=[0.5, 1., 1.5],
    cropsize=[512, 1024],
    eval_crop=[512, 1024], #[1024, 1024],
    eval_scales=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
    ims_per_gpu=10,
    eval_ims_per_gpu=1,
    use_fp16=True,
    use_sync_bn=False, ## default Yes.
    respth='res/',
)
