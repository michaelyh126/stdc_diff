# dataset settings
dataset_type = 'MoNuSegSAPRDataset'
val_dataset_type = 'MoNuSegDataset'

train_data_root = '/root/autodl-tmp/MoNuSeg'
test_data_root = '/root/autodl-tmp/MoNuSeg'

img_norm_cfg = dict(
    mean=[164.258, 114.090, 153.999],
    std=[59.028, 59.278, 47.782],
    to_rgb=True)

crop_size = (1000, 1000)
grid_shape = (4, 4)

# This starts from the baseline MoNuSegMix train augmentation. The dataset
# automatically increases probabilistic random-patch augmentation to compensate
# for untouched real-image patches inserted later for SAPR.
random_patch_pipeline = [
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomRotate', prob=0.5, degree=(90, 270)),
    dict(type='PhotoMetricDistortion'),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=crop_size,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    # One dataloader sample contains num_pseudo pseudo images and num_real real
    # images. With these defaults, samples_per_gpu=1 gives 4 pseudo images.
    samples_per_gpu=1,
    workers_per_gpu=16,

    train=dict(
        type=dataset_type,
        data_root=train_data_root,
        img_dir='imgs/train',
        ann_dir='labels/train',
        img_suffix='.tif',
        seg_map_suffix='.png',
        out_size=crop_size,
        grid_shape=grid_shape,
        num_pseudo=4,
        num_real=1,
        mask_pattern='checkerboard',
        mask_offset=0,
        ignore_index=255,
        img_norm_cfg=img_norm_cfg,
        compensate_random_patch_aug=True,
        random_patch_pipeline=random_patch_pipeline),

    val=dict(
        type=val_dataset_type,
        data_root=test_data_root,
        img_dir='imgs/test',
        ann_dir='labels/test',
        pipeline=test_pipeline),

    test=dict(
        type=val_dataset_type,
        data_root=test_data_root,
        img_dir='imgs/test',
        ann_dir='labels/test',
        pipeline=test_pipeline))
