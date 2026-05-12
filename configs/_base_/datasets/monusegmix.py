# dataset settings
dataset_type = 'MoNuSegRandomPatchComposeDataset'
val_dataset_type = 'MoNuSegDataset'

train_data_root = '/root/autodl-tmp/MoNuSeg'
test_data_root = '/root/autodl-tmp/MoNuSeg'

img_norm_cfg = dict(
    mean=[164.258, 114.090, 153.999],
    std=[59.028, 59.278, 47.782],
    to_rgb=True)

crop_size = (1000, 1000)
patch_size = (250, 250)

train_pipeline = [
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomRotate', prob=0.5, degree=(90, 270)),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
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
    samples_per_gpu=4,
    workers_per_gpu=16,

    train=dict(
        type=dataset_type,
        data_root=train_data_root,
        img_dir='imgs/train',
        ann_dir='labels/train',
        img_suffix='.tif',
        seg_map_suffix='.png',
        patch_size=patch_size,
        out_size=crop_size,
        ignore_index=255,
        pipeline=train_pipeline),

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
