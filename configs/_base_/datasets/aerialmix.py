# dataset settings
dataset_type = 'AerialComposeDataset'
val_dataset_type = 'InriaAerialDataset'
data_root = '/root/autodl-tmp/aerial'
train_data_root = '/root/autodl-tmp/aerial_500'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

patch_size = (500, 500)
crop_size = (2500, 2500)

train_pipeline = [
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
        img_scale=(2500, 2500),
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
    samples_per_gpu=2,
    workers_per_gpu=16,
    train=dict(
        type=dataset_type,
        data_root=train_data_root,
        img_dir='imgs/train',
        ann_dir='labels/train',
        patch_size=patch_size,
        out_size=crop_size,
        pipeline=train_pipeline),
    val=dict(
        type=val_dataset_type,
        data_root=data_root,
        img_dir='imgs/test',
        ann_dir='labels/test',
        pipeline=test_pipeline),
    test=dict(
        type=val_dataset_type,
        data_root='/root/autodl-tmp/aerial_2500',
        img_dir='imgs/test',
        ann_dir='labels/test',
        pipeline=test_pipeline))
