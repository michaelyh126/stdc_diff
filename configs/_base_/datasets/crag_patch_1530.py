# dataset settings
dataset_type = 'CragRandomPatchComposeDataset'
val_dataset_type = 'InriaAerialDataset'

img_norm_cfg = dict(
    mean=[211.82, 184.24, 218.64],
    std=[34.06, 44.00, 25.41],
    to_rgb=True
)

crop_size = (1530, 1530)
patch_size = (510, 510)

train_pipeline = [
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
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
            dict(type='Resize', keep_ratio=False),
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
        data_root='/root/autodl-tmp/Crag',
        img_dir='train/Images',
        ann_dir='train/Annotation',
        img_suffix='.png',
        seg_map_suffix='.png',
        patch_size=patch_size,
        out_size=crop_size,
        ignore_index=255,
        pipeline=train_pipeline
    ),

    val=dict(
        type=val_dataset_type,
        data_root='/root/autodl-tmp/Crag',
        img_dir='valid/Images',
        ann_dir='valid/Annotation',
        pipeline=test_pipeline
    ),

    test=dict(
        type=val_dataset_type,
        data_root='/root/autodl-tmp/Crag',
        img_dir='valid/Images',
        ann_dir='valid/Annotation',
        pipeline=test_pipeline
    )
)
