# dataset settings
dataset_type = 'DeepGlobeRandomPatchComposeDataset'
val_dataset_type = 'DeepGlobeDataset'

train_data_root = '/root/autodl-tmp/land-train612'
test_data_root = '/root/autodl-tmp/land-train'

img_norm_cfg = dict(
    mean=[0, 0, 0],
    std=[255, 255, 255],
    to_rgb=True
)

crop_size = (1224, 1224)
patch_size = (612, 612)

train_pipeline = [
    dict(type='Resize', img_scale=(2448, 2448), ratio_range=(0.5, 2.)),
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
        img_scale=(2448,2448),
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
    samples_per_gpu=8,
    workers_per_gpu=16,

    train=dict(
        type=dataset_type,
        data_root=train_data_root,
        img_dir='img_dir/train',
        ann_dir='rgb2id/train',
        img_suffix='.png',
        seg_map_suffix='.png',
        patch_size=patch_size,
        out_size=crop_size,
        ignore_index=255,
        pipeline=train_pipeline
    ),

    val=dict(
        type=val_dataset_type,
        data_root=test_data_root,
        img_dir='img_dir/test',
        ann_dir='rgb2id/test',
        pipeline=test_pipeline
    ),

    test=dict(
        type=val_dataset_type,
        data_root=test_data_root,
        img_dir='img_dir/test',
        ann_dir='rgb2id/test',
        pipeline=test_pipeline
    )
)
