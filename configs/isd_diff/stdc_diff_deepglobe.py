_base_ = [
    '../_base_/models/stdc_diff.py', '../_base_/datasets/deepglobe_1224x1224.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    pretrained='open-mmlab://resnet18_v1c',
    down_ratio=4,
    backbone=dict(depth=18),
    decode_head=[
        dict(
            type='RefineASPPHead',
            in_channels=512,
            in_index=3,
            channels=128,
            dilations=(1, 12, 24, 36),
            dropout_ratio=0.1,
            num_classes=7,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),

        dict(
            type='STDCDiffHead',
            in_channels=3,
            prev_channels=128,
            down_ratio=4,
            img_size=(1224,1224),
            channels=128,
            num_classes=7,
            dropout_ratio=0.1,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        dict(
            type='DiffHead',
            in_channels=1,
            in_index=3,
            channels=64,
            dropout_ratio=0.1,
            num_classes=7,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='BCEDiceLoss')),
    ],
    auxiliary_head=dict(in_channels=256, channels=64, num_classes=7))
