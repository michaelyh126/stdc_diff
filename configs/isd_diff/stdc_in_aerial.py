_base_ = [
    '../_base_/models/stdc_in.py', '../_base_/datasets/aeril_2500x2500.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    # pretrained='open-mmlab://resnet18_v1c',
    # down_ratio=4,
    # backbone=dict(depth=18),
    decode_head=[
        # dict(
        #     type='RefineASPPHead',
        #     in_channels=512,
        #     in_index=3,
        #     channels=128,
        #     dilations=(1, 12, 24, 36),
        #     dropout_ratio=0.1,
        #     num_classes=2,
        #     norm_cfg=norm_cfg,
        #     align_corners=False,
        #     loss_decode=dict(
        #         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),

        dict(
            type='STDCInHead',
            in_channels=3,
            prev_channels=128,
            img_size=(2500,2500),
            down_ratio=4,
            channels=128,
            num_classes=2,
            dropout_ratio=0.1,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        # dict(
        #     type='DiffHead',
        #     in_channels=1,
        #     in_index=3,
        #     channels=64,
        #     dropout_ratio=0.1,
        #     num_classes=2,
        #     norm_cfg=norm_cfg,
        #     align_corners=False,
        #     loss_decode=dict(
        #         type='BCEDiceLoss')),
    ],
    auxiliary_head=dict(in_channels=256, channels=64, num_classes=2))
