_base_ = [
    '../_base_/models/stdc.py', '../_base_/datasets/monuseg_sapr.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_10k.py'
]

norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='StdcSAPR',
    enable_sapr=True,
    sapr_loss_weight=0.05,
    sapr_num_tokens=4,
    sapr_context_radius=1,
    sapr_temperature=0.07,
    sapr_grid_shape=(4, 4),
    sapr_loss_type='qkv',
    sapr_pool_size=4,
    sapr_pool_with_max=True,
    pretrained='open-mmlab://resnet18_v1c',
    down_ratio=4,
    backbone=dict(depth=18),
    decode_head=[
        dict(
            type='MyStdcSAPRHead',
            in_channels=3,
            prev_channels=128,
            down_ratio=4,
            img_size=(1000, 1000),
            channels=128,
            num_classes=2,
            dropout_ratio=0.1,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    ],
    auxiliary_head=dict(in_channels=256, channels=64, num_classes=2))
