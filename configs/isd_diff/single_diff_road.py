_base_ = [
    '../_base_/models/single_diff.py', '../_base_/datasets/road.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
checkpoint_teacher = '/root/autodl-tmp/Teacher_SegFormer_B3_city.pth'
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    # pretrained='open-mmlab://resnet18_v1c',
    # down_ratio=4,
    # backbone=dict(depth=18),
    decode_head=[
        dict(
            type='SingleDiffHead',
            in_channels=3,
            prev_channels=128,
            down_ratio=4,
            img_size=(2500,2500),
            channels=128,
            num_classes=2,
            dropout_ratio=0.1,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),

        # dict(
        #     type='VitGuidanceHead',
        #     init_cfg=dict(
        #         type='Pretrained',
        #         checkpoint=checkpoint_teacher),
        #     in_channels=256,
        #     channels=256,
        #     base_channels=64,
        #     in_index=0,
        #     num_classes=7,
        #     loss_decode=dict(type='AlignmentLoss', loss_weight=[3, 15, 15, 15])),

    ],
    auxiliary_head=dict(in_channels=256, channels=64, num_classes=2)
)

