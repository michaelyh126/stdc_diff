_base_ = [
    '../_base_/datasets/deepglobe_1224x1224.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]

# LSKNet is currently defined in mmseg/models/decode_heads/lsknet.py,
# but the class itself is registered as a backbone.
custom_imports = dict(
    imports=['mmseg.models.decode_heads.lsknet'],
    allow_failed_imports=False)

norm_cfg = dict(type='BN', requires_grad=True)

model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='LSKNet',
        img_size=1224,
        in_chans=3,
        embed_dims=[32, 64, 160, 256],
        mlp_ratios=[8, 8, 4, 4],
        drop_rate=0.,
        drop_path_rate=0.1,
        depths=[3, 3, 5, 2],
        num_stages=4,
        norm_cfg=norm_cfg),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[32, 64, 160, 256],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=7,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# 1224x1224 crops are memory-heavy. Increase this after confirming GPU memory.
data = dict(samples_per_gpu=8, workers_per_gpu=16)

# LSKNet is transformer/MetaFormer-like, so AdamW is usually more stable than
# the SGD setting used by the STDC baseline.
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=5e-5,
    weight_decay=5e-4)
lr_config = dict(policy='poly', power=0.9, min_lr=0.0, by_epoch=False)
