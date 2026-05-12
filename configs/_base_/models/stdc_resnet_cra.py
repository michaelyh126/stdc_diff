# model settings
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='STDCResNetCRA',
    num_classes=19,
    stdc_base=64,
    fusion_channels=128,
    num_heads=8,
    cross_size=12,
    drop_rate=0.,
    drop_path_rate=0.,
    use_aux_head=True,
    in_channels=3,
    resnet_cfg=dict(
        type='ResNetV1c',
        depth=18,
        in_channels=64,
        stem_channels=64,
        base_channels=64,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    ignore_index=255,
    align_corners=False,
    loss_decode=dict(
        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
    loss_aux=dict(
        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
