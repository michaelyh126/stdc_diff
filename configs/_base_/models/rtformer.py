# model settings
model = dict(
    type='RTFormer',
    num_classes=19,
    layer_nums=(2, 2, 2, 2),
    base_channels=64,
    spp_channels=128,
    num_heads=8,
    head_channels=128,
    drop_rate=0.,
    drop_path_rate=0.2,
    use_aux_head=True,
    use_injection=(True, True),
    cross_size=12,
    in_channels=3,
    pretrained=None,
    ignore_index=255,
    align_corners=False,
    loss_decode=dict(
        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
    loss_aux=dict(
        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
