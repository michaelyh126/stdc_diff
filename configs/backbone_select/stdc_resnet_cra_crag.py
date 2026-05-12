_base_ = [
    '../_base_/models/stdc_resnet_cra.py',
    '../_base_/datasets/crag.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]

model = dict(
    num_classes=2,
    test_cfg=dict(mode='whole'))
