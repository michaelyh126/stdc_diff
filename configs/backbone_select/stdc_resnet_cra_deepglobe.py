_base_ = [
    '../_base_/models/stdc_resnet_cra.py',
    '../_base_/datasets/deepglobe_1224x1224.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]

model = dict(num_classes=7)
