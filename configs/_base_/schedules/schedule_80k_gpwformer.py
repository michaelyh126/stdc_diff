# GPWFormer training schedule following the paper implementation details.
# Paper setting:
# - optimizer: Adam
# - initial lr: 5e-5
# - lr decay: poly, power=0.9
# - max iters: 40k / 80k / 160k / 80k for
#   Inria Aerial / DeepGlobe / Cityscapes / ISIC respectively
#
# This file uses the 80k DeepGlobe setting, which matches the current
# WFormer config in this repository.

# optimizer
optimizer = dict(type='Adam', lr=5e-5, weight_decay=0.0)
optimizer_config = dict()

# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=0.0, by_epoch=False)

# runtime settings
runner = dict(type='IterBasedRunner', max_iters=80000)
checkpoint_config = dict(by_epoch=False, interval=4000)
evaluation = dict(interval=4000, metric='mIoU')
