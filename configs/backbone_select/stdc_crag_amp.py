_base_ = ['./stdc_crag.py']

# Enable MMCV mixed precision training for this config.
# The hook wraps the model for fp16 forward/backward and keeps dynamic loss
# scaling to reduce the chance of fp16 gradient underflow.
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale='dynamic')

# Keep this flag so test/benchmark scripts that check cfg.fp16 also wrap the
# model consistently when this AMP config is reused for evaluation.
fp16 = dict(loss_scale='dynamic')
