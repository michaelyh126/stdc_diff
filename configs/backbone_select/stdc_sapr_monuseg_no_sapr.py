_base_ = ['./stdc_sapr_monuseg.py']

# Keep the same SAPR dataset construction, so real-image patches are still
# mixed into pseudo-large images. Only the SAPR matching loss is disabled.
model = dict(
    enable_sapr=False,
    sapr_loss_weight=0.0)
