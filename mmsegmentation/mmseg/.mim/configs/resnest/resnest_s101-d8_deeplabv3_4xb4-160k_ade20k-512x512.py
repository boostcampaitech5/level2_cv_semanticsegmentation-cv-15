_base_ = "../deeplabv3/deeplabv3_r101-d8_4xb4-160k_ade20k-512x512.py"
model = dict(
    pretrained="open-mmlab://resnest101",
    backbone=dict(
        type="ResNeSt",
        stem_channels=128,
        radix=2,
        reduction_factor=4,
        avg_down_stride=True,
    ),
)
