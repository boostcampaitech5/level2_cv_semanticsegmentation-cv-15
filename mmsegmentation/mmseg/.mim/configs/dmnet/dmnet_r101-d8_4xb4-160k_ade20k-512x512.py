_base_ = "./dmnet_r50-d8_4xb4-160k_ade20k-512x512.py"
model = dict(pretrained="open-mmlab://resnet101_v1c", backbone=dict(depth=101))
