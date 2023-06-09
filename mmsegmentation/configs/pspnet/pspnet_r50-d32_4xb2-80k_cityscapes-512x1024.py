_base_ = [
    "../_base_/models/pspnet_r50-d8.py",
    "../_base_/datasets/cityscapes.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/schedule_80k.py",
]
crop_size = (512, 1024)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(dilations=(1, 1, 2, 4), strides=(1, 2, 2, 2)),
)
