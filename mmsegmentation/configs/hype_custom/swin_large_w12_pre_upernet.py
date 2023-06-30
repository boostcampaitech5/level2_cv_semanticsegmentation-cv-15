_base_ = [
    "../_base_/models/upernet_swin.py",
    "../_base_/datasets/handbone.py",
    "../_base_/default_runtime.py",
]
class_weight = [
    0.9,  # finger-1
    0.9,  # finger-2
    0.9,  # finger-3
    0.9,  # finger-4
    0.9,  # finger-5
    0.9,  # finger-6
    0.9,  # finger-7
    0.9,  # finger-8
    0.9,  # finger-9
    0.9,  # finger-10
    0.9,  # finger-11
    0.9,  # finger-12
    0.9,  # finger-13
    0.9,  # finger-14
    0.9,  # finger-15
    0.9,  # finger-16
    0.9,  # finger-17
    0.9,  # finger-18
    0.9,  # finger-19
    1.0,  # Trapezium
    1.0,  # Trapezoid
    0.9,  # Capitate
    0.9,  # Hamate
    0.9,  # Scaphoid
    0.9,  # Lunate
    1.0,  # Triquetrum
    1.0,  # Pisiform
    0.9,  # Radius
    0.9,  # Ulna
]
data_preprocessor = dict(
    type="SegDataPreProcessor",
    mean=[0.0, 0.0, 0.0],
    std=[255.0, 255.0, 255.0],
    bgr_to_rgb=True,
    size=(512, 512),
    pad_val=0,
    seg_pad_val=255,
)
checkpoint_file = "https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_large_patch4_window12_384_22k_20220412-6580f57d.pth"  # noqa

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        init_cfg=dict(type="Pretrained", checkpoint=checkpoint_file),
        pretrain_img_size=384,
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        use_abs_pos_embed=False,
        drop_path_rate=0.3,
        patch_norm=True,
    ),
    decode_head=dict(
        in_channels=[192, 384, 768, 1536],
        num_classes=29,
        loss_decode=[
            dict(
                type="CrossEntropyLoss",
                use_sigmoid=True,
                loss_weight=0.1,
            ),
            # dict(
            #     type="DiceLoss",
            #     loss_weight=1.0,
            #     # class_weight=class_weight,
            # ),
        ],
    ),
    auxiliary_head=dict(
        in_channels=768,
        num_classes=29,
        loss_decode=[
            dict(
                type="CrossEntropyLoss",
                use_sigmoid=True,
                loss_weight=0.4,
            ),
            # dict(
            #     type="DiceLoss",
            #     loss_weight=0.4,
            #     # class_weight=class_weight,
            # ),
        ],
    ),
)

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
# optimizer
optimizer = dict(type="AdamW", lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01)
optim_wrapper = dict(type="OptimWrapper", optimizer=optimizer, clip_grad=None)
# learning policy
param_scheduler = [
    dict(type="LinearLR", start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type="PolyLR",
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=32000,
        by_epoch=False,
    ),
]
# training schedule for 20k
train_cfg = dict(type="IterBasedTrainLoop", max_iters=32000)
val_cfg = None
test_cfg = dict(type="TestLoop")
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(type="CheckpointHook", by_epoch=False, interval=2000),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="SegVisualizationHook"),
)

# By default, models are trained on 8 GPUs with 2 images per GPU
train_dataloader = dict(batch_size=2)
val_dataloader = None
test_dataloader = dict(batch_size=1)

val_cfg = None
