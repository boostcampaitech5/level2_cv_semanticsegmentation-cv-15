_base_ = ["./seg_former.py"]

norm_cfg = dict(type="SyncBN", requires_grad=True)

# dataset settings
data_preprocessor = dict(
    type="SegDataPreProcessor",
    mean=[0.0, 0.0, 0.0],
    std=[255.0, 255.0, 255.0],
    bgr_to_rgb=True,
    size=(512, 512),
    pad_val=0,
    seg_pad_val=255,
)

# model settings
model = dict(
    data_preprocessor=data_preprocessor,
    pretrained="https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b5_640x640_160k_ade20k/segformer_mit-b5_640x640_160k_ade20k_20210801_121243-41d2845b.pth",
    backbone=dict(embed_dims=64, num_heads=[1, 2, 5, 8], num_layers=[3, 6, 40, 3]),
    decode_head=dict(in_channels=[64, 128, 320, 512]),
    auxiliary_head=dict(
        type="FCNHead",
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=29,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=0.4),
    ),
)
