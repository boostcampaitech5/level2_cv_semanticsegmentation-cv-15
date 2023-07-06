# dataset settings
dataset_type = "HandBoneDataset"
data_root = "/opt/ml/input/data"
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="CustomLoadAnnotations"),
    dict(type="Resize", scale=(1024, 1024)),
    dict(type="TransposeAnnotations"),
    dict(type="CLAHE", clip_limit=20.0, tile_grid_size=(12, 12)),
    # dict(type="RandomRotate", prob=0.5, degree=10),
    dict(type="PhotoMetricDistortion"),
    dict(type="PackSegInputs"),
]
# val_pipeline = [
#     dict(type="LoadImageFromFile"),
#     dict(type="Resize", scale=(1024, 1024)),
#     # add loading annotation after ``Resize`` because ground truth
#     # does not need to do resize data transform
#     dict(type="CustomLoadAnnotations"),
#     dict(type="TransposeAnnotations"),
#     dict(type="PackSegInputs"),
# ]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(1024, 1024), keep_ratio=True),
    dict(type="CLAHE"),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    # dict(type="CustomLoadAnnotations", reduce_zero_label=True),
    # dict(type="TransposeAnnotations"),
    dict(type="PackSegInputs"),
]
train_dataloader = dict(
    batch_size=2,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        reduce_zero_label=False,
        data_prefix=dict(
            img_path="train_new/DCM", seg_map_path="train_new/outputs_json"
        ),
        pipeline=train_pipeline,
    ),
)
# val_dataloader = dict(
#     batch_size=1,
#     num_workers=1,
#     persistent_workers=True,
#     sampler=dict(type="DefaultSampler", shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         reduce_zero_label=False,
#         data_prefix=dict(img_path="train/DCM", seg_map_path="train/outputs_json"),
#         pipeline=val_pipeline,
#     ),
# )
test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        reduce_zero_label=False,
        data_prefix=dict(img_path="test/DCM"),
        pipeline=test_pipeline,
        test_mode=True,
    ),
)

# val_evaluator = dict(type="DiceMetric")
test_evaluator = dict(type="DiceMetric")
