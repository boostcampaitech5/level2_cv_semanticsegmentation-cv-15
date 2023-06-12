from albumentations import (
    CLAHE,
    Compose,
    ElasticTransform,
    GridDistortion,
    HorizontalFlip,
    Normalize,
    OpticalDistortion,
    Resize,
    Sharpen,
    ToGray,
)
from albumentations.pytorch import ToTensorV2
from hydra.core.config_store import ConfigStore
from hydra_zen import builds
from torch.utils.data import DataLoader

from src.config import full_builds, partial_builds
from src.data.datamodule import XRayDataModule
from src.data.dataset import XRayDataset

######################################################
#         albumentations augmentation config         #
######################################################

ResizeConfig = full_builds(Resize, height=512, width=512, always_apply=True)

NormalizeConfig = full_builds(
    Normalize,
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
    max_pixel_value=255.0,
    always_apply=True,
)

ToGrayConfig = full_builds(ToGray, always_apply=True)

ToTensorV2Config = full_builds(ToTensorV2, always_apply=True)

HorizontalFlipConfig = full_builds(HorizontalFlip, p=0.5)

ElasticTransformConfig = full_builds(ElasticTransform, p=0.5)

GridDistortionConfig = full_builds(GridDistortion, p=0.5)

OpticalDistortionConfig = full_builds(OpticalDistortion, p=0.5)

SharpenConfig = full_builds(Sharpen, p=1.0, alpha=(0.2, 0.5), lightness=(0.5, 1.0))

CLAHEConfig = full_builds(CLAHE, p=1.0, clip_limit=(1, 4), tile_grid_size=(8, 8))

BasicTrainTransformConfig = full_builds(
    Compose,
    transforms=builds(
        list,
        [
            ResizeConfig,
            NormalizeConfig,
            ToTensorV2Config,
        ],
    ),
)

BasicValidationTransformConfig = full_builds(
    Compose,
    transforms=builds(
        list,
        [
            ResizeConfig,
            NormalizeConfig,
            ToTensorV2Config,
        ],
    ),
)

BasicTestTransformConfig = full_builds(
    Compose,
    transforms=builds(
        list,
        [
            ResizeConfig,
            NormalizeConfig,
            ToTensorV2Config,
        ],
    ),
)

ClaheSharpenTransformConfig = full_builds(
    Compose,
    transforms=builds(
        list,
        [
            SharpenConfig,
            CLAHEConfig,
            ResizeConfig,
            NormalizeConfig,
            ToTensorV2Config,
        ],
    ),
)

######################################################
#             lightning datamodule config            #
######################################################

BasicTrainDatasetConfig = partial_builds(
    XRayDataset,
    data_path="/opt/ml/level2_cv_semanticsegmentation-cv-15/data",
    split="train",
    transforms=None,
)

BasicValidationDatasetConfig = partial_builds(
    XRayDataset,
    data_path="/opt/ml/level2_cv_semanticsegmentation-cv-15/data",
    split="val",
    transforms=None,
)

BasicTestDatasetConfig = partial_builds(
    XRayDataset,
    data_path="/opt/ml/level2_cv_semanticsegmentation-cv-15/data",
    split="test",
    transforms=None,
)

BasicTrainDataloaderConfig = partial_builds(
    DataLoader,
    batch_size=8,
    shuffle=True,
    num_workers=6,
    prefetch_factor=2,
    persistent_workers=True,
    pin_memory=True,
)

BasicValidationDataloaderConfig = partial_builds(
    DataLoader,
    batch_size=2,
    num_workers=2,
    prefetch_factor=2,
    persistent_workers=True,
    pin_memory=True,
)

BasicTestDataloaderConfig = partial_builds(
    DataLoader,
    batch_size=2,
    num_workers=2,
    prefetch_factor=2,
    persistent_workers=True,
    pin_memory=True,
)

XRayDataModuleConfig = full_builds(
    XRayDataModule,
)


def _register_configs():
    cs = ConfigStore.instance()

    # datamodule config
    cs.store(group="datamodule", name="xray", node=XRayDataModuleConfig)

    # datamodule/dataset config
    cs.store(
        group="datamodule/train_dataset",
        name="basic_train_dataset",
        node=BasicTrainDatasetConfig,
    )
    cs.store(
        group="datamodule/val_dataset",
        name="basic_val_dataset",
        node=BasicValidationDatasetConfig,
    )
    cs.store(
        group="datamodule/test_dataset",
        name="basic_test_dataset",
        node=BasicTestDatasetConfig,
    )

    # datamodule/dataset/transforms config
    cs.store(
        group="datamodule/train_dataset/transforms",
        name="basic_train_transform",
        node=BasicTrainTransformConfig,
    )
    cs.store(
        group="datamodule/val_dataset/transforms",
        name="basic_val_transform",
        node=BasicValidationTransformConfig,
    )
    cs.store(
        group="datamodule/test_dataset/transforms",
        name="basic_test_transform",
        node=BasicTestTransformConfig,
    )
    cs.store(
        group="datamodule/train_dataset/transforms",
        name="aug1_train_transform",
        node=ClaheSharpenTransformConfig,
    )
    cs.store(
        group="datamodule/val_dataset/transforms",
        name="aug1_val_transform",
        node=ClaheSharpenTransformConfig,
    )
    cs.store(
        group="datamodule/test_dataset/transforms",
        name="aug1_test_transform",
        node=ClaheSharpenTransformConfig,
    )

    # datamodule/dataloader config
    cs.store(
        group="datamodule/train_loader",
        name="basic_train_loader",
        node=BasicTrainDataloaderConfig,
    )
    cs.store(
        group="datamodule/val_loader",
        name="basic_val_loader",
        node=BasicValidationDataloaderConfig,
    )
    cs.store(
        group="datamodule/test_loader",
        name="basic_test_loader",
        node=BasicTestDataloaderConfig,
    )
