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

from src.config import full_builds

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


ResizeSmallConfig = full_builds(
    Compose,
    transforms=builds(
        list,
        [
            ResizeConfig(height=512, width=512),
            NormalizeConfig,
            ToTensorV2Config,
        ],
    ),
)

ResizeMediumConfig = full_builds(
    Compose,
    transforms=builds(
        list,
        [
            ResizeConfig(height=1024, width=1024),
            NormalizeConfig,
            ToTensorV2Config,
        ],
    ),
)

ResizeLargeConfig = full_builds(
    Compose,
    transforms=builds(
        list,
        [
            ResizeConfig(height=2048, width=2048),
            NormalizeConfig,
            ToTensorV2Config,
        ],
    ),
)

ClaheSharpenConfig = full_builds(
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

ClaheConfig = full_builds(
    Compose,
    transforms=builds(
        list,
        [
            CLAHEConfig,
            ResizeConfig,
            NormalizeConfig,
            ToTensorV2Config,
        ],
    ),
)

ResizeMediumClaheConfig = full_builds(
    Compose,
    transforms=builds(
        list,
        [
            CLAHEConfig,
            ResizeConfig(height=1024, width=1024),
            NormalizeConfig,
            ToTensorV2Config,
        ],
    ),
)


def _register_configs():
    cs = ConfigStore.instance()

    cs.store(
        group="transforms",
        name="resize_small",
        node=ResizeSmallConfig,
    )
    cs.store(
        group="transforms",
        name="resize_medium",
        node=ResizeMediumConfig,
    )
    cs.store(
        group="transforms",
        name="resize_large",
        node=ResizeLargeConfig,
    )

    cs.store(
        group="transforms",
        name="clahe_sharpen",
        node=ClaheSharpenConfig,
    )

    cs.store(
        group="transforms",
        name="clahe",
        node=ClaheConfig,
    )

    cs.store(
        group="transforms",
        name="clahe_medium",
        node=ResizeMediumClaheConfig,
    )
