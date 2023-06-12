from hydra.core.config_store import ConfigStore
from segmentation_models_pytorch import DeepLabV3, DeepLabV3Plus, Unet, UnetPlusPlus

from src.config import full_builds

UnetConfig = full_builds(
    Unet,
    encoder_name="resnet50",
    encoder_weights="imagenet",
    in_channels=3,
    classes=29,
)

UnetPlusPlusConfig = full_builds(
    UnetPlusPlus,
    encoder_name="timm-resnest269e",
    encoder_weights="imagenet",
    in_channels=3,
    classes=29,
)

DeepLabV3Config = full_builds(
    DeepLabV3,
    encoder_name="resnet50",
    encoder_weights="imagenet",
    in_channels=3,
    classes=29,
)

DeepLabV3PlusConfig = full_builds(
    DeepLabV3Plus,
    encoder_name="resnet50",
    encoder_weights="imagenet",
    in_channels=3,
    classes=29,
)


def _register_configs():
    cs = ConfigStore.instance()

    cs.store(
        group="architecture",
        name="unet",
        node=UnetConfig,
    )
    cs.store(
        group="architecture",
        name="unet++",
        node=UnetPlusPlusConfig,
    )
    cs.store(
        group="architecture",
        name="deeplabv3",
        node=DeepLabV3Config,
    )
    cs.store(
        group="architecture",
        name="deeplabv3plus",
        node=DeepLabV3PlusConfig,
    )
