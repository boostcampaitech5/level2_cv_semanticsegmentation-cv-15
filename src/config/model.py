from hydra.core.config_store import ConfigStore
from segmentation_models_pytorch import DeepLabV3, DeepLabV3Plus, Unet, UnetPlusPlus

from src.config import full_builds

UnetResNet50ImagenetConfig = full_builds(
    Unet,
    encoder_name="resnet50",
    encoder_weights="imagenet",
    in_channels=3,
    classes=29,
)

UnetPlusPlusResNet50ImagenetConfig = full_builds(
    UnetPlusPlus,
    encoder_name="resnet50",
    encoder_weights="imagenet",
    in_channels=3,
    classes=29,
)

UnetPlusPlusResNet152ImagenetConfig = full_builds(
    UnetPlusPlus,
    encoder_name="resnet152",
    encoder_weights="imagenet",
    in_channels=3,
    classes=29,
)

UnetPlusPlusTimmResNestImagenetConfig = full_builds(
    UnetPlusPlus,
    encoder_name="timm-resnest269e",
    encoder_weights="imagenet",
    in_channels=3,
    classes=29,
)

UnetPlusPlusResNestImagenetConfig = full_builds(
    UnetPlusPlus,
    encoder_name="resnest",
    encoder_weights="imagenet",
    in_channels=3,
    classes=29,
)

DeepLabV3ResNet50ImagenetConfig = full_builds(
    DeepLabV3,
    encoder_name="resnet50",
    encoder_weights="imagenet",
    in_channels=3,
    classes=29,
)

DeepLabV3PlusResNet50ImagenetConfig = full_builds(
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
        name="unet_resnet50_imagenet",
        node=UnetResNet50ImagenetConfig,
    )
    cs.store(
        group="architecture",
        name="unet++_resnet50_imagenet",
        node=UnetPlusPlusResNet50ImagenetConfig,
    )
    cs.store(
        group="architecture",
        name="unet++_resnet152_imagenet",
        node=UnetPlusPlusResNet152ImagenetConfig,
    )
    cs.store(
        group="architecture",
        name="unet++_timm_resnest_imagenet",
        node=UnetPlusPlusTimmResNestImagenetConfig,
    )
    cs.store(
        group="architecture",
        name="deeplabv3_resnet50_imagenet",
        node=DeepLabV3ResNet50ImagenetConfig,
    )
    cs.store(
        group="architecture",
        name="deeplabv3+_resnet50_imagenet",
        node=DeepLabV3PlusResNet50ImagenetConfig,
    )
