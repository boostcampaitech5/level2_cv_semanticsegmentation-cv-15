from hydra.core.config_store import ConfigStore
from segmentation_models_pytorch import DeepLabV3, DeepLabV3Plus, Unet, UnetPlusPlus

from src.architecture.HRNet import HighResolutionNet
from src.architecture.Unet3Plus import Unet3Plus
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

HRNetConfig = full_builds(HighResolutionNet, version_number=48, output_size=512)

Unet3PlusConfig = full_builds(
    Unet3Plus,
    encoder_name="timm-resnest269e",
    encoder_weights="imagenet",
    in_channels=3,
    classes=29,
)

HRNetUnet3PlusConfig = full_builds(
    Unet3Plus,
    encoder_name="tu-hrnet_w48",
    encoder_depth=3,
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
    cs.store(
        group="architecture",
        name="hrnet",
        node=HRNetConfig,
    )
    cs.store(
        group="architecture",
        name="unet3plus",
        node=HRNetConfig,
    )
    cs.store(
        group="architecture",
        name="hrnetunet3plus",
        node=HRNetUnet3PlusConfig,
    )
