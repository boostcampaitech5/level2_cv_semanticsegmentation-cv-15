import pkgutil
from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore
from hydra_zen import make_custom_builds_fn
from omegaconf import MISSING

partial_builds = make_custom_builds_fn(
    populate_full_signature=True,
    zen_partial=True,
)

full_builds = make_custom_builds_fn(
    populate_full_signature=True,
)

defaults = [
    "_self_",
    {"architecture": "unet++_resnet152_imagenet"},
    {"optimizer": "adam"},
    {"loss": "dice_bce"},
    {"scheduler": "onecycle"},
    {"datamodule": "xray"},
    {"datamodule/train_dataset": "basic_train_dataset"},
    {"datamodule/val_dataset": "basic_val_dataset"},
    {"datamodule/test_dataset": "basic_test_dataset"},
    {"datamodule/train_dataset/transforms": "basic_train_transform"},
    {"datamodule/val_dataset/transforms": "basic_val_transform"},
    {"datamodule/test_dataset/transforms": "basic_test_transform"},
    {"datamodule/train_loader": "basic_train_loader"},
    {"datamodule/val_loader": "basic_val_loader"},
    {"datamodule/test_loader": "basic_test_loader"},
    {"trainer": "basic"},
    {"trainer/callbacks": "basic_callbacks"},
    {"trainer/logger": "wandb"},
]


@dataclass
class Config:
    defaults: list[Any] = field(default_factory=lambda: defaults)
    architecture: Any = MISSING
    optimizer: Any = MISSING
    loss: Any = MISSING
    scheduler: Any = MISSING
    datamodule: Any = MISSING
    trainer: Any = MISSING


def register_configs():
    cs = ConfigStore.instance()

    cs.store(name="default", node=Config)

    for module_info in pkgutil.walk_packages(__path__):
        name = module_info.name
        module_finder = module_info.module_finder

        module = module_finder.find_module(name).load_module(name)
        if hasattr(module, "_register_configs"):
            module._register_configs()
