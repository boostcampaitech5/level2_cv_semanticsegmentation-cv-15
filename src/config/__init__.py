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
    {"architecture": "unet++"},
    {"optimizer": "adam"},
    {"loss": "dice_bce"},
    {"scheduler": "onecycle"},
    {"train_dataset": "basic_train_dataset"},
    {"val_dataset": "basic_val_dataset"},
    {"test_dataset": "basic_test_dataset"},
    {"train_loader": "basic_train_loader"},
    {"val_loader": "basic_val_loader"},
    {"test_loader": "basic_test_loader"},
    {"transforms": "resize_small"},
    {"trainer": "basic"},
    {"callbacks": "basic_callbacks"},
    {"logger": "wandb"},
]


@dataclass
class Config:
    defaults: list[Any] = field(default_factory=lambda: defaults)
    architecture: Any = MISSING
    optimizer: Any = MISSING
    loss: Any = MISSING
    scheduler: Any = MISSING
    train_dataset: Any = MISSING
    val_dataset: Any = MISSING
    test_dataset: Any = MISSING
    train_loader: Any = MISSING
    val_loader: Any = MISSING
    test_loader: Any = MISSING
    transforms: Any = MISSING
    trainer: Any = MISSING
    callbacks: Any = MISSING
    logger: Any = MISSING


def register_configs():
    cs = ConfigStore.instance()

    cs.store(name="default", node=Config)

    for module_info in pkgutil.walk_packages(__path__):
        name = module_info.name
        module_finder = module_info.module_finder

        module = module_finder.find_module(name).load_module(name)
        if hasattr(module, "_register_configs"):
            module._register_configs()
