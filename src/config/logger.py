from hydra.core.config_store import ConfigStore
from pytorch_lightning.loggers.wandb import WandbLogger

from src.config import full_builds

WandbLoggerConfig = full_builds(
    WandbLogger, entity="hype-squad", project="Segmentation", name="lightning-test"
)


def _register_configs():
    cs = ConfigStore.instance()
    cs.store(group="logger", name="wandb", node=WandbLoggerConfig)
