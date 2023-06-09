from hydra.core.config_store import ConfigStore
from torch.nn import BCEWithLogitsLoss

from src.config import full_builds

BCEWithLogitsLossConfig = full_builds(BCEWithLogitsLoss)


def _register_configs():
    cs = ConfigStore.instance()

    cs.store(group="loss", name="bce_with_logits", node=BCEWithLogitsLossConfig)
