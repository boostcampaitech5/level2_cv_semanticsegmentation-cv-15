from hydra.core.config_store import ConfigStore
from torch.nn import BCEWithLogitsLoss

from src.config import full_builds
from src.loss import DiceBCELoss, DiceLoss

BCEWithLogitsLossConfig = full_builds(BCEWithLogitsLoss)

DiceLossConfig = full_builds(DiceLoss, smooth=1.0)

DiceBCELossConfig = full_builds(DiceBCELoss, dice_smooth=1.0, bce_weight=0.5)


def _register_configs():
    cs = ConfigStore.instance()

    cs.store(group="loss", name="bce_with_logits", node=BCEWithLogitsLossConfig)
    cs.store(group="loss", name="dice", node=DiceLossConfig)
    cs.store(group="loss", name="dice_bce", node=DiceBCELossConfig)
