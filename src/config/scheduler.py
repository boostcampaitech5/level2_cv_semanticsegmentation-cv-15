from hydra.core.config_store import ConfigStore
from torch.optim.lr_scheduler import OneCycleLR

from src.config import partial_builds
from src.scheduler import CosineAnnealingWarmUpRestarts

OneCycleLRConfig = partial_builds(
    OneCycleLR,
    max_lr="${optimizer.lr}",
    pct_start=0.2,
    anneal_strategy="cos",
    div_factor=1e3,
    final_div_factor=1e6,
)

CosineAnnealingWarmUpRestartsConfig = partial_builds(
    CosineAnnealingWarmUpRestarts,
    T_0=2000,
    T_mult=1,
    eta_max=0.003,
    T_up=600,
    gamma=0.5,
)


def _register_configs():
    cs = ConfigStore.instance()

    cs.store(group="scheduler", name="onecycle", node=OneCycleLRConfig)
    cs.store(
        group="scheduler", name="cosinecustom", node=CosineAnnealingWarmUpRestartsConfig
    )
