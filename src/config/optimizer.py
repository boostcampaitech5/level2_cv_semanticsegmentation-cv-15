from hydra.core.config_store import ConfigStore
from torch.optim import SGD, Adam, AdamW, RAdam, RMSprop

from src.config import partial_builds

SGDConfig = partial_builds(SGD, lr=0.0001)
RMSPropConfig = partial_builds(RMSprop, lr=0.0001)
AdamConfig = partial_builds(Adam, lr=0.0001)
AdamWConfig = partial_builds(AdamW, lr=0.0001)
RAdamConfig = partial_builds(RAdam, lr=0.0001)


def _register_configs():
    cs = ConfigStore.instance()

    cs.store(group="optimizer", name="sgd", node=SGDConfig)
    cs.store(group="optimizer", name="rmsprop", node=RMSPropConfig)
    cs.store(group="optimizer", name="adam", node=AdamConfig)
    cs.store(group="optimizer", name="adamw", node=AdamWConfig)
    cs.store(group="optimizer", name="radam", node=RAdamConfig)
