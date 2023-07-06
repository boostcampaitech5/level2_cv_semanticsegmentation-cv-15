from hydra.core.config_store import ConfigStore
from pytorch_lightning import Trainer
from pytorch_lightning.profilers import PyTorchProfiler

from src.config import full_builds, partial_builds

TorchProfilerConfig = full_builds(
    PyTorchProfiler,
    dirpath="logs/",
    filename="profile-lightning-test",
    export_to_chrome=True,
)

TrainerConfig = partial_builds(
    Trainer,
    max_epochs=150,
    gradient_clip_algorithm="norm",
    gradient_clip_val=1.0,
    log_every_n_steps=1,
    check_val_every_n_epoch=10,
    accelerator="gpu",
    devices="auto",
)


def _register_configs():
    cs = ConfigStore.instance()

    cs.store(group="trainer", name="basic", node=TrainerConfig)
    cs.store(
        group="trainer",
        name="profiler",
        node=TrainerConfig(profiler=TorchProfilerConfig),
    )
