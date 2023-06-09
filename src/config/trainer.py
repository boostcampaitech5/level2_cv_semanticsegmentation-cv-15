from hydra.core.config_store import ConfigStore
from hydra_zen import builds
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.profilers import PyTorchProfiler

from src.config import full_builds

######################################################
#              lightning profiler config             #
######################################################

ProfilerConfig = full_builds(
    PyTorchProfiler,
    dirpath="logs/",
    filename="profile-lightning-test",
    export_to_chrome=True,
)

######################################################
#               lightning logger config              #
######################################################

WandbLoggerConfig = full_builds(
    WandbLogger, entity="hype-squad", project="Segmentation", name="lightning-test"
)

######################################################
#             lightning callbacks config             #
######################################################

ModelCheckpointConfig = full_builds(
    ModelCheckpoint,
    dirpath="checkpoint/lightning-test",
    filename="epoch={epoch:02d}-val_loss={val_loss:.2f}",
    save_last=True,
    monitor="val_loss",
    save_top_k=3,
    mode="min",
    save_weights_only=False,
    auto_insert_metric_name=False,
)

LearningRateMonitorConfig = full_builds(
    LearningRateMonitor,
    logging_interval="step",
)

EarlyStoppingConfig = full_builds(
    EarlyStopping,
    monitor="val_loss",
    patience=5,
    mode="min",
    strict=True,
    check_finite=True,
)

RichModelSummaryConfig = full_builds(RichModelSummary, max_depth=3)

RichProgressBarThemeConfig = full_builds(
    RichProgressBarTheme,
    description="green_yellow",
    progress_bar="green1",
    progress_bar_finished="green1",
    progress_bar_pulse="#6206E0",
    batch_progress="green_yellow",
    time="grey82",
    processing_speed="grey82",
    metrics="grey82",
)

RichProgressBarConfig = full_builds(RichProgressBar, theme=RichProgressBarThemeConfig)

BasicCallbackConfig = builds(
    list,
    [
        ModelCheckpointConfig,
        LearningRateMonitorConfig,
        EarlyStoppingConfig,
        RichModelSummaryConfig,
        RichProgressBarConfig,
    ],
)

######################################################
#              lightning trainer config              #
######################################################

TrainerConfig = full_builds(
    Trainer,
    logger=WandbLoggerConfig,
    callbacks=BasicCallbackConfig,
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
        group="trainer", name="profile", node=TrainerConfig(profiler=ProfilerConfig)
    )

    cs.store(
        group="trainer/callbacks", name="basic_callbacks", node=BasicCallbackConfig
    )
    cs.store(group="trainer/logger", name="wandb", node=WandbLoggerConfig)
