from hydra.core.config_store import ConfigStore
from hydra_zen import builds
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme

from src.config import full_builds

ModelCheckpointConfig = full_builds(
    ModelCheckpoint,
    dirpath="checkpoint",
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

NoEarlyStopCallbackConfig = builds(
    list,
    [
        ModelCheckpointConfig,
        LearningRateMonitorConfig,
        RichModelSummaryConfig,
        RichProgressBarConfig,
    ],
)

NoCheckPointConfig = builds(
    list,
    [
        LearningRateMonitorConfig,
        EarlyStoppingConfig,
        RichModelSummaryConfig,
        RichProgressBarConfig,
    ],
)

NoEalryCheckPointConfig = builds(
    list,
    [
        LearningRateMonitorConfig,
        RichModelSummaryConfig,
        RichProgressBarConfig,
    ],
)


def _register_configs():
    cs = ConfigStore.instance()

    cs.store(group="callbacks", name="basic_callbacks", node=BasicCallbackConfig)
    cs.store(group="callbacks", name="no_ealrystop", node=NoEarlyStopCallbackConfig)
    cs.store(group="callbacks", name="no_checkpoint", node=NoCheckPointConfig)
    cs.store(
        group="callbacks", name="no_ealry_no_ckeckpoint", node=NoEalryCheckPointConfig
    )
