from functools import partial

import albumentations as A
import segmentation_models_pytorch as smp
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
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR

from data.datamodule import XRayDataModule
from model import FCNResNet50


def train():
    logger = WandbLogger(
        entity="hype-squad", project="Segmentation", name="lightning-test"
    )

    model = smp.UnetPlusPlus(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=3,
        classes=29,
    )

    optimizer = partial(Adam, lr=0.0001)
    loss = BCEWithLogitsLoss()
    scheduler = partial(OneCycleLR)

    model = FCNResNet50(
        model,
        optimizer,
        loss,
        scheduler,
    )

    logger.watch(
        model=model,
        log="gradients",
        log_freq=10,
    )

    transform = A.Compose([A.Resize(512, 512)])

    datamodule = XRayDataModule(
        data_path="/opt/ml/level2_cv_semanticsegmentation-cv-15/data",
        batch_size=8,
        shuffle=True,
        transforms=transform,
    )

    callbacks = [
        ModelCheckpoint(
            dirpath="checkpoint/lightning-test",
            filename="epoch={epoch:02d}-val_loss={val_loss:.2f}",
            save_last=True,
            monitor="val_loss",
            save_top_k=3,
            mode="min",
            save_weights_only=False,
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor(logging_interval="step"),
        EarlyStopping(
            monitor="val_loss", patience=5, mode="min", strict=True, check_finite=True
        ),
        RichModelSummary(max_depth=3),
        RichProgressBar(
            theme=RichProgressBarTheme(
                description="green_yellow",
                progress_bar="green1",
                progress_bar_finished="green1",
                progress_bar_pulse="#6206E0",
                batch_progress="green_yellow",
                time="grey82",
                processing_speed="grey82",
                metrics="grey82",
            )
        ),
    ]

    profiler = PyTorchProfiler(
        dirpath="logs/",
        filename="profile-lightning-test",
        export_to_chrome=True,
    )

    trainner = Trainer(
        logger=logger,
        callbacks=callbacks,
        max_epochs=10,
        gradient_clip_algorithm="norm",
        gradient_clip_val=1.0,
        log_every_n_steps=1,
        accelerator="gpu",
        devices="auto",
        profiler=profiler,
    )

    trainner.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    train()
