from hydra_zen import instantiate
from omegaconf import OmegaConf

from src.data.datamodule import XRayDataModule
from src.model import LightningModel
from src.utils import set_seed


def train(config):
    exp = instantiate(config)

    set_seed(42)

    architecture = exp.architecture
    optimizer = exp.optimizer
    loss = exp.loss
    scheduler = exp.scheduler
    trainer = exp.trainer(logger=exp.logger, callbacks=exp.callbacks)

    model = LightningModel(
        model=architecture,
        optimizer=optimizer,
        loss=loss,
        scheduler=scheduler,
    )

    datamodule = XRayDataModule(
        train_dataset=exp.train_dataset,
        train_loader=exp.train_loader,
        val_dataset=exp.val_dataset,
        val_loader=exp.val_loader,
        test_dataset=exp.test_dataset,
        test_loader=exp.test_loader,
        transforms=exp.transforms,
    )

    trainer.logger.watch(
        model=model,
        log="gradients",
        log_freq=10,
    )

    trainer.logger.experiment.config.update(OmegaConf.to_container(config))

    trainer.fit(model=model, datamodule=datamodule)

    trainer.test(dataloaders=datamodule.test_dataloader())


if __name__ == "__main__":
    train()
