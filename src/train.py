from hydra_zen import instantiate
from omegaconf import OmegaConf

from src.model import UNetPlusPlusResNet50
from src.utils import set_seed


def train(config):
    exp = instantiate(config)

    set_seed(42)

    architecture = exp.architecture
    optimizer = exp.optimizer
    loss = exp.loss
    scheduler = exp.scheduler
    datamodule = exp.datamodule
    trainer = exp.trainer

    model = UNetPlusPlusResNet50(
        model=architecture,
        optimizer=optimizer,
        loss=loss,
        scheduler=scheduler,
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
