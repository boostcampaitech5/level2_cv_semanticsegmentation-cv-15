import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class FCNResNet50(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        loss: nn.Module,
        scheduler: LRScheduler,
    ):
        super().__init__()
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dice = None
        # fmt: off
        self.classes = [
            "finger-1", "finger-2", "finger-3", "finger-4", "finger-5",
            "finger-6", "finger-7", "finger-8", "finger-9", "finger-10",
            "finger-11", "finger-12", "finger-13", "finger-14", "finger-15",
            "finger-16", "finger-17", "finger-18", "finger-19", "Trapezium",
            "Trapezoid", "Capitate", "Hamate", "Scaphoid", "Lunate",
            "Triquetrum", "Pisiform", "Radius", "Ulna",
        ]
        # fmt: on

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x, y = batch
        pred = self.forward(x)
        loss = self.loss(pred, y)

        self.log(
            name="train_loss",
            value=round(loss.item(), 4),
            on_step=True,
            on_epoch=False,
            logger=True,
        )
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        x, y = batch
        outputs = self.forward(x)

        loss = self.loss(outputs, y)

        output_h, output_w = outputs.size(-2), outputs.size(-1)
        mask_h, mask_w = y.size(-2), y.size(-1)

        if output_h != mask_h or output_w != mask_w:
            outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear")

        outputs = torch.sigmoid(outputs)
        outputs = (outputs > 0.5).detach().cpu()
        masks = y.detach().cpu()

        y_true_f = masks.flatten(2)
        y_pred_f = outputs.flatten(2)
        intersection = torch.sum(y_true_f * y_pred_f, -1)

        eps = 0.0001
        self.dice = (2.0 * intersection + eps) / (
            torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) + eps
        )

        self.log(
            name="val_loss",
            value=loss,
            on_step=False,
            on_epoch=True,
            logger=True,
        )

    def on_validation_epoch_end(self):
        dices_per_class = torch.mean(self.dice, 0)
        dice_str = [
            f"{c:<12}: {d.item():.4f}" for c, d in zip(self.classes, dices_per_class)
        ]
        dice_str = "\n".join(dice_str)
        print(dice_str)

        avg_dice = torch.mean(dices_per_class).item()
        self.log(
            name="val_avg_dice",
            value=avg_dice,
            on_step=False,
            on_epoch=True,
            logger=True,
        )

    def configure_optimizers(self):
        optimizer = self.optimizer(params=self.parameters())
        scheduler = self.scheduler(
            optimizer=optimizer,
            total_steps=self.trainer.estimated_stepping_batches,
            max_lr=0.0001,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def optimizer_zero_grad(
        self, epoch: int, batch_idx: int, optimizer: Optimizer
    ) -> None:
        optimizer.zero_grad(set_to_none=True)
