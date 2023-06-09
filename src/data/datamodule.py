from pytorch_lightning import LightningDataModule


class XRayDataModule(LightningDataModule):
    def __init__(
        self,
        train_dataset,
        train_loader,
        val_dataset,
        val_loader,
        test_dataset,
        test_loader,
    ):
        super().__init__()

        self.train_dataset = train_dataset
        self.train_loader = train_loader
        self.val_dataset = val_dataset
        self.val_loader = val_loader
        self.test_dataset = test_dataset
        self.test_loader = test_loader

    def train_dataloader(self):
        return self.train_loader(dataset=self.train_dataset())

    def val_dataloader(self):
        return self.val_loader(dataset=self.val_dataset())

    def test_dataloader(self):
        return self.test_loader(dataset=self.test_dataset())
