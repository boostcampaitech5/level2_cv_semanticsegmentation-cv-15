import albumentations as A
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from data.dataset import XRayDataset


class XRayDataModule(LightningDataModule):
    def __init__(
        self,
        data_path: str,
        batch_size: int,
        shuffle: bool = True,
        split: str = "train",
        transforms: A.Compose = None,
    ):
        super().__init__()

        self.data_path = data_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.split = split
        self.transforms = transforms

    def train_dataloader(self):
        return DataLoader(
            dataset=XRayDataset(
                data_path=self.data_path,
                split="train",
                transforms=self.transforms,
            ),
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=6,
            prefetch_factor=2,
            persistent_workers=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=XRayDataset(data_path=self.data_path, split="val"),
            batch_size=2,
            shuffle=self.shuffle,
            num_workers=2,
            prefetch_factor=2,
            persistent_workers=True,
            pin_memory=True,
        )
