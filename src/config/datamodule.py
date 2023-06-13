from hydra.core.config_store import ConfigStore
from torch.utils.data import DataLoader

from src.config import partial_builds
from src.data.dataset import XRayDataset

BasicTrainDatasetConfig = partial_builds(
    XRayDataset,
    data_path="/opt/ml/level2_cv_semanticsegmentation-cv-15/data",
    split="train",
    transforms=None,
)

BasicValidationDatasetConfig = partial_builds(
    XRayDataset,
    data_path="/opt/ml/level2_cv_semanticsegmentation-cv-15/data",
    split="val",
    transforms=None,
)

BasicTestDatasetConfig = partial_builds(
    XRayDataset,
    data_path="/opt/ml/level2_cv_semanticsegmentation-cv-15/data",
    split="test",
    transforms=None,
)

BasicTrainDataloaderConfig = partial_builds(
    DataLoader,
    batch_size=8,
    shuffle=True,
    num_workers=6,
    prefetch_factor=2,
    persistent_workers=True,
    pin_memory=True,
)

BasicValidationDataloaderConfig = partial_builds(
    DataLoader,
    batch_size=2,
    num_workers=2,
    prefetch_factor=2,
    persistent_workers=True,
    pin_memory=True,
)

BasicTestDataloaderConfig = partial_builds(
    DataLoader,
    batch_size=2,
    num_workers=2,
    prefetch_factor=2,
    persistent_workers=True,
    pin_memory=True,
)


def _register_configs():
    cs = ConfigStore.instance()

    cs.store(
        group="train_dataset",
        name="basic_train_dataset",
        node=BasicTrainDatasetConfig,
    )
    cs.store(
        group="val_dataset",
        name="basic_val_dataset",
        node=BasicValidationDatasetConfig,
    )
    cs.store(
        group="test_dataset",
        name="basic_test_dataset",
        node=BasicTestDatasetConfig,
    )

    cs.store(
        group="train_loader",
        name="basic_train_loader",
        node=BasicTrainDataloaderConfig,
    )
    cs.store(
        group="val_loader",
        name="basic_val_loader",
        node=BasicValidationDataloaderConfig,
    )
    cs.store(
        group="test_loader",
        name="basic_test_loader",
        node=BasicTestDataloaderConfig,
    )
