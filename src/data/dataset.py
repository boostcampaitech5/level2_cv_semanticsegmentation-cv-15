import json
import os

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class XRayDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        split: str = "train",
        transforms: A.Compose = None,
    ):
        self.data_path = data_path
        self.split = split
        self.transforms = transforms
        self.file_name = []
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
        self.class_to_index = {v: i for i, v in enumerate(self.classes)}
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}

        if self.split == "train":
            with open(os.path.join(self.data_path, "train_label.json"), "r") as f:
                self.file_name = json.load(f)
        else:
            with open(os.path.join(self.data_path, "val_label.json"), "r") as f:
                self.file_name = json.load(f)

    def __len__(self) -> int:
        return len(self.file_name)

    def __getitem__(self, index):
        image_path = os.path.join(
            self.data_path, "train", self.file_name[index] + ".png"
        )
        image = cv2.imread(image_path)
        image = image / 255.0

        label_shape = tuple(image.shape[:2]) + (len(self.classes),)
        label = np.zeros(label_shape, dtype=np.uint8)

        with open(
            os.path.join(self.data_path, "annotation", self.file_name[index] + ".json"),
            "r",
        ) as f:
            annotations = json.load(f)
        annotations = annotations["annotations"]

        for anno in annotations:
            c = anno["label"]
            class_index = self.class_to_index[c]
            points = np.array(anno["points"])

            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [points], 1)
            label[..., class_index] = mask

        if self.transforms is not None:
            if self.split == "train":
                aug_img = self.transforms(image=image, mask=label)
                image = aug_img["image"]
                label = aug_img["mask"]
            else:
                aug_img = self.transforms(image=image)
                image = aug_img["image"]

        image = image.transpose(2, 0, 1)  # make channel first
        label = label.transpose(2, 0, 1)

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()
        return image, label
