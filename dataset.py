import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import torchvision.transforms.v2 as v2
from typing import Tuple
from image_patcher import ImagePatcher
import os
import numpy as np
from PIL import Image
import albumentations as A
import pandas as pd
import pydicom
import matplotlib.pyplot as plt


class MILDataset(Dataset):
    def __init__(self, dataset_csv: str, image_patcher: ImagePatcher, dirs_with_classes: dict = None, transform=None) -> None:
        super().__init__()

        # Prepare image transforms
        if transform is None:
            self.transform = A.Compose([
                A.ToTensorV2(),
                ])
        else:
            self.transform = transform

        # Init image patcher
        self.image_patcher = image_patcher

        self.df = pd.read_csv(dataset_csv)
        self.classes_mapping = {label: idx for idx, label in enumerate(self.df["label"].unique())}
        
        self.labels = torch.tensor(self.df["label"].map(lambda x: self.classes_mapping[x]))
        self.classes = list(self.classes_mapping.keys())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index) -> Tuple:
        dcm_path, label = self.df.iloc[index]["new_path"], self.df.iloc[index]["label"]
        label = self.classes_mapping[label] # Label from string to int
        label = torch.tensor(label, dtype=torch.long)

        if dcm_path.endswith(".dcm"):
            image = pydicom.dcmread(dcm_path).pixel_array
        else:
            image = plt.imread(dcm_path)

        # Normalization
        image = np.array(image)
        image = image.astype(np.float32)

        if image.shape[-1] != 3:    # Check if image is RGB or GRAYSCALE
            image = np.expand_dims(image, axis=-1)      # Add channel dimension to grayscale image
            image = image.repeat(repeats=3, axis=-1)    # Grayscale to RGB
        image = (image - image.min()) / (image.max() - image.min())

        image = self.transform(image=image)["image"]

        # If transformation to Tensor was not applied by albumentations (p=0.9) apply it manually
        if isinstance(image, np.ndarray):
            image = torch.tensor(image)
            image = image.permute(2, 0, 1)

        # Scale to [0, 1] range
        image = image.to(torch.float32)

        c, h, w = image.shape
        self.image_patcher.get_tiles(h, w)
        instances, instances_idx, instances_cords = self.image_patcher.convert_img_to_bag(image)
        return instances, label, instances_idx, instances_cords