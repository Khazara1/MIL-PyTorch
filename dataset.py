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
import re
import cv2

IMG_W = 1024
IMG_H = 2048


# pads the image to the defined width and height
def pad_to_fixed_size(image: np.ndarray, target_h: int = IMG_H, target_w: int = IMG_W):
    h, w = image.shape[:2]

    scale = min(target_h / h, target_w / w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    if image.ndim == 2:
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((target_h, target_w), dtype=resized.dtype)
    else:
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((target_h, target_w, image.shape[2]), dtype=resized.dtype)

    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2

    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas
    
# Deletes rows where spot_mag value is not NaN
def remove_spotmag(df: pd.DataFrame):
    df.drop(df[df.spot_mag.notna()].index, inplace=True)

# Deletes rows where spot_mag value is not NaN or rectangle
def remove_spotmag_type(df: pd.DataFrame):
    mask = df["pred_spot_mag_type"].isna() | (df["pred_spot_mag_type"] == "") | (df["pred_spot_mag_type"] == "rectangle")
    df.drop(df[~mask].index, inplace=True)

# Parses crop coordinates from string to tuple
def parse_crop_coords(crop_coords: str):
    vals = list(map(int, re.findall(r"\d+", str(crop_coords)))) #wyciaga nieprzerwane ciagi liczb z tekstu
    x1, y1, x2, y2 = vals
    return x1, y1, x2, y2

# Crops the image at specified coordinates
def crop_image_from_coords(image: np.ndarray, crop_coords: str):
    x1, y1, x2, y2 = parse_crop_coords(crop_coords)
    return image[y1:y2, x1:x2]

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
        self.classes_mapping = {"negative": 0, "suspicious": 1}
        
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
    

class YourDataset(Dataset):
    def __init__(self, dataset_csv: str, transform=None) -> None:
        super().__init__()

        # Prepare image transforms
        if transform is None:
            self.transform = A.Compose([
                A.ToTensorV2(),
                ])
        else:
            self.transform = transform

        self.df = pd.read_csv(dataset_csv)
        remove_spotmag(self.df)
        
        self.classes_mapping = {"negative": 0, "suspicious": 1}
        
        self.labels = torch.tensor(self.df["label"].map(lambda x: self.classes_mapping[x]).tolist())
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
            raise ValueError(f"Unsupported file format: {dcm_path}")

        image = pad_to_fixed_size(image)

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

        return image, label

#klasa do testu na np ResNet ze trzeba usunac spotmagi (w tej klasie sa wsyztkie zdj niewazne czy maja spotmagi i jakiego typu)
class AllImagesDataset(Dataset):
    def __init__(self, dataset_csv: str, transform=None, image_patcher=None) -> None:
        super().__init__()

        # Prepare image transforms
        if transform is None:
            self.transform = A.Compose([
                A.ToTensorV2(),
                ])
        else:
            self.transform = transform

        self.image_patcher = image_patcher

        self.df = pd.read_csv(dataset_csv)
        
        self.classes_mapping = {"negative": 0, "suspicious": 1}
        
        self.labels = torch.tensor(self.df["label"].map(lambda x: self.classes_mapping[x]).tolist())
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
            raise ValueError(f"Unsupported file format: {dcm_path}")

        image = pad_to_fixed_size(image)

        # Normalization
        image = np.array(image)
        image = image.astype(np.float32)

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

        if self.image_patcher is None:
            return image, label
        else:
            c, h, w = image.shape
            self.image_patcher.get_tiles(h, w)
            instances, instances_idx, instances_cords = self.image_patcher.convert_img_to_bag(image)
            return instances, label, instances_idx, instances_cords

# Dataset for training without spot_mags and with YOLO used for cropping the image
class CroppedDataset(Dataset):
    def __init__(self, dataset_csv: str, transform=None) -> None:
        super().__init__()

        # Prepare image transforms
        if transform is None:
            self.transform = A.Compose([
                A.ToTensorV2(),
                ])
        else:
            self.transform = transform

        self.df = pd.read_csv(dataset_csv)
        remove_spotmag(self.df)
        
        self.classes_mapping = {"negative": 0, "suspicious": 1}
        
        self.labels = torch.tensor(self.df["label"].map(lambda x: self.classes_mapping[x]).tolist())
        self.classes = list(self.classes_mapping.keys())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index) -> Tuple:
        dcm_path, label, crop_coords = self.df.iloc[index]["new_path"], self.df.iloc[index]["label"], self.df.iloc[index]["crop_coords"]
        label = self.classes_mapping[label] # Label from string to int
        label = torch.tensor(label, dtype=torch.long)

        if dcm_path.endswith(".dcm"):
            image = pydicom.dcmread(dcm_path).pixel_array
        else:
            raise ValueError(f"Unsupported file format: {dcm_path}")

        image = crop_image_from_coords(image, crop_coords)
        image = pad_to_fixed_size(image)

        # Normalization
        image = np.array(image)
        image = image.astype(np.float32)

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

        return image, label

# Dataset for training MIL model without spot_mags and with YOLO used for cropping the image
class CroppedMILDataset(Dataset):
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
        remove_spotmag(self.df)

        self.classes_mapping = {"negative": 0, "suspicious": 1}
        
        self.labels = torch.tensor(self.df["label"].map(lambda x: self.classes_mapping[x]))
        self.classes = list(self.classes_mapping.keys())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index) -> Tuple:
        dcm_path, label, crop_coords = self.df.iloc[index]["new_path"], self.df.iloc[index]["label"], self.df.iloc[index]["crop_coords"]
        label = self.classes_mapping[label] # Label from string to int
        label = torch.tensor(label, dtype=torch.long)

        if dcm_path.endswith(".dcm"):
            image = pydicom.dcmread(dcm_path).pixel_array
        else:
            image = plt.imread(dcm_path)

        image = crop_image_from_coords(image, crop_coords)

        # Normalization
        image = np.array(image)
        image = image.astype(np.float32)

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

# Dataset for training MIL model with rectangle spot_mags cropped and YOLO used for cropping the breast
class GetRectCroppedMILDataset(Dataset):
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
        remove_spotmag_type(self.df)

        self.classes_mapping = {"negative": 0, "suspicious": 1}
        
        self.labels = torch.tensor(self.df["label"].map(lambda x: self.classes_mapping[x]))
        self.classes = list(self.classes_mapping.keys())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index) -> Tuple:
        dcm_path, label, crop_coords = self.df.iloc[index]["new_path"], self.df.iloc[index]["label"], self.df.iloc[index]["crop_coords"]
        label = self.classes_mapping[label] # Label from string to int
        label = torch.tensor(label, dtype=torch.long)

        if dcm_path.endswith(".dcm"):
            image = pydicom.dcmread(dcm_path).pixel_array
        else:
            image = plt.imread(dcm_path)

        image = crop_image_from_coords(image, crop_coords)

        # Normalization
        image = np.array(image)
        image = image.astype(np.float32)

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