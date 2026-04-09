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
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
            orig_img = image.clone()
            c, h, w = image.shape
            self.image_patcher.get_tiles(h, w)
            instances, instances_idx, instances_cords = self.image_patcher.convert_img_to_bag(image)
            return instances, label, instances_idx, instances_cords, orig_img

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
        return instances, label, instances_idx, instances_cords, image

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
        return instances, label, instances_idx, instances_cords, image
    

NUM_COLS = ["age_at_study", "tissueden"]
CAT_COLS = ["ETHNIC_GROUP_DESC", "race"]

UNK = "UNK"
CD_REGEX = re.compile(r"^cd:\d+", flags=re.IGNORECASE)
UNK_SUBSTRINGS = [
    "unknown", "unreported", "unavailable", "not recorded",
    "not reported", "missing", "n/a", "na", "none", "null"
]

def normalize_cat(x) -> str:
    if pd.isna(x):
        return UNK
    s = str(x).strip()
    if s == "" or CD_REGEX.match(s):
        return UNK
    low = s.lower()
    for sub in UNK_SUBSTRINGS:
        if sub in low:
            return UNK
    return s

AGE_BINS = [40, 50, 60, 70, 80]

def age_to_bin(age: float) -> int:
    b = 0
    for thr in AGE_BINS:
        if age >= thr:
            b += 1
        else:
            break
    return b + 1  # 1..6







class ClinicalOnlyDataset(Dataset):
    def __init__(self, dataset_csv: str, cat2idx: dict = None, num_stats: dict = None) -> None:
        super().__init__()

        self.df = pd.read_csv(dataset_csv, low_memory=False)

       # remove_spotmag(self.df)
        
        self.classes_mapping = {"negative": 0, "suspicious": 1}
        self.labels = torch.tensor(
            self.df["label"].map(lambda x: self.classes_mapping[x]).values,
            dtype=torch.long
        )
        self.classes = list(self.classes_mapping.keys())

        if num_stats is None:
            age = pd.to_numeric(self.df["age_at_study"], errors="coerce")
            td = pd.to_numeric(self.df["tissueden"], errors="coerce")

            age_median = float(age.median(skipna=True))
            age_mean = float(age.fillna(age_median).mean())
            age_std = float(age.fillna(age_median).std(ddof=0))
            age_std = age_std if age_std > 1e-6 else 1.0

            td_median = float(td.median(skipna=True))

            self.num_stats = {
                "age_median": age_median,
                "age_mean": age_mean,
                "age_std": age_std,
                "td_median": td_median,
            }
        else:
            self.num_stats = num_stats

        if cat2idx is None:
            self.cat2idx = {}
            for col in CAT_COLS:
                vals = self.df[col].apply(normalize_cat).unique().tolist()
                vocab = [UNK] + sorted([v for v in vals if v != UNK])
                self.cat2idx[col] = {v: i for i, v in enumerate(vocab)}
        else:
            self.cat2idx = cat2idx

        self.cat_vocab_sizes = {col: len(self.cat2idx[col]) for col in CAT_COLS}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        label = torch.tensor(self.classes_mapping[row["label"]], dtype=torch.long)

        age = pd.to_numeric(row["age_at_study"], errors="coerce")
        if pd.isna(age):
            age = self.num_stats["age_median"]
        age = float(age)
        age_bin = age_to_bin(age)

        td = pd.to_numeric(row["tissueden"], errors="coerce")
        if pd.isna(td):
            td = self.num_stats["td_median"]
        td = float(td)
        td = max(1.0, min(4.0, td))
        td_bin = int(round(td))

        clin_num = torch.tensor([age_bin, td_bin], dtype=torch.long)

        eth = normalize_cat(row["ETHNIC_GROUP_DESC"])
        race = normalize_cat(row["race"])

        clin_cat = torch.tensor([
            self.cat2idx["ETHNIC_GROUP_DESC"].get(eth, 0),
            self.cat2idx["race"].get(race, 0),
        ], dtype=torch.long)

        inputs = {
            "clin_num": clin_num,
            "clin_cat": clin_cat,
        }
        return inputs, label
    
class ClinicalAgeDensityDataset(Dataset):
    def __init__(self, dataset_csv: str, num_stats: dict = None, remove_spotmag_rows: bool = True) -> None:
        super().__init__()

        self.df = pd.read_csv(dataset_csv, low_memory=False)

        if remove_spotmag_rows:
            remove_spotmag(self.df)

        self.classes_mapping = {"negative": 0, "suspicious": 1}
        self.labels = torch.tensor(
            self.df["label"].map(lambda x: self.classes_mapping[x]).values,
            dtype=torch.long
        )
        self.classes = list(self.classes_mapping.keys())

        if num_stats is None:
            age = pd.to_numeric(self.df["age_at_study"], errors="coerce")
            td = pd.to_numeric(self.df["tissueden"], errors="coerce")

            age_median = float(age.median(skipna=True))
            age_filled = age.fillna(age_median)

            age_min = float(age_filled.min())
            age_max = float(age_filled.max())
            if abs(age_max - age_min) < 1e-8:
                age_max = age_min + 1.0

            td_median = float(td.median(skipna=True))

            self.num_stats = {
                "age_median": age_median,
                "age_min": age_min,
                "age_max": age_max,
                "td_median": td_median,
            }
        else:
            self.num_stats = num_stats

        

    def __len__(self):
        return len(self.df)
    
    def _scale_age_01(self, age: float) -> float:
        age_min = self.num_stats["age_min"]
        age_max = self.num_stats["age_max"]
        age_scaled = (age - age_min) / (age_max - age_min)
        age_scaled = max(0.0, min(1.0, age_scaled))
        return float(age_scaled)

    

    
    
    def __getitem__(self, index):
        row = self.df.iloc[index]

        label = torch.tensor(self.classes_mapping[row["label"]], dtype=torch.long)

        age = pd.to_numeric(row["age_at_study"], errors="coerce")
        if pd.isna(age):
            age = self.num_stats["age_median"]
        age = float(age)

        td = pd.to_numeric(row["tissueden"], errors="coerce")
        if pd.isna(td):
            td = self.num_stats["td_median"]
        td = float(td)
        td = max(1.0, min(4.0, td))
        td_bin = int(round(td))

        age_scaled = self._scale_age_01(age)

        inputs = {
            "age": torch.tensor([age_scaled], dtype=torch.float32),
            "td": torch.tensor(td_bin, dtype=torch.long),
        }

        

        return inputs, label   
    
def prepare_clinical_tabular_dataframe(
    dataset_csv: str,
    remove_spotmag_rows: bool = False,
):
    df = pd.read_csv(dataset_csv, low_memory=False)

    if remove_spotmag_rows:
        remove_spotmag(df)

    classes_mapping = {"negative": 0, "suspicious": 1}
    y = df["label"].map(lambda x: classes_mapping[x]).astype(int)

    X = df[NUM_COLS + CAT_COLS].copy()

    X["age_at_study"] = pd.to_numeric(X["age_at_study"], errors="coerce")
    X["tissueden"] = pd.to_numeric(X["tissueden"], errors="coerce")

    for col in CAT_COLS:
        X[col] = X[col].apply(normalize_cat).astype(str)

    return X, y


def build_clinical_tabular_preprocessor(scale_numeric: bool = True):
    if scale_numeric:
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
    else:
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUM_COLS),
            ("cat", categorical_transformer, CAT_COLS),
        ]
    )

    return preprocessor


def prepare_age_density_ml_dataframe(
    dataset_csv: str,
    remove_spotmag_rows: bool = True,
    num_stats: dict = None,
):
    """
    Przygotowuje cechy do klasycznych modeli ML zgodnie z logiką:
    - age: min-max do [0,1] na statystykach train
    - tissueden: clamp do [1,4], round, cumulative one-hot
    """
    df = pd.read_csv(dataset_csv, low_memory=False)

    if remove_spotmag_rows:
        remove_spotmag(df)

    classes_mapping = {"negative": 0, "suspicious": 1}
    y = df["label"].map(lambda x: classes_mapping[x]).astype(int).to_numpy()

    age = pd.to_numeric(df["age_at_study"], errors="coerce")
    td = pd.to_numeric(df["tissueden"], errors="coerce")

    if num_stats is None:
        age_median = float(age.median(skipna=True))
        age_filled = age.fillna(age_median)

        age_min = float(age_filled.min())
        age_max = float(age_filled.max())
        if abs(age_max - age_min) < 1e-8:
            age_max = age_min + 1.0

        td_median = float(td.median(skipna=True))

        num_stats = {
            "age_median": age_median,
            "age_min": age_min,
            "age_max": age_max,
            "td_median": td_median,
        }

    age = age.fillna(num_stats["age_median"]).astype(float)
    age_scaled = (age - num_stats["age_min"]) / (num_stats["age_max"] - num_stats["age_min"])
    age_scaled = age_scaled.clip(0.0, 1.0)

    td = td.fillna(num_stats["td_median"]).astype(float)
    td = td.clip(1.0, 4.0).round().astype(int)

    X = pd.DataFrame({
        "age": age_scaled.astype(np.float32),
        "td_ge_1": (td >= 1).astype(np.float32),
        "td_ge_2": (td >= 2).astype(np.float32),
        "td_ge_3": (td >= 3).astype(np.float32),
        "td_ge_4": (td >= 4).astype(np.float32),
    })

    return X, y, num_stats
