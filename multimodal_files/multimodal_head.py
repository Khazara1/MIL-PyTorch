import os
import json
import math
import random
from time import gmtime, strftime
from typing import Dict, List, Tuple

import joblib
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from optuna.samplers import TPESampler
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt

from metrics import BinaryMetricsCalculator
from model import (
    build_logreg_classifier,
    build_xgb_classifier,
    build_dt_classifier,
    build_rf_classifier,
    build_svm_classifier,
    build_nb_classifier,
    build_rbf_svm_classifier,
)


# ============================================================
# USER CONFIG
# ============================================================
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# wybierz jeden z aliasów z DATASET_CONFIGS
EXPERIMENT_NAME = "convnext_base"

# heady do uruchomienia
HEADS_TO_RUN = [
    "weighted_sum",
    "logreg",
    "xgb",
    "dt",
    "rf",
    "svm",
    "nb",
    "svm_rbf",
    "mlp",
]

# features do fusion
USE_FEATURES = [
    "clinical_probability",
    "image_probability",
    # "image_logit",
]

LABEL_COL = "label"
THRESH = 0.5
NUM_WORKERS = 0   # dla maksymalnego determinizmu trzymaj 0
MLP_NUM_EPOCHS = 50
MLP_EARLY_STOPPING_PATIENCE = 7
NUM_TRIALS_PER_HEAD = 25

OUTPUT_ROOT = "multimodal_results"
PLOT_ROC = True
SAVE_OUTPUTS_AND_TARGETS = True

# jeśli chcesz bazować na yaml jak wcześniej, ustaw ścieżki i USE_YAML_CONFIGS=True
USE_YAML_CONFIGS = False
MODEL_CONFIG_FILE = "config/model_config.yaml"
OPTUNA_PARAMS_FILE = "config/optuna_config.yaml"


# ============================================================
# DATASET MAP
# Uzupełnij ścieżki dokładnie tak, jak masz u siebie.
# ============================================================
DATASET_CONFIGS = {
    "resnet18": {
        "train_csv": "/users/scratch1/s189710/Multimodalny/MIL-PyTorch/head_classifier_data/nenczak_resnet18__xgb_without_spotmag__train.csv",
        "val_csv": "/users/scratch1/s189710/Multimodalny/MIL-PyTorch/head_classifier_data/nenczak_resnet18__xgb_without_spotmag__val.csv",
        "test_csv": "/users/scratch1/s189710/Multimodalny/MIL-PyTorch/head_classifier_data/nenczak_resnet18__xgb_without_spotmag__test.csv",
    },
    "resnet50": {
        "train_csv": "/users/scratch1/s189710/Multimodalny/MIL-PyTorch/head_classifier_data/nenczak_resnet50__xgb_without_spotmag__train.csv",
        "val_csv": "/users/scratch1/s189710/Multimodalny/MIL-PyTorch/head_classifier_data/nenczak_resnet50__xgb_without_spotmag__val.csv",
        "test_csv": "/users/scratch1/s189710/Multimodalny/MIL-PyTorch/head_classifier_data/nenczak_resnet50__xgb_without_spotmag__test.csv",
    },
    "convnext_tiny": {
        "train_csv": "/users/scratch1/s189710/Multimodalny/MIL-PyTorch/head_classifier_data/nenczak_convnext_tiny__xgb_without_spotmag__train.csv",
        "val_csv": "/users/scratch1/s189710/Multimodalny/MIL-PyTorch/head_classifier_data/nenczak_convnext_tiny__xgb_without_spotmag__val.csv",
        "test_csv": "/users/scratch1/s189710/Multimodalny/MIL-PyTorch/head_classifier_data/nenczak_convnext_tiny__xgb_without_spotmag__test.csv",
    },
    "convnext_base": {
        "train_csv": "/users/scratch1/s189710/Multimodalny/MIL-PyTorch/head_classifier_data/nenczak_convnext_base__xgb_without_spotmag__train.csv",
        "val_csv": "/users/scratch1/s189710/Multimodalny/MIL-PyTorch/head_classifier_data/nenczak_convnext_base__xgb_without_spotmag__val.csv",
        "test_csv": "/users/scratch1/s189710/Multimodalny/MIL-PyTorch/head_classifier_data/nenczak_convnext_base__xgb_without_spotmag__test.csv",
    },
    "mil_resnet18_removed": {
        "train_csv": "/users/scratch1/s189710/Multimodalny/MIL-PyTorch/head_classifier_data/wilk_resnet18_removed__xgb_without_spotmag__train.csv",
        "val_csv": "/users/scratch1/s189710/Multimodalny/MIL-PyTorch/head_classifier_data/wilk_resnet18_removed__xgb_without_spotmag__val.csv",
        "test_csv": "/users/scratch1/s189710/Multimodalny/MIL-PyTorch/head_classifier_data/wilk_resnet18_removed__xgb_without_spotmag__test.csv",
    },
    "mil_resnet50_removed": {
        "train_csv": "/users/scratch1/s189710/Multimodalny/MIL-PyTorch/head_classifier_data/wilk_resnet50_removed__xgb_without_spotmag__train.csv",
        "val_csv": "/users/scratch1/s189710/Multimodalny/MIL-PyTorch/head_classifier_data/wilk_resnet50_removed__xgb_without_spotmag__val.csv",
        "test_csv": "/users/scratch1/s189710/Multimodalny/MIL-PyTorch/head_classifier_data/wilk_resnet50_removed__xgb_without_spotmag__test.csv",
    },
    "mil_resnet18_cut": {
        "train_csv": "/users/scratch1/s189710/Multimodalny/MIL-PyTorch/head_classifier_data/wilk_resnet18_cut__rf_trainval_with_spotmag_test_without_spotmag__train.csv",
        "val_csv": "/users/scratch1/s189710/Multimodalny/MIL-PyTorch/head_classifier_data/wilk_resnet18_cut__rf_trainval_with_spotmag_test_without_spotmag__val.csv",
        "test_csv": "/users/scratch1/s189710/Multimodalny/MIL-PyTorch/head_classifier_data/wilk_resnet18_cut__rf_trainval_with_spotmag_test_without_spotmag__test.csv",
    },
    "mil_resnet50_cut": {
        "train_csv": "/users/scratch1/s189710/Multimodalny/MIL-PyTorch/head_classifier_data/wilk_resnet50_cut__rf_trainval_with_spotmag_test_without_spotmag__train.csv",
        "val_csv": "/users/scratch1/s189710/Multimodalny/MIL-PyTorch/head_classifier_data/wilk_resnet50_cut__rf_trainval_with_spotmag_test_without_spotmag__val.csv",
        "test_csv": "/users/scratch1/s189710/Multimodalny/MIL-PyTorch/head_classifier_data/wilk_resnet50_cut__rf_trainval_with_spotmag_test_without_spotmag__test.csv",
    },
}


# ============================================================
# DETERMINISM
# ============================================================
def set_all_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

    # ważne dla części operacji cuBLAS
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ============================================================
# IO / HELPERS
# ============================================================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_yaml(yaml_path: str) -> dict:
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


def safe_float(x):
    if torch.is_tensor(x):
        return float(x.detach().cpu().item())
    return float(x)


def to_prob_from_logit(logit: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    p = 1.0 / (1.0 + np.exp(-logit))
    return np.clip(p, eps, 1.0 - eps)


def plot_roc_curve_with_threshold(roc_data, save_path: str, threshold: float, auroc_score=None):
    fpr, tpr, thresholds = roc_data

    idx = np.argmin(np.abs(thresholds - threshold))
    chosen_thresh = thresholds[idx]
    chosen_tpr = tpr[idx]
    chosen_fpr = fpr[idx]

    plt.figure(figsize=(8, 6))
    label = "ROC Curve"
    if auroc_score is not None:
        label += f" (AUC = {auroc_score:.4f})"

    plt.plot(fpr, tpr, label=label)
    plt.scatter(chosen_fpr, chosen_tpr, marker="o", s=100, label=f"Chosen Threshold: {chosen_thresh:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random Guess")

    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.annotate(
        f"Thresh={chosen_thresh:.2f}\nTPR={chosen_tpr:.2f}\nFPR={chosen_fpr:.2f}",
        xy=(chosen_fpr, chosen_tpr),
        xytext=(min(chosen_fpr + 0.1, 0.95), max(chosen_tpr - 0.1, 0.05)),
        arrowprops=dict(shrink=0.05),
    )
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


# ============================================================
# DATASET
# ============================================================
class FusionProbabilityDataset(Dataset):
    def __init__(self, csv_path: str, feature_cols: List[str], label_col: str = "label") -> None:
        super().__init__()
        self.csv_path = csv_path
        self.feature_cols = feature_cols
        self.label_col = label_col

        self.df = pd.read_csv(csv_path, low_memory=False)
        missing_cols = [c for c in feature_cols + [label_col] if c not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Brakuje kolumn w {csv_path}: {missing_cols}")

        X = self.df[feature_cols].copy()
        for col in feature_cols:
            X[col] = pd.to_numeric(X[col], errors="coerce")

        if X.isna().any().any():
            bad_counts = X.isna().sum().to_dict()
            raise ValueError(f"NaNy w cechach w {csv_path}: {bad_counts}")

        y = pd.to_numeric(self.df[label_col], errors="raise").astype(int)
        if y.isna().any():
            raise ValueError(f"NaNy w labelach w {csv_path}")

        self.X = X.to_numpy(dtype=np.float32)
        self.y = y.to_numpy(dtype=np.int64)
        self.labels = torch.tensor(self.y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        x = torch.tensor(self.X[index], dtype=torch.float32)
        y = torch.tensor(self.y[index], dtype=torch.long)
        return x, y


# ============================================================
# MODEL: WEIGHTED SUM HEAD
# ============================================================
class WeightedSumModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha_raw = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, 2]
        if x.ndim != 2 or x.shape[1] < 2:
            raise ValueError(f"WeightedSumModel expected input shape [B, 2], got {tuple(x.shape)}")

        p_clin = x[:, 0]
        p_img = x[:, 1]

        w_img = torch.sigmoid(self.alpha_raw)
        w_clin = 1.0 - w_img

        p = w_img * p_img + w_clin * p_clin
        p = torch.clamp(p, 1e-6, 1.0 - 1e-6)

        logits = torch.logit(p)
        return logits.unsqueeze(1)   # [B, 1]


# ============================================================
# MODEL: MLP HEAD
# ============================================================
class FusionMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 16,
        depth: int = 2,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be > 0")

        act = nn.ReLU if activation.lower() == "relu" else nn.GELU

        layers = []
        d = input_dim
        for _ in range(int(depth)):
            layers += [
                nn.Linear(d, int(hidden_dim)),
                act(),
                nn.Dropout(float(dropout)),
            ]
            d = int(hidden_dim)
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================
# MODEL BUILDERS FOR META-HEADS
# ============================================================
def get_ml_base_config(head_name: str) -> dict:
    if USE_YAML_CONFIGS:
        cfg_all = load_yaml(MODEL_CONFIG_FILE)
        key_map = {
            "logreg": "clinical_logreg",
            "xgb": "clinical_xgb",
            "dt": "clinical_dt",
            "rf": "clinical_rf",
            "svm": "clinical_svm",
            "nb": "clinical_nb",
            "svm_rbf": "clinical_svm_rbf",
        }
        return dict(cfg_all[key_map[head_name]])

    base_cfgs = {
        "logreg": {
            "solver": "lbfgs",
            "max_iter": 10000,
        },
        "xgb": {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "n_jobs": 1,
        },
        "dt": {},
        "rf": {
            "n_jobs": 1,
        },
        "svm": {
            "max_iter": 10000,
            "dual": False,
        },
        "nb": {},
        "svm_rbf": {
            "kernel": "rbf",
        },
    }
    return dict(base_cfgs[head_name])


def build_model_from_name(head_name: str, params: dict, y_train=None):
    cfg = get_ml_base_config(head_name)
    cfg.update(params)

    if head_name == "logreg":
        return build_logreg_classifier(cfg, seed=SEED)

    if head_name == "xgb":
        pos = int((y_train == 1).sum())
        neg = int((y_train == 0).sum())
        scale_pos_weight = float(neg / max(pos, 1))
        return build_xgb_classifier(cfg, seed=SEED, scale_pos_weight=scale_pos_weight)

    if head_name == "dt":
        return build_dt_classifier(cfg, seed=SEED)

    if head_name == "rf":
        return build_rf_classifier(cfg, seed=SEED)

    if head_name == "svm":
        return build_svm_classifier(cfg, seed=SEED)

    if head_name == "nb":
        return build_nb_classifier(cfg)

    if head_name == "svm_rbf":
        return build_rbf_svm_classifier(cfg, seed=SEED)

    raise ValueError(f"Unknown head_name: {head_name}")


# ============================================================
# SEARCH SPACES
# ============================================================
def suggest_ml_params(trial: optuna.Trial, head_name: str) -> dict:
    if head_name == "logreg":
        return {
            "C": trial.suggest_float("C", 1e-4, 1e3, log=True),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        }

    if head_name == "xgb":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 1500),
            "max_depth": trial.suggest_int("max_depth", 1, 6),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 3e-1, log=True),
            "subsample": trial.suggest_float("subsample", 0.4, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 10.0, log=True),
        }

    if head_name == "dt":
        return {
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
            "max_depth": trial.suggest_int("max_depth", 1, 12),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 128),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 64),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        }

    if head_name == "rf":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 2000),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
            "max_depth": trial.suggest_int("max_depth", 1, 16),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 128),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 64),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        }

    if head_name == "svm":
        return {
            "C": trial.suggest_float("C", 1e-5, 1e3, log=True),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        }

    if head_name == "nb":
        return {
            "var_smoothing": trial.suggest_float("var_smoothing", 1e-12, 1e-1, log=True),
        }

    if head_name == "svm_rbf":
        return {
            "C": trial.suggest_float("C", 1e-5, 1e3, log=True),
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        }

    raise ValueError(f"Unknown head_name: {head_name}")


def suggest_mlp_params(trial: optuna.Trial) -> dict:
    return {
        "hidden_dim": trial.suggest_categorical("hidden_dim", [4, 8, 16, 32, 64]),
        "depth": trial.suggest_int("depth", 1, 3),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "activation": trial.suggest_categorical("activation", ["relu", "gelu"]),
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-8, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [4, 8, 16, 32, 64, 128]),
    }


# ============================================================
# METRICS / EVAL
# ============================================================

def find_best_threshold(outputs: np.ndarray, targets: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(
        np.asarray(targets).astype(np.int8),
        np.asarray(outputs).astype(np.float32)
    )
    J = tpr - fpr
    ix = np.argmax(J)
    return float(thresholds[ix])

def evaluate_binary_outputs(outputs: np.ndarray, targets: np.ndarray, threshold: float = 0.5):
    metrics_calculator = BinaryMetricsCalculator(threshold=threshold)
    outputs_t = torch.tensor(outputs, dtype=torch.float32)
    targets_t = torch.tensor(targets, dtype=torch.float32)

    accuracy, f1, auprc, auroc, precision, recall, confusion_matrix = \
        metrics_calculator.calculate(outputs_t, targets_t)

    cm = confusion_matrix.detach().cpu().numpy()

    # binary confusion matrix:
    # [[TN, FP],
    #  [FN, TP]]
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        specificity = float("nan")

    fpr, tpr, thresholds = roc_curve(
        targets_t.to(torch.int8).cpu().numpy(),
        outputs_t.cpu().numpy()
    )
    roc_data = (fpr, tpr, thresholds)

    return {
        "accuracy": safe_float(accuracy),
        "f1": safe_float(f1),
        "auprc": safe_float(auprc),
        "auroc": safe_float(auroc),
        "precision": safe_float(precision),
        "recall": safe_float(recall),
        "specificity": float(specificity),
        "confusion_matrix": cm,
        "roc_data": roc_data,
    }


def get_scores(model, X: np.ndarray, head_name: str) -> np.ndarray:
    if head_name == "xgb":
        try:
            scores = model.predict_proba(X)[:, 1]
            return np.asarray(scores, dtype=np.float32)
        except Exception:
            raw = model.predict(X, output_margin=True)
            return to_prob_from_logit(np.asarray(raw, dtype=np.float32))

    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)[:, 1]
        return np.asarray(scores, dtype=np.float32)

    if hasattr(model, "decision_function"):
        decision = model.decision_function(X)
        return to_prob_from_logit(np.asarray(decision, dtype=np.float32))

    pred = model.predict(X).astype(np.float32)
    return np.asarray(pred, dtype=np.float32)


# ============================================================
# MLP TRAINING
# ============================================================
def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker if NUM_WORKERS > 0 else None,
        generator=g,
    )


@torch.no_grad()
def validate_mlp(model: nn.Module, val_dl: DataLoader, criterion: nn.Module, device: str):
    model.eval()
    val_loss = 0.0
    outputs_list = []
    targets_list = []

    for x, y in val_dl:
        x = x.to(device)
        y = y.to(device).to(torch.float32)

        logits = model(x).squeeze(1)
        loss = criterion(logits, y)
        probs = torch.sigmoid(logits)

        val_loss += loss.item()
        outputs_list.extend(probs.detach().cpu().tolist())
        targets_list.extend(y.detach().cpu().tolist())

    avg_val_loss = val_loss / max(len(val_dl), 1)
    metrics = evaluate_binary_outputs(np.array(outputs_list), np.array(targets_list), threshold=THRESH)
    metrics["avg_val_loss"] = float(avg_val_loss)
    metrics["outputs"] = np.array(outputs_list, dtype=np.float32)
    metrics["targets"] = np.array(targets_list, dtype=np.float32)
    return metrics


def train_mlp(
    model: nn.Module,
    train_dl: DataLoader,
    val_dl: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    num_epochs: int,
    patience: int,
    experiment_dir: str,
    verbose: bool = True,
):
    best_val_auprc = -float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_epoch = -1
    epochs_without_improve = 0
    train_history = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        outputs_list = []
        targets_list = []

        iterator = tqdm(train_dl, desc=f"Epoch {epoch+1}/{num_epochs} - Training", leave=False) if verbose else train_dl

        for x, y in iterator:
            x = x.to(device)
            y = y.to(device).to(torch.float32)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x).squeeze(1)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            probs = torch.sigmoid(logits)
            epoch_loss += loss.item()
            outputs_list.extend(probs.detach().cpu().tolist())
            targets_list.extend(y.detach().cpu().tolist())

        avg_train_loss = epoch_loss / max(len(train_dl), 1)
        train_metrics = evaluate_binary_outputs(np.array(outputs_list), np.array(targets_list), threshold=THRESH)
        val_metrics = validate_mlp(model, val_dl, criterion, device)

        row = {
            "epoch": epoch + 1,
            "train_loss": float(avg_train_loss),
            "train_accuracy": train_metrics["accuracy"],
            "train_f1": train_metrics["f1"],
            "train_auprc": train_metrics["auprc"],
            "train_auroc": train_metrics["auroc"],
            "train_precision": train_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "val_loss": val_metrics["avg_val_loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_f1": val_metrics["f1"],
            "val_auprc": val_metrics["auprc"],
            "val_auroc": val_metrics["auroc"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
        }
        train_history.append(row)

        if verbose:
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            print(
                f"\tTrain Loss: {row['train_loss']:.4f}, "
                f"Train Accuracy: {row['train_accuracy']:.4f}, "
                f"Train F1 Score: {row['train_f1']:.4f}, "
                f"Train AUPRC: {row['train_auprc']:.4f}, "
                f"Train AUROC: {row['train_auroc']:.4f}, "
                f"Train Precision: {row['train_precision']:.4f}, "
                f"Train Recall: {row['train_recall']:.4f}"
            )
            print(
                f"\tVal Loss: {row['val_loss']:.4f}, "
                f"Val Accuracy: {row['val_accuracy']:.4f}, "
                f"Val F1 Score: {row['val_f1']:.4f}, "
                f"Val AUPRC: {row['val_auprc']:.4f}, "
                f"Val AUROC: {row['val_auroc']:.4f}, "
                f"Val Precision: {row['val_precision']:.4f}, "
                f"Val Recall: {row['val_recall']:.4f}"
            )

        if val_metrics["auprc"] > best_val_auprc:
            best_val_auprc = val_metrics["auprc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        if epochs_without_improve >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
            break

    history_df = pd.DataFrame(train_history)
    history_df.to_csv(os.path.join(experiment_dir, "mlp_training_history.csv"), index=False)

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), os.path.join(experiment_dir, "best_mlp.pth"))

    return {
        "best_val_auprc": float(best_val_auprc),
        "best_epoch": int(best_epoch),
        "history_df": history_df,
    }


@torch.no_grad()
def predict_mlp(model: nn.Module, dl: DataLoader, device: str):
    model.eval()
    outputs_list = []
    targets_list = []

    for x, y in dl:
        x = x.to(device)
        y = y.to(device).to(torch.float32)
        logits = model(x).squeeze(1)
        probs = torch.sigmoid(logits)
        outputs_list.extend(probs.detach().cpu().tolist())
        targets_list.extend(y.detach().cpu().tolist())

    return np.array(outputs_list, dtype=np.float32), np.array(targets_list, dtype=np.float32)


# ============================================================
# OPTUNA OBJECTIVES
# ============================================================
def objective_ml(trial: optuna.Trial, head_name: str, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
    params = suggest_ml_params(trial, head_name)
    model = build_model_from_name(head_name, params, y_train=y_train)
    model.fit(X_train, y_train)

    val_scores = get_scores(model, X_val, head_name)
    metrics = evaluate_binary_outputs(val_scores, y_val, threshold=THRESH)

    trial.set_user_attr("accuracy", metrics["accuracy"])
    trial.set_user_attr("f1", metrics["f1"])
    trial.set_user_attr("auprc", metrics["auprc"])
    trial.set_user_attr("auroc", metrics["auroc"])
    trial.set_user_attr("precision", metrics["precision"])
    trial.set_user_attr("recall", metrics["recall"])

    return -metrics["auprc"]


def objective_mlp(trial: optuna.Trial, train_dataset: Dataset, val_dataset: Dataset, input_dim: int, experiment_dir: str):
    params = suggest_mlp_params(trial)

    train_dl = make_loader(train_dataset, batch_size=params["batch_size"], shuffle=True, seed=SEED)
    val_dl = make_loader(val_dataset, batch_size=params["batch_size"], shuffle=False, seed=SEED)

    model = FusionMLP(
        input_dim=input_dim,
        hidden_dim=params["hidden_dim"],
        depth=params["depth"],
        dropout=params["dropout"],
        activation=params["activation"],
    ).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])

    res = train_mlp(
        model=model,
        train_dl=train_dl,
        val_dl=val_dl,
        optimizer=optimizer,
        criterion=criterion,
        device=DEVICE,
        num_epochs=MLP_NUM_EPOCHS,
        patience=MLP_EARLY_STOPPING_PATIENCE,
        experiment_dir=experiment_dir,
        verbose=True,
    )

    trial.set_user_attr("best_epoch", res["best_epoch"])
    trial.set_user_attr("best_val_auprc", res["best_val_auprc"])

    return -res["best_val_auprc"]


# ============================================================
# RUN SINGLE HEAD
# ============================================================
def run_single_ml_head(
    head_name: str,
    train_csv: str,
    val_csv: str,
    test_csv: str,
    feature_cols: List[str],
    output_dir: str,
) -> dict:
    print(f"\n===== Running ML head: {head_name} =====")
    ensure_dir(output_dir)

    train_ds = FusionProbabilityDataset(train_csv, feature_cols, LABEL_COL)
    val_ds = FusionProbabilityDataset(val_csv, feature_cols, LABEL_COL)
    test_ds = FusionProbabilityDataset(test_csv, feature_cols, LABEL_COL)

    X_train, y_train = train_ds.X, train_ds.y
    X_val, y_val = val_ds.X, val_ds.y
    X_test, y_test = test_ds.X, test_ds.y

    sampler = TPESampler(seed=SEED)
    study = optuna.create_study(sampler=sampler)
    study.optimize(
        lambda trial: objective_ml(trial, head_name, X_train, y_train, X_val, y_val),
        n_trials=NUM_TRIALS_PER_HEAD,
        show_progress_bar=False,
    )

    best_params = study.best_params
    best_val_auprc = -float(study.best_value)

    with open(os.path.join(output_dir, "best_params.json"), "w") as f:
        json.dump(best_params, f, indent=4)

    trial_attrs = {
        "best_value": float(study.best_value),
        "best_params": best_params,
        "best_trial_user_attrs": study.best_trial.user_attrs,
    }
    with open(os.path.join(output_dir, "optuna_summary.json"), "w") as f:
        json.dump(trial_attrs, f, indent=4)

    final_model = build_model_from_name(head_name, best_params, y_train=y_train)
    final_model.fit(X_train, y_train)

    joblib.dump(final_model, os.path.join(output_dir, f"{head_name}_model.joblib"))

    val_scores = get_scores(final_model, X_val, head_name)
    best_thresh = find_best_threshold(val_scores, y_val)

    test_scores = get_scores(final_model, X_test, head_name)
    test_metrics = evaluate_binary_outputs(test_scores, y_test, threshold=best_thresh)

    if SAVE_OUTPUTS_AND_TARGETS:
        torch.save(torch.tensor(test_scores, dtype=torch.float32), os.path.join(output_dir, "test_outputs.pth"))
        torch.save(torch.tensor(y_test, dtype=torch.float32), os.path.join(output_dir, "test_targets.pth"))

    pd.DataFrame(test_metrics["confusion_matrix"]).to_csv(os.path.join(output_dir, "test_confusion_matrix.csv"), index=False)

    if PLOT_ROC:
        plot_roc_curve_with_threshold(
            test_metrics["roc_data"],
            save_path=os.path.join(output_dir, "test_roc_curve.png"),
            threshold=best_thresh,
            auroc_score=test_metrics["auroc"],
        )
  
        

    result_row = {
        "experiment_name": EXPERIMENT_NAME,
        "head_name": head_name,
        "feature_cols": ",".join(feature_cols),
        "train_csv": train_csv,
        "val_csv": val_csv,
        "test_csv": test_csv,
        "best_val_auprc": float(best_val_auprc),
        "threshold": float(best_thresh),
        "test_accuracy": test_metrics["accuracy"],
        "test_f1": test_metrics["f1"],
        "test_auprc": test_metrics["auprc"],
        "test_auroc": test_metrics["auroc"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_specificity": test_metrics["specificity"],
        "best_params_json": json.dumps(best_params),
    }

    pd.DataFrame([result_row]).to_csv(os.path.join(output_dir, "test_results.csv"), index=False)

    print(f"Best params ({head_name}): {best_params}")
    print(
        f"TEST | acc={result_row['test_accuracy']:.4f}, "
        f"f1={result_row['test_f1']:.4f}, "
        f"auprc={result_row['test_auprc']:.4f}, "
        f"auroc={result_row['test_auroc']:.4f}, "
        f"precision={result_row['test_precision']:.4f}, "
        f"recall={result_row['test_recall']:.4f}"
    )

    return result_row


def run_single_mlp_head(
    train_csv: str,
    val_csv: str,
    test_csv: str,
    feature_cols: List[str],
    output_dir: str,
) -> dict:
    head_name = "mlp"
    print(f"\n===== Running NN head: {head_name} =====")
    ensure_dir(output_dir)

    train_ds = FusionProbabilityDataset(train_csv, feature_cols, LABEL_COL)
    val_ds = FusionProbabilityDataset(val_csv, feature_cols, LABEL_COL)
    test_ds = FusionProbabilityDataset(test_csv, feature_cols, LABEL_COL)

    input_dim = train_ds.X.shape[1]

    sampler = TPESampler(seed=SEED)
    study = optuna.create_study(sampler=sampler)
    study.optimize(
        lambda trial: objective_mlp(trial, train_ds, val_ds, input_dim, output_dir),
        n_trials=NUM_TRIALS_PER_HEAD,
        show_progress_bar=False,
    )

    best_params = study.best_params
    best_val_auprc = -float(study.best_value)

    with open(os.path.join(output_dir, "best_params.json"), "w") as f:
        json.dump(best_params, f, indent=4)

    trial_attrs = {
        "best_value": float(study.best_value),
        "best_params": best_params,
        "best_trial_user_attrs": study.best_trial.user_attrs,
    }
    with open(os.path.join(output_dir, "optuna_summary.json"), "w") as f:
        json.dump(trial_attrs, f, indent=4)

    train_dl = make_loader(train_ds, batch_size=best_params["batch_size"], shuffle=True, seed=SEED)
    val_dl = make_loader(val_ds, batch_size=best_params["batch_size"], shuffle=False, seed=SEED)
    test_dl = make_loader(test_ds, batch_size=best_params["batch_size"], shuffle=False, seed=SEED)

    model = FusionMLP(
        input_dim=input_dim,
        hidden_dim=best_params["hidden_dim"],
        depth=best_params["depth"],
        dropout=best_params["dropout"],
        activation=best_params["activation"],
    ).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"])

    train_res = train_mlp(
        model=model,
        train_dl=train_dl,
        val_dl=val_dl,
        optimizer=optimizer,
        criterion=criterion,
        device=DEVICE,
        num_epochs=MLP_NUM_EPOCHS,
        patience=MLP_EARLY_STOPPING_PATIENCE,
        experiment_dir=output_dir,
        verbose=True,
    )

    val_scores, val_targets = predict_mlp(model, val_dl, DEVICE)
    best_thresh = find_best_threshold(val_scores, val_targets)

    test_scores, test_targets = predict_mlp(model, test_dl, DEVICE)
    test_metrics = evaluate_binary_outputs(test_scores, test_targets, threshold=best_thresh)

    if SAVE_OUTPUTS_AND_TARGETS:
        torch.save(torch.tensor(test_scores, dtype=torch.float32), os.path.join(output_dir, "test_outputs.pth"))
        torch.save(torch.tensor(test_targets, dtype=torch.float32), os.path.join(output_dir, "test_targets.pth"))

    pd.DataFrame(test_metrics["confusion_matrix"]).to_csv(os.path.join(output_dir, "test_confusion_matrix.csv"), index=False)

    if PLOT_ROC:
        plot_roc_curve_with_threshold(
            test_metrics["roc_data"],
            save_path=os.path.join(output_dir, "test_roc_curve.png"),
            threshold=best_thresh,
            auroc_score=test_metrics["auroc"],
        )
   
        

    result_row = {
        "experiment_name": EXPERIMENT_NAME,
        "head_name": head_name,
        "feature_cols": ",".join(feature_cols),
        "train_csv": train_csv,
        "val_csv": val_csv,
        "test_csv": test_csv,
        "best_val_auprc": float(best_val_auprc),
        "best_epoch": int(train_res["best_epoch"]),
        "threshold": float(best_thresh),
        "test_accuracy": test_metrics["accuracy"],
        "test_f1": test_metrics["f1"],
        "test_auprc": test_metrics["auprc"],
        "test_auroc": test_metrics["auroc"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_specificity": test_metrics["specificity"],
        "best_params_json": json.dumps(best_params),
    }

    pd.DataFrame([result_row]).to_csv(os.path.join(output_dir, "test_results.csv"), index=False)

    print(f"Best params ({head_name}): {best_params}")
    print(
        f"TEST | acc={result_row['test_accuracy']:.4f}, "
        f"f1={result_row['test_f1']:.4f}, "
        f"auprc={result_row['test_auprc']:.4f}, "
        f"auroc={result_row['test_auroc']:.4f}, "
        f"precision={result_row['test_precision']:.4f}, "
        f"recall={result_row['test_recall']:.4f}"
    )

    return result_row


# ============================================================
# WEIGHTED SUM RUNNER
# ============================================================
def run_weighted_sum_head(train_csv, val_csv, test_csv, feature_cols, output_dir):
    head_name = "weighted_sum"
    print(f"===== Running head: {head_name} =====")
    ensure_dir(output_dir)

    train_ds = FusionProbabilityDataset(train_csv, feature_cols, LABEL_COL)
    val_ds = FusionProbabilityDataset(val_csv, feature_cols, LABEL_COL)
    test_ds = FusionProbabilityDataset(test_csv, feature_cols, LABEL_COL)

    train_dl = make_loader(train_ds, batch_size=64, shuffle=True, seed=SEED)
    val_dl = make_loader(val_ds, batch_size=64, shuffle=False, seed=SEED)
    test_dl = make_loader(test_ds, batch_size=64, shuffle=False, seed=SEED)

    model = WeightedSumModel().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.BCEWithLogitsLoss()

    best_val_auprc = -1
    best_state = None

    for epoch in range(100):
        model.train()
        for x, y in train_dl:
            x = x.to(DEVICE)
            y = y.to(DEVICE).float().unsqueeze(1)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        val_scores, val_targets = predict_mlp(model, val_dl, DEVICE)
        metrics = evaluate_binary_outputs(val_scores, val_targets)

        if metrics["auprc"] > best_val_auprc:
            best_val_auprc = metrics["auprc"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)

    val_scores, val_targets = predict_mlp(model, val_dl, DEVICE)
    best_thresh = find_best_threshold(val_scores, val_targets)

    test_scores, test_targets = predict_mlp(model, test_dl, DEVICE)
    test_metrics = evaluate_binary_outputs(test_scores, test_targets, threshold=best_thresh)
    print("\n===== BEST weighted_sum RESULTS =====")
    print(
        f"BEST VAL AUPRC: {best_val_auprc:.4f} | "
        f"TEST ACC: {test_metrics['accuracy']:.4f} | "
        f"TEST F1: {test_metrics['f1']:.4f} | "
        f"TEST AUPRC: {test_metrics['auprc']:.4f} | "
        f"TEST AUROC: {test_metrics['auroc']:.4f} | "
        f"TEST PREC: {test_metrics['precision']:.4f} | "
        f"TEST REC: {test_metrics['recall']:.4f} | "
        f"TEST SPEC: {test_metrics['specificity']:.4f}"
    )
    if PLOT_ROC:
        plot_roc_curve_with_threshold(
            test_metrics["roc_data"],
            save_path=os.path.join(output_dir, "test_roc_curve.png"),
            threshold=best_thresh,
            auroc_score=test_metrics["auroc"],
        )

    pd.DataFrame(test_metrics["confusion_matrix"]).to_csv(
        os.path.join(output_dir, "test_confusion_matrix.csv"),
        index=False
    )

    if SAVE_OUTPUTS_AND_TARGETS:
        torch.save(torch.tensor(test_scores, dtype=torch.float32), os.path.join(output_dir, "test_outputs.pth"))
        torch.save(torch.tensor(test_targets, dtype=torch.float32), os.path.join(output_dir, "test_targets.pth"))

    w_img = torch.sigmoid(model.alpha_raw).item()

    result_row = {
        "experiment_name": EXPERIMENT_NAME,
        "head_name": head_name,
        "best_val_auprc": float(best_val_auprc),
        "threshold": float(best_thresh),
        "test_accuracy": test_metrics["accuracy"],
        "test_f1": test_metrics["f1"],
        "test_auprc": test_metrics["auprc"],
        "test_auroc": test_metrics["auroc"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_specificity": test_metrics["specificity"],
        "w_image": w_img,
        "w_clinical": 1.0 - w_img,
    }

    pd.DataFrame([result_row]).to_csv(os.path.join(output_dir, "test_results.csv"), index=False)

    print(f"Learned weights -> image: {w_img:.4f}, clinical: {1-w_img:.4f}")

    return result_row


# ============================================================
# MAIN
# ============================================================
def main():
    set_all_seeds(SEED)

    if EXPERIMENT_NAME not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown EXPERIMENT_NAME={EXPERIMENT_NAME}. "
            f"Available: {list(DATASET_CONFIGS.keys())}"
        )

    cfg = DATASET_CONFIGS[EXPERIMENT_NAME]
    train_csv = cfg["train_csv"]
    val_csv = cfg["val_csv"]
    test_csv = cfg["test_csv"]

    for p in [train_csv, val_csv, test_csv]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Nie istnieje plik: {p}")

    timestamp = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
    run_root = os.path.join(OUTPUT_ROOT, f"{EXPERIMENT_NAME}_{timestamp}")
    ensure_dir(run_root)

    run_meta = {
        "seed": SEED,
        "device": DEVICE,
        "experiment_name": EXPERIMENT_NAME,
        "heads_to_run": HEADS_TO_RUN,
        "use_features": USE_FEATURES,
        "threshold": THRESH,
        "num_trials_per_head": NUM_TRIALS_PER_HEAD,
        "mlp_num_epochs": MLP_NUM_EPOCHS,
        "mlp_early_stopping_patience": MLP_EARLY_STOPPING_PATIENCE,
        "train_csv": train_csv,
        "val_csv": val_csv,
        "test_csv": test_csv,
    }
    with open(os.path.join(run_root, "run_config.json"), "w") as f:
        json.dump(run_meta, f, indent=4)

    print("=" * 80)
    print(f"Running experiment: {EXPERIMENT_NAME}")
    print(f"Train CSV: {train_csv}")
    print(f"Val CSV:   {val_csv}")
    print(f"Test CSV:  {test_csv}")
    print(f"Features:  {USE_FEATURES}")
    print(f"Heads:     {HEADS_TO_RUN}")
    print(f"Device:    {DEVICE}")
    print("=" * 80)

    all_results = []

    for head_name in HEADS_TO_RUN:
        head_dir = os.path.join(run_root, head_name)
        ensure_dir(head_dir)

        try:
            if head_name == "mlp":
                row = run_single_mlp_head(
                    train_csv=train_csv,
                    val_csv=val_csv,
                    test_csv=test_csv,
                    feature_cols=USE_FEATURES,
                    output_dir=head_dir,
                )
            elif head_name == "weighted_sum":
                row = run_weighted_sum_head(
                    train_csv, val_csv, test_csv, USE_FEATURES, head_dir
                )
            else:
                row = run_single_ml_head(
                    head_name=head_name,
                    train_csv=train_csv,
                    val_csv=val_csv,
                    test_csv=test_csv,
                    feature_cols=USE_FEATURES,
                    output_dir=head_dir,
                )
            row["status"] = "ok"
        except Exception as e:
            row = {
                "experiment_name": EXPERIMENT_NAME,
                "head_name": head_name,
                "status": "failed",
                "error": str(e),
            }
            print(f"[ERROR] Head {head_name} failed: {e}")

        all_results.append(row)
        pd.DataFrame(all_results).to_csv(os.path.join(run_root, "all_results_partial.csv"), index=False)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(run_root, "all_results.csv"), index=False)

    if "test_auprc" in results_df.columns:
        try:
            sorted_df = results_df.sort_values(by=["status", "test_auprc"], ascending=[True, False])
        except Exception:
            sorted_df = results_df
    else:
        sorted_df = results_df

    sorted_df.to_csv(os.path.join(run_root, "all_results_sorted.csv"), index=False)

    print("\n===== FINAL RESULTS =====")
    cols_to_show = [
        c for c in [
            "head_name",
            "status",
            "best_val_auprc",
            "test_accuracy",
            "test_f1",
            "test_auprc",
            "test_auroc",
            "test_precision",
            "test_recall",
            "best_params_json",
            "error",
        ]
        if c in sorted_df.columns
    ]
    print(sorted_df[cols_to_show].to_string(index=False))
    print(f"\nSaved results to: {run_root}")


if __name__ == "__main__":
    main()