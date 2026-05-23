import os
import yaml
import random
import numpy as np
import optuna

from time import gmtime, strftime
from optuna.samplers import TPESampler

import torch
from metrics import BinaryMetricsCalculator

from dataset import prepare_age_density_ml_dataframe
from model import (
    build_logreg_classifier,
    build_xgb_classifier,
    build_dt_classifier,
    build_rf_classifier,
    build_svm_classifier,
    build_nb_classifier,
    build_rbf_svm_classifier,
)

SEED = 42
MODEL_NAME = "clinical_svm_rbf"   # zmieniaj na:
# "clinical_logreg"
# "clinical_xgb"
# "clinical_dt"
# "clinical_rf"
# "clinical_svm"
# "clinical_nb"
# "clinical_svm_rbf"

OPTUNA_PARAMS_FILE = "config/optuna_config.yaml"
MODEL_CONFIG_FILE = "config/model_config.yaml"

LOG_NAME = f"{MODEL_NAME}_{strftime('%Y-%m-%d_%H:%M:%S', gmtime())}"
NUM_TRIALS = 200
REMOVE_SPOTMAG = True
THRESH = 0.5
TRAIN_CSV = "/users/scratch1/s189710/Multimodalny/data/data_buler/train_split_clean_cords_spot.csv"
VAL_CSV = "/users/scratch1/s189710/Multimodalny/data/data_buler/val_split_clean_cords_spot.csv"


def set_all_seeds(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_yaml(yaml_path):
    with open(yaml_path, "r") as file:
        return yaml.safe_load(file)


def build_model_from_name(model_name: str, params: dict, base_cfg: dict, y_train=None):
    cfg = dict(base_cfg)
    cfg.update(params)

    if model_name == "clinical_logreg":
        return build_logreg_classifier(cfg, seed=SEED)

    if model_name == "clinical_xgb":
        pos = int((y_train == 1).sum())
        neg = int((y_train == 0).sum())
        scale_pos_weight = float(neg / max(pos, 1))
        return build_xgb_classifier(cfg, seed=SEED, scale_pos_weight=scale_pos_weight)

    if model_name == "clinical_dt":
        return build_dt_classifier(cfg, seed=SEED)

    if model_name == "clinical_rf":
        return build_rf_classifier(cfg, seed=SEED)

    if model_name == "clinical_svm":
        return build_svm_classifier(cfg, seed=SEED)

    if model_name == "clinical_nb":
        return build_nb_classifier(cfg)
    
    if model_name == "clinical_svm_rbf":
        return build_rbf_svm_classifier(cfg, seed=SEED)

    raise ValueError(f"Unknown MODEL_NAME: {model_name}")


def get_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        # monotoniczne przekształcenie do [0,1], wygodne do raportowania
        return 1.0 / (1.0 + np.exp(-scores))

    # awaryjnie
    return model.predict(X).astype(np.float32)


def evaluate_binary(y_true, y_score, threshold=0.5):
    metrics_calculator = BinaryMetricsCalculator(threshold=threshold)

    y_true = torch.tensor(y_true, dtype=torch.float32)
    y_score = torch.tensor(y_score, dtype=torch.float32)

    accuracy, f1, auprc, auroc, precision, recall, confusion_matrix = \
        metrics_calculator.calculate(y_score, y_true)

    metrics = {
        "accuracy": float(accuracy),
        "f1": float(f1),
        "auprc": float(auprc),
        "auroc": float(auroc),
        "precision": float(precision),
        "recall": float(recall),
        "confusion_matrix": confusion_matrix,
    }

    return metrics


def objective(trial):
    search_space_cfg = load_yaml(OPTUNA_PARAMS_FILE)[MODEL_NAME]
    model_cfg = load_yaml(MODEL_CONFIG_FILE)[MODEL_NAME]

    params = {}
    for param_name, config in search_space_cfg.items():
        suggest_method = getattr(trial, config["type"])
        params[param_name] = suggest_method(name=param_name, **config["kwargs"])
    print(f"REMOVE_SPOTMAG = {REMOVE_SPOTMAG}")
    X_train, y_train, num_stats = prepare_age_density_ml_dataframe(
        TRAIN_CSV,
        remove_spotmag_rows=REMOVE_SPOTMAG,
        num_stats=None,
    )
    print(f"REMOVE_SPOTMAG = {REMOVE_SPOTMAG}")
    X_val, y_val, _ = prepare_age_density_ml_dataframe(
        VAL_CSV,
        remove_spotmag_rows=REMOVE_SPOTMAG,
        num_stats=num_stats,
    )

    X_train = X_train.to_numpy(dtype=np.float32)
    X_val = X_val.to_numpy(dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int64)
    y_val = np.asarray(y_val, dtype=np.int64)

    model = build_model_from_name(MODEL_NAME, params, model_cfg, y_train=y_train)
    model.fit(X_train, y_train)

    val_scores = get_scores(model, X_val)
    metrics = evaluate_binary(y_val, val_scores, threshold=THRESH)

    trial.set_user_attr("accuracy", float(metrics["accuracy"]))
    trial.set_user_attr("f1", float(metrics["f1"]))
    trial.set_user_attr("auprc", float(metrics["auprc"]))
    trial.set_user_attr("auroc", float(metrics["auroc"]))
    trial.set_user_attr("precision", float(metrics["precision"]))
    trial.set_user_attr("recall", float(metrics["recall"]))

    print(f"\nTrial {trial.number}")
    print("params:", params)
    print(
        f"VAL | "
        f"acc={metrics['accuracy']:.4f}, "
        f"f1={metrics['f1']:.4f}, "
        f"auprc={metrics['auprc']:.4f}, "
        f"auroc={metrics['auroc']:.4f}, "
        f"precision={metrics['precision']:.4f}, "
        f"recall={metrics['recall']:.4f}"
    )

    return -metrics["auprc"]


def main():
    set_all_seeds(SEED)

    sampler = TPESampler(seed=SEED)
    study = optuna.create_study(sampler=sampler)

    study.optimize(objective, n_trials=NUM_TRIALS)

    print("\n=== Best parameters ===")
    print(study.best_params)

    print("\n=== Best value ===")
    print(study.best_value)

    print("\n=== Best trial attrs ===")
    for k, v in study.best_trial.user_attrs.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()