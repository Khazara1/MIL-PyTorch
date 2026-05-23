import os
import numpy as np
import pandas as pd
import random

from dataset import prepare_age_density_ml_dataframe
from model import build_rf_classifier, build_xgb_classifier

SEED = 42

TRAIN_CSV = "/users/scratch1/s189710/Multimodalny/data/data_buler/train_split_clean_cords_spot.csv"
VAL_CSV   = "/users/scratch1/s189710/Multimodalny/data/data_buler/val_split_clean_cords_spot.csv"
TEST_CSV  = "/users/scratch1/s189710/Multimodalny/data/data_buler/test_split_clean_cords_spot.csv"

OUTPUT_DIR = "clinical_probability_exports"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def set_all_seeds(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_paths_and_labels(dataset_csv: str, remove_spotmag_rows: bool) -> pd.DataFrame:
    """
    Wczytuje minimalny dataframe do eksportu:
    - new_path
    - label (string)
    - label_bin (0/1)
    Z tą samą logiką filtrowania spotmag co features.
    """
    df = pd.read_csv(dataset_csv, low_memory=False)

    if remove_spotmag_rows:
        df = df[df["spot_mag"].isna()].copy()

    classes_mapping = {"negative": 0, "suspicious": 1}

    out = df[["new_path", "label"]].copy()
    out["label_bin"] = out["label"].map(classes_mapping).astype(int)
    out = out.reset_index(drop=True)
    return out


def build_rf_model(seed: int = 42):
    best_params_rf = {
        "n_estimators": 2998,
        "criterion": 'entropy',
        "max_depth": 2,
        "min_samples_split": 95,
        "min_samples_leaf": 7,
        "max_features": "sqrt",
        "class_weight": 'balanced',
    }

    model = build_rf_classifier(best_params_rf, seed=seed)
    return model


def build_xgb_model(y_train: np.ndarray, seed: int = 42):
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = float(neg / max(pos, 1))

    best_params_xgb = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "n_jobs": 1,
        "n_estimators": 1497,
        "max_depth": 2,
        "learning_rate": 0.00020199257075716713,
        "subsample": 0.4452583229967411,
        "colsample_bytree": 0.5964296254566144,
        "min_child_weight": 3,
        "reg_lambda": 0.8690298190207,
        "reg_alpha": 0.003610322134211217,
    }

    model = build_xgb_classifier(
        best_params_xgb,
        seed=seed,
        scale_pos_weight=scale_pos_weight
    )
    return model


def export_split_predictions(
    model,
    split_name: str,
    split_csv: str,
    remove_spotmag_rows: bool,
    num_stats: dict,
    model_name: str,
):
    """
    Tworzy CSV z kolumnami:
    - new_path
    - label
    - label_bin
    - probability
    """
    X_df, y, _ = prepare_age_density_ml_dataframe(
        split_csv,
        remove_spotmag_rows=remove_spotmag_rows,
        num_stats=num_stats,
    )

    meta_df = load_paths_and_labels(
        split_csv,
        remove_spotmag_rows=remove_spotmag_rows,
    )

    X_np = X_df.to_numpy(dtype=np.float32)
    prob = model.predict_proba(X_np)[:, 1]

    if len(meta_df) != len(prob):
        raise ValueError(
            f"Niezgodność długości dla {model_name} / {split_name}: "
            f"meta_df={len(meta_df)}, prob={len(prob)}"
        )

    export_df = meta_df.copy()
    export_df["probability"] = prob.astype(np.float32)

    save_path = os.path.join(
        OUTPUT_DIR,
        f"{model_name}_{split_name}_probabilities.csv"
    )
    export_df.to_csv(save_path, index=False)
    print(f"Saved: {save_path} | n={len(export_df)}")


def run_rf_with_spotmag():
    """
    RF na danych ZE spotmagami:
    remove_spotmag_rows=False
    """
    model_name = "rf_with_spotmag"
    remove_spotmag_rows = False

    X_train_df, y_train, num_stats = prepare_age_density_ml_dataframe(
        TRAIN_CSV,
        remove_spotmag_rows=remove_spotmag_rows,
        num_stats=None,
    )

    X_train_np = X_train_df.to_numpy(dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int64)

    model = build_rf_model(seed=SEED)
    model.fit(X_train_np, y_train)

    export_split_predictions(
        model=model,
        split_name="train",
        split_csv=TRAIN_CSV,
        remove_spotmag_rows=remove_spotmag_rows,
        num_stats=num_stats,
        model_name=model_name,
    )

    export_split_predictions(
        model=model,
        split_name="val",
        split_csv=VAL_CSV,
        remove_spotmag_rows=remove_spotmag_rows,
        num_stats=num_stats,
        model_name=model_name,
    )

    export_split_predictions(
        model=model,
        split_name="test",
        split_csv=TEST_CSV,
        remove_spotmag_rows=remove_spotmag_rows,
        num_stats=num_stats,
        model_name=model_name,
    )


def run_xgb_without_spotmag():
    """
    XGB na danych BEZ spotmagów:
    remove_spotmag_rows=True
    """
    model_name = "xgb_without_spotmag"
    remove_spotmag_rows = True

    X_train_df, y_train, num_stats = prepare_age_density_ml_dataframe(
        TRAIN_CSV,
        remove_spotmag_rows=remove_spotmag_rows,
        num_stats=None,
    )

    X_train_np = X_train_df.to_numpy(dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int64)

    model = build_xgb_model(y_train=y_train, seed=SEED)
    model.fit(X_train_np, y_train)

    export_split_predictions(
        model=model,
        split_name="train",
        split_csv=TRAIN_CSV,
        remove_spotmag_rows=remove_spotmag_rows,
        num_stats=num_stats,
        model_name=model_name,
    )

    export_split_predictions(
        model=model,
        split_name="val",
        split_csv=VAL_CSV,
        remove_spotmag_rows=remove_spotmag_rows,
        num_stats=num_stats,
        model_name=model_name,
    )

    export_split_predictions(
        model=model,
        split_name="test",
        split_csv=TEST_CSV,
        remove_spotmag_rows=remove_spotmag_rows,
        num_stats=num_stats,
        model_name=model_name,
    )


def main():
    set_all_seeds(SEED)

    run_rf_with_spotmag()
    run_xgb_without_spotmag()

    print("\nGotowe.")
    print("Wygenerowane pliki są w folderze:", OUTPUT_DIR)


if __name__ == "__main__":
    main()