import os
import json
import random
import joblib
import numpy as np
import pandas as pd
import torch
import shap
import matplotlib.pyplot as plt

from model import build_logreg_classifier


SEED = 42

TRAIN_CSV = "/users/scratch1/s189710/Multimodalny/MIL-PyTorch/head_classifier_data/nenczak_convnext_base__xgb_without_spotmag__train.csv"
VAL_CSV = "/users/scratch1/s189710/Multimodalny/MIL-PyTorch/head_classifier_data/nenczak_convnext_base__xgb_without_spotmag__val.csv"
TEST_CSV = "/users/scratch1/s189710/Multimodalny/MIL-PyTorch/head_classifier_data/nenczak_convnext_base__xgb_without_spotmag__test.csv"

RUN_NAME = "a_multimodal_shap_logreg_convnextbasexgb_v2"

FEATURE_COLS = [
    "clinical_probability",
    "image_probability",
]

LABEL_COL = "label"

BALANCED_SHAP_REPEATS = 200
BALANCED_SHAP_TOPK = 2

BEST_PARAMS_LOGREG = {
    # WSTAW tutaj parametry z best_params.json dla logreg
    "C": 0.2500241994484591,
    "class_weight": "balanced",
}


def set_all_seeds(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_xy(csv_path):
    df = pd.read_csv(csv_path, low_memory=False)

    missing = [c for c in FEATURE_COLS + [LABEL_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")

    X = df[FEATURE_COLS].copy()
    for c in FEATURE_COLS:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    if X.isna().any().any():
        raise ValueError(f"NaNs in features: {X.isna().sum().to_dict()}")

    y = pd.to_numeric(df[LABEL_COL], errors="raise").astype(int).to_numpy()

    return X.astype(np.float32), y


def build_logreg_head(params):
    cfg = {
        "solver": "lbfgs",
        "max_iter": 10000,
    }
    cfg.update(params)
    return build_logreg_classifier(cfg, seed=SEED)


def fix_shap_shape(shap_values):
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_values = np.array(shap_values)

    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]
    elif shap_values.ndim == 2:
        pass
    else:
        raise ValueError(f"Unexpected SHAP shape: {shap_values.shape}")

    return shap_values


def run_balanced_shap_stability(model, X_df, y, repeats=1000, top_k=2):
    feature_names = list(X_df.columns)
    label_map = {
        "clinical_probability": "Clinical model\nprobability",
        "image_probability": "Image model\nprobability",
    }
    background_np = X_df.to_numpy(dtype=np.float32)

    def predict_fn(x):
        x = np.asarray(x, dtype=np.float32)
        return model.predict_proba(x)[:, 1]

    explainer = shap.Explainer(predict_fn, background_np)

    y = np.asarray(y)
    idx_0 = np.where(y == 0)[0]
    idx_1 = np.where(y == 1)[0]

    n_per_class = min(len(idx_0), len(idx_1))

    print(f"Class 0 samples: {len(idx_0)}")
    print(f"Class 1 samples: {len(idx_1)}")
    print(f"Samples per balanced repeat: {2 * n_per_class}")

    rows = []

    for i in range(repeats):
        rng = np.random.default_rng(SEED + i)

        sample_0 = rng.choice(idx_0, size=n_per_class, replace=False)
        sample_1 = rng.choice(idx_1, size=n_per_class, replace=False)

        idx = np.concatenate([sample_0, sample_1])
        rng.shuffle(idx)

        X_bal = X_df.iloc[idx].copy()
        X_bal_np = X_bal.to_numpy(dtype=np.float32)

        shap_values = explainer(X_bal_np).values
        shap_values = fix_shap_shape(shap_values)

        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        row = {"repeat": i}
        for feat, val in zip(feature_names, mean_abs_shap):
            row[feat] = float(val)

        rows.append(row)

    stability_df = pd.DataFrame(rows)
    stability_df.to_csv(f"{RUN_NAME}_balanced_shap_stability.csv", index=False)

    mean_importance = (
        stability_df
        .drop(columns=["repeat"])
        .mean(axis=0)
        .sort_values(ascending=False)
    )

    mean_importance.to_csv(f"{RUN_NAME}_balanced_shap_mean_importance.csv")

    top_features = mean_importance.head(top_k).index.tolist()
    ordered_features = top_features[::-1]
    data_to_plot = [stability_df[f].values for f in ordered_features]
    ordered_labels = [label_map.get(f, f) for f in ordered_features]
    plt.figure(figsize=(5.2, 4.8))
    plt.violinplot(
        data_to_plot,
        vert=False,
        showmeans=True,
        showextrema=False,
        widths=0.9,
        bw_method=0.5,
    )
    plt.yticks(np.arange(1, len(ordered_features) + 1), ordered_labels)
    plt.xlabel("Mean absolute SHAP value")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RUN_NAME}_balanced_shap_violin.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{RUN_NAME}_balanced_shap_violin.eps", format="eps", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(5.2, 4.8))
    spacing = 1.2
    height_scale = 0.6

    for j, feat in enumerate(ordered_features):
        vals = stability_df[feat].values
        hist, bin_edges = np.histogram(vals, bins=30, density=True)
        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        if hist.max() > 0:
            hist = hist / hist.max() * height_scale

        y0 = j * spacing
        plt.plot(centers, hist + y0, linewidth=1.5)
        plt.fill_between(centers, y0, hist + y0, alpha=0.25)

    plt.yticks(np.arange(len(ordered_features)) * spacing, ordered_labels)
    plt.xlabel("Mean absolute SHAP value")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RUN_NAME}_balanced_shap_density.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{RUN_NAME}_balanced_shap_density.eps", format="eps", bbox_inches="tight")
    plt.close()


def main():
    set_all_seeds(SEED)

    X_train_df, y_train = load_xy(TRAIN_CSV)
    X_val_df, y_val = load_xy(VAL_CSV)
    X_test_df, y_test = load_xy(TEST_CSV)

    print("Train:", X_train_df.shape)
    print("Val:", X_val_df.shape)
    print("Test:", X_test_df.shape)
    print("Features:", FEATURE_COLS)

    model = build_logreg_head(BEST_PARAMS_LOGREG)
    model.fit(X_train_df.to_numpy(dtype=np.float32), y_train)

    joblib.dump(model, f"{RUN_NAME}_model.joblib")

    with open(f"{RUN_NAME}_best_params.json", "w") as f:
        json.dump(BEST_PARAMS_LOGREG, f, indent=4)

    run_balanced_shap_stability(
        model=model,
        X_df=X_test_df,
        y=y_test,
        repeats=BALANCED_SHAP_REPEATS,
        top_k=BALANCED_SHAP_TOPK,
    )


if __name__ == "__main__":
    main()