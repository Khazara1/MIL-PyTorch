import os
import torch
import pandas as pd
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import numpy as np
import shap
import json
import yaml
import random
import joblib

from dataset import prepare_age_density_ml_dataframe
from model import (
    build_xgb_classifier,
    build_rf_classifier,
)
from metrics import BinaryMetricsCalculator
from matplotlib.ticker import MaxNLocator

# Paths and constants
TRAIN_CSV = "/users/scratch1/s189710/Multimodalny/data/data_buler/train_split_clean_cords_spot.csv"
VAL_CSV = "/users/scratch1/s189710/Multimodalny/data/data_buler/val_split_clean_cords_spot.csv"
TEST_CSV = "/users/scratch1/s189710/Multimodalny/data/data_buler/test_split_clean_cords_spot.csv"

MODEL_CONFIG_FILE = "config/model_config.yaml"

RUN_SHAP = True
RUN_BALANCED_SHAP = True

SHAP_BACKGROUND = None       # None = cały train
SHAP_EXPLAIN = 5000          # do summary plot
BALANCED_SHAP_REPEATS = 200 #wczesniej 200 bylo
BALANCED_SHAP_TOPK = 2

REMOVE_SPOTMAG = False

RUN_NAME = "a_test_spot_rf_200_resize_v1"

MODEL_NAME = "clinical_rf"   
# "clinical_xgb"
# "clinical_rf"

SEED = 42




BEST_PARAMS_ALL = {
    "clinical_xgb": {
        "n_estimators": 1497,
        "max_depth": 2,
        "learning_rate": 0.00020199257075716713,
        "subsample": 0.4452583229967411,
        "colsample_bytree": 0.5964296254566144,
        "min_child_weight": 3,
        "reg_lambda": 0.8690298190207,
        "reg_alpha": 0.003610322134211217,
    },
    
    "clinical_rf": {
        "n_estimators": 2998,
        "criterion": "entropy",
        "max_depth": 2,
        "min_samples_split": 95,
        "min_samples_leaf": 7,
        "max_features": "sqrt",
        "class_weight": "balanced",
    },
    
}

def compute_best_youden_threshold(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    J = tpr - fpr
    ix = np.argmax(J)
    best_thresh = thresholds[ix]
    return float(best_thresh), (fpr, tpr, thresholds), float(tpr[ix]), float(fpr[ix])

def compute_specificity_from_confusion_matrix(confusion_matrix):
    cm = confusion_matrix.cpu().numpy() if torch.is_tensor(confusion_matrix) else np.asarray(confusion_matrix)

    if cm.shape != (2, 2):
        return float("nan")

    tn, fp, fn, tp = cm.ravel()
    denom = tn + fp
    if denom == 0:
        return float("nan")
    return float(tn / denom)

def set_all_seeds(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

def load_yaml(yaml_path):
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)
    
def plot_roc_curve_with_best_threshold(roc_data, auroc_score=None):
    fpr, tpr, thresholds = roc_data

    J = tpr - fpr
    ix = np.argmax(J)
    best_thresh = thresholds[ix]
    best_tpr = tpr[ix]
    best_fpr = fpr[ix]

    plt.figure(figsize=(8, 6))

    label = "ROC Curve"
    if auroc_score is not None:
        label += f" (AUC = {auroc_score:.4f})"

    plt.plot(fpr, tpr, label=label)
    plt.scatter(best_fpr, best_tpr, marker="o", s=100, label=f"Best Threshold: {best_thresh:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random Guess")

    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    plt.annotate(
        f"Thresh={best_thresh:.2f}\nTPR={best_tpr:.2f}\nFPR={best_fpr:.2f}",
        xy=(best_fpr, best_tpr),
        xytext=(best_fpr + 0.1, best_tpr - 0.1),
        arrowprops=dict(shrink=0.05)
    )

    plt.savefig(f"{RUN_NAME}_val_ROC_curve.png")
    plt.close()

    return best_thresh





def build_model_from_name(model_name: str, params: dict, base_cfg: dict, y_train=None):
    cfg = dict(base_cfg)
    cfg.update(params)

   

    if model_name == "clinical_xgb":
        pos = int((y_train == 1).sum())
        neg = int((y_train == 0).sum())
        scale_pos_weight = float(neg / max(pos, 1))
        return build_xgb_classifier(cfg, seed=SEED, scale_pos_weight=scale_pos_weight)

   

    if model_name == "clinical_rf":
        return build_rf_classifier(cfg, seed=SEED)

    

    raise ValueError(f"Unknown MODEL_NAME: {model_name}")


def get_scores(model, X, model_name=None):
    if model_name == "clinical_xgb":
        try:
            return model.predict_proba(X)[:, 1]
        except Exception:
            raw = model.predict(X, output_margin=True)
            return 1.0 / (1.0 + np.exp(-raw))

    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-scores))

    return model.predict(X).astype(np.float32)


def evaluate_ml(model, X_np, y_true, threshold=0.5, model_name=None, save_outputs_prefix=None):
    metrics_calculator = BinaryMetricsCalculator(threshold=threshold)

    y_score = get_scores(model, X_np, model_name=model_name)

    outputs = torch.tensor(y_score, dtype=torch.float32)
    targets = torch.tensor(y_true, dtype=torch.float32)

    if save_outputs_prefix is not None:
        torch.save(outputs, f"{save_outputs_prefix}_outputs.pth")
        torch.save(targets, f"{save_outputs_prefix}_targets.pth")

    accuracy, f1_score, auprc, auroc, precision, recall, confusion_matrix = \
        metrics_calculator.calculate(outputs, targets)

    specificity = compute_specificity_from_confusion_matrix(confusion_matrix)

    y_true_np = targets.to(torch.int8).cpu().numpy()
    y_score_np = outputs.cpu().numpy()
    fpr, tpr, thresholds = roc_curve(y_true_np, y_score_np)
    roc_data = (fpr, tpr, thresholds)

    avg_loss = torch.tensor(float("nan"))

    return {
        "avg_loss": avg_loss,
        "accuracy": float(accuracy),
        "f1_score": float(f1_score),
        "auprc": float(auprc),
        "auroc": float(auroc),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "confusion_matrix": confusion_matrix,
        "roc_data": roc_data,
        "y_score": y_score_np,
        "y_true": y_true_np,
    }





def run_shap_analysis(model, X_background_df, X_explain_df, background_size=None, explain_size=500):
    if background_size is None:
        background_df = X_background_df.copy()
    else:
        bg_n = min(background_size, len(X_background_df))
        background_df = X_background_df.sample(n=bg_n, random_state=SEED).copy()

    ex_n = min(explain_size, len(X_explain_df))
    explain_df = X_explain_df.sample(n=ex_n, random_state=SEED).copy()

    feature_names = list(explain_df.columns)

    if MODEL_NAME == "clinical_xgb":
        background_np = background_df.to_numpy(dtype=np.float32)
        explain_np = explain_df.to_numpy(dtype=np.float32)

        predict_fn = lambda x: model.predict_proba(np.asarray(x, dtype=np.float32))[:, 1]
        explainer = shap.Explainer(predict_fn, background_np)
        shap_values = explainer(explain_np).values

    elif MODEL_NAME == "clinical_rf":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(explain_df)

    

    
    else:
        raise ValueError("Only XGB and RF supported")

    if isinstance(shap_values, list):
        shap_values = shap_values[1]


    shap_values = np.array(shap_values)

    if shap_values.ndim == 3:
        # (samples, features, classes)
        shap_values = shap_values[:, :, 1]

    elif shap_values.ndim == 2:
        pass

    else:
        raise ValueError(f"Unexpected SHAP shape: {shap_values.shape}")
    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(1, -1)

    # =========================
    # Group cumulative density features into one tissueden feature
    # =========================

    td_cols = ["td_ge_1", "td_ge_2", "td_ge_3", "td_ge_4"]

    age_idx = feature_names.index("age")
    td_idx = [feature_names.index(c) for c in td_cols]

    shap_values_grouped = np.column_stack([
        shap_values[:, age_idx],
        shap_values[:, td_idx].sum(axis=1),
    ])

    tissueden_vals = (
        explain_df[td_cols].sum(axis=1)
        .to_numpy(dtype=np.float32)
    )

    shap_input_grouped = np.column_stack([
        explain_df["age"].to_numpy(dtype=np.float32),
        tissueden_vals,
    ])

    feature_names_grouped = ["age", "tissueden"]

    importance_df = pd.DataFrame({
        "feature": feature_names_grouped,
        "mean_abs_shap": np.abs(shap_values_grouped).mean(axis=0),
        "mean_shap": shap_values_grouped.mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)

    importance_df.to_csv(
        f"{RUN_NAME}_shap_importance.csv",
        index=False
    )

    plt.figure(figsize=(8, 5))
    shap.summary_plot(
        shap_values_grouped,
        shap_input_grouped,
        feature_names=feature_names_grouped,
        plot_type="bar",
        show=False
    )
    plt.tight_layout()
    plt.savefig(f"{RUN_NAME}_shap_bar.png", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    shap.summary_plot(
        shap_values_grouped,
        shap_input_grouped,
        feature_names=feature_names_grouped,
        show=False
    )
    plt.tight_layout()
    plt.savefig(f"{RUN_NAME}_shap_summary.png", bbox_inches="tight")
    plt.close()
    
def run_balanced_shap_stability(model, X_df, y, repeats=100, top_k=5):
    if MODEL_NAME not in {"clinical_xgb", "clinical_rf"}:
        return

    feature_names = list(X_df.columns)

    if MODEL_NAME == "clinical_xgb":
        X_np = X_df.to_numpy(dtype=np.float32)
        predict_fn = lambda x: model.predict_proba(np.asarray(x, dtype=np.float32))[:, 1]
        explainer = shap.Explainer(predict_fn, X_np)
        use_call_api = True
    else:
        explainer = shap.TreeExplainer(model)
        use_call_api = False

    y = np.asarray(y)
    idx_0 = np.where(y == 0)[0]
    idx_1 = np.where(y == 1)[0]
    n_per_class = min(len(idx_0), len(idx_1))

    rows = []
    for i in range(repeats):
        rng = np.random.default_rng(SEED + i)

        sample_0 = rng.choice(idx_0, size=n_per_class, replace=False)
        sample_1 = rng.choice(idx_1, size=n_per_class, replace=False)

        idx = np.concatenate([sample_0, sample_1])
        rng.shuffle(idx)

        X_bal = X_df.iloc[idx].copy()

        if use_call_api:
            X_bal_np = X_bal.to_numpy(dtype=np.float32)
            shap_values = explainer(X_bal_np).values
        else:
            shap_values = explainer.shap_values(X_bal)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        shap_values = np.array(shap_values)

        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]
        elif shap_values.ndim == 2:
            pass
        else:
            raise ValueError(f"Unexpected SHAP shape: {shap_values.shape}")
        if shap_values.ndim == 1:
            shap_values = shap_values.reshape(1, -1)

        td_cols = ["td_ge_1", "td_ge_2", "td_ge_3", "td_ge_4"]

        age_idx = feature_names.index("age")
        td_idx = [feature_names.index(c) for c in td_cols]

        shap_values_grouped = np.column_stack([
            shap_values[:, age_idx],
            shap_values[:, td_idx].sum(axis=1),
        ])

        mean_abs_shap = np.abs(shap_values_grouped).mean(axis=0)

        row = {
            "repeat": i,
            "age": float(mean_abs_shap[0]),
            "tissueden": float(mean_abs_shap[1]),
        }
        rows.append(row)

    stability_df = pd.DataFrame(rows)
    stability_df.to_csv(f"{RUN_NAME}_balanced_shap_stability.csv", index=False)

    mean_importance = stability_df.drop(columns=["repeat"]).mean(axis=0).sort_values(ascending=False)
    top_features = mean_importance.head(top_k).index.tolist()

    ordered_features = top_features[::-1]
    data_to_plot = [stability_df[f].values for f in ordered_features]

    plt.figure(figsize=(5.2,4.8))
    plt.violinplot(data_to_plot, vert=False, showmeans=True, showextrema=False)
    label_map = {
        "age": "Age",
        "tissueden": "Tissue density"
    }

    ordered_labels = [label_map[f] for f in ordered_features]

    plt.yticks(
        np.arange(1,len(ordered_features)+1),
        ordered_labels
    )
    plt.xlabel("Mean absolute SHAP value")
    plt.grid(alpha=0.3)
    plt.gca().xaxis.set_major_locator(MaxNLocator(5))
    plt.tight_layout()
    plt.savefig(f"{RUN_NAME}_balanced_shap_violin.png", bbox_inches="tight")
    plt.savefig(f"{RUN_NAME}_balanced_shap_violin.eps", format="eps", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(5.2,4.8))
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

    label_map = {
        "age": "Age",
        "tissueden": "Tissue density"
    }

    ordered_labels = [label_map[f] for f in ordered_features]

    plt.yticks(
        np.arange(len(ordered_features))*spacing,
        ordered_labels
    )
    plt.xlabel("Mean absolute SHAP value")
    plt.grid(axis="x", alpha=0.3)
    plt.gca().xaxis.set_major_locator(MaxNLocator(5))
    plt.tight_layout()
    plt.savefig(f"{RUN_NAME}_balanced_shap_density.png", bbox_inches="tight")
    plt.savefig(f"{RUN_NAME}_balanced_shap_density.eps", format="eps", bbox_inches="tight")
    plt.close()

def main():
    set_all_seeds(SEED)
    print_spotmag_stats(TRAIN_CSV, "TRAIN (before)")
    print_spotmag_stats(VAL_CSV, "VAL (before)")
    print_spotmag_stats(TEST_CSV, "TEST (before)")
    model_cfg = load_yaml(MODEL_CONFIG_FILE)[MODEL_NAME]
    best_params = BEST_PARAMS_ALL[MODEL_NAME]

    print(f"REMOVE_SPOTMAG = {REMOVE_SPOTMAG}")

    X_train_df, y_train, num_stats = prepare_age_density_ml_dataframe(
        TRAIN_CSV,
        remove_spotmag_rows=REMOVE_SPOTMAG,
        num_stats=None,
    )
    print(f"TRAIN after filtering: {len(X_train_df)} samples")
    
    X_val_df, y_val, _ = prepare_age_density_ml_dataframe(
        VAL_CSV,
        remove_spotmag_rows=REMOVE_SPOTMAG,
        num_stats=num_stats,
    )
    print(f"VAL after filtering: {len(X_val_df)} samples")
    
    X_test_df, y_test, _ = prepare_age_density_ml_dataframe(
        TEST_CSV,
        remove_spotmag_rows=REMOVE_SPOTMAG,
        num_stats=num_stats,
    )
    print(f"TEST after filtering: {len(X_test_df)} samples")
    X_train_np = X_train_df.to_numpy(dtype=np.float32)
    X_val_np = X_val_df.to_numpy(dtype=np.float32)
    X_test_np = X_test_df.to_numpy(dtype=np.float32)

    y_train = np.asarray(y_train, dtype=np.int64)
    y_val = np.asarray(y_val, dtype=np.int64)
    y_test = np.asarray(y_test, dtype=np.int64)

    model = build_model_from_name(
        MODEL_NAME,
        best_params,
        model_cfg,
        y_train=y_train
    )

    model.fit(X_train_np, y_train)

    joblib.dump(model, f"{RUN_NAME}_model.joblib")
    with open(f"{RUN_NAME}_best_params.json", "w") as f:
        json.dump(best_params, f, indent=4)

    # =========================
    # 1) VAL: wybór threshold przez Youden
    # =========================
    y_val_score = get_scores(model, X_val_np, model_name=MODEL_NAME)
    best_thresh, val_roc_data, best_tpr, best_fpr = compute_best_youden_threshold(y_val, y_val_score)

    plot_roc_curve_with_best_threshold(val_roc_data, auroc_score=None)

    print(f"Best threshold from validation = {best_thresh:.6f}")

    # =========================
    # 2) TEST: metryki z thresholdem z VAL
    # =========================
    test_res = evaluate_ml(
        model=model,
        X_np=X_test_np,
        y_true=y_test,
        threshold=best_thresh,
        model_name=MODEL_NAME,
        save_outputs_prefix=RUN_NAME
    )

    confusion_matrix_df = pd.DataFrame(test_res["confusion_matrix"].cpu().numpy())

    results_df = pd.DataFrame([[ 
        MODEL_NAME,
        float(test_res["auprc"]),
        float(test_res["auroc"]),
        float(test_res["accuracy"]),
        float(test_res["f1_score"]),
        float(test_res["precision"]),
        float(test_res["recall"]),
        float(test_res["specificity"]),
        float(best_thresh),
    ]], columns=[
        "Model",
        "AUPRC",
        "AUROC",
        "Accuracy",
        "F1 score",
        "Precision",
        "Recall",
        "Specificity",
        "Threshold",
    ])

    results_df.to_csv(f"{RUN_NAME}_test_results.csv", index=False)
    confusion_matrix_df.to_csv(f"{RUN_NAME}_confusion_matrix.csv", index=False)

    with open(f"{RUN_NAME}_val_youden_threshold.json", "w") as f:
        json.dump({
            "threshold": float(best_thresh),
            "best_tpr": float(best_tpr),
            "best_fpr": float(best_fpr),
        }, f, indent=4)

    # =========================
    # 3) SHAP
    # =========================
    if RUN_SHAP:
        run_shap_analysis(
            model=model,
            X_background_df=X_train_df,
            X_explain_df=X_test_df,
            background_size=SHAP_BACKGROUND,
            explain_size=SHAP_EXPLAIN
        )

    # =========================
    # 4) Balanced repeated SHAP
    # =========================
    if RUN_BALANCED_SHAP and MODEL_NAME in {"clinical_xgb", "clinical_rf"}:
        run_balanced_shap_stability(
            model=model,
            X_df=X_test_df,
            y=y_test,
            repeats=BALANCED_SHAP_REPEATS,
            top_k=BALANCED_SHAP_TOPK
        )

    
def print_spotmag_stats(csv_path, name):
    df = pd.read_csv(csv_path, low_memory=False)

    total = len(df)
    with_spot = df["spot_mag"].notna().sum()
    without_spot = df["spot_mag"].isna().sum()

    print(f"\n=== {name} ===")
    print(f"Total rows: {total}")
    print(f"With spot_mag: {with_spot}")
    print(f"Without spot_mag: {without_spot}")

if __name__ == "__main__":
    main()



    


