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
    build_logreg_classifier,
    build_xgb_classifier,
    build_dt_classifier,
    build_rf_classifier,
    build_svm_classifier,
    build_rbf_svm_classifier,
    build_nb_classifier,
)
from metrics import BinaryMetricsCalculator


# Paths and constants
TRAIN_CSV = "/users/scratch1/s189710/Multimodalny/data/data_buler/train_split_clean_cords_spot.csv"
TEST_CSV = "/users/scratch1/s189710/Multimodalny/data/data_buler/test_split_clean_cords_spot.csv"


MODEL_CONFIG_FILE = "config/model_config.yaml"

THRESH = 0.5

RUN_SHAP = False
SHAP_BACKGROUND = 58257
SHAP_EXPLAIN = 58257

REMOVE_SPOTMAG = True

RUN_NAME = "test_no_spot_xgb_weryfikacja_v1"
MODEL_NAME = "clinical_xgb"   # zmieniasz zależnie od eksperymentu
# "clinical_logreg"
# "clinical_xgb"
# "clinical_dt"
# "clinical_rf"
# "clinical_svm"
# "clinical_nb"
# "clinical_svm_rbf"
SEED = 42

BEST_PARAMS_ALL = {
    "clinical_logreg": {
        "C": 55.471420895759614,
        "class_weight": None,
    },
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
    "clinical_dt": {
        "criterion": "log_loss",
        "max_depth": 4,
        "min_samples_split": 4,
        "min_samples_leaf": 6,
        "class_weight": 'balanced',
    },
    "clinical_rf": {
        "n_estimators": 2304,
        "criterion": "log_loss",
        "max_depth": 4,
        "min_samples_split": 91,
        "min_samples_leaf": 15,
        "max_features": "sqrt",
        "class_weight": None,
    },
    "clinical_svm": {
        "C": 0.004436926015598806,
        "class_weight": None,
    },
    "clinical_svm_rbf": {
        "C": 0.0021408012971941815,
        "kernel": "rbf",
        "gamma": "auto",
        "class_weight": 'balanced',
    },
    "clinical_nb": {
        "var_smoothing": 0.00026383884554283255,
    },
}

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

    plt.savefig(f"{RUN_NAME}_ROC_curve.png")
    plt.close()

    return best_thresh





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

    if model_name == "clinical_svm_rbf":
        return build_rbf_svm_classifier(cfg, seed=SEED)

    if model_name == "clinical_nb":
        return build_nb_classifier(cfg)

    raise ValueError(f"Unknown MODEL_NAME: {model_name}")


def get_scores(model, X):
    if MODEL_NAME == "clinical_xgb":
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


def evaluate_ml(model, X_test_np, y_test, threshold=0.5):
    metrics_calculator = BinaryMetricsCalculator(threshold=threshold)

    y_score = get_scores(model, X_test_np)

    outputs = torch.tensor(y_score, dtype=torch.float32)
    targets = torch.tensor(y_test, dtype=torch.float32)

    torch.save(outputs, f"{RUN_NAME}_outputs.pth")
    torch.save(targets, f"{RUN_NAME}_targets.pth")

    val_accuracy, val_f1_score, val_auprc, val_auroc, val_precision, val_recall, confusion_matrix = \
        metrics_calculator.calculate(outputs, targets)

    y_true_np = targets.to(torch.int8).cpu().numpy()
    y_score_np = outputs.cpu().numpy()
    fpr, tpr, thresholds = roc_curve(y_true_np, y_score_np)
    roc_data = (fpr, tpr, thresholds)

    avg_val_loss = torch.tensor(float("nan"))

    return avg_val_loss, val_accuracy, val_f1_score, val_auprc, val_auroc, val_precision, val_recall, confusion_matrix, roc_data


def run_shap_analysis(model, X_df, background_size=64, explain_size=256):
    bg_n = min(background_size, len(X_df))
    ex_n = min(explain_size, len(X_df))

    bg_idx = np.linspace(0, len(X_df) - 1, bg_n, dtype=int)
    ex_idx = np.linspace(0, len(X_df) - 1, ex_n, dtype=int)

    background_df = X_df.iloc[bg_idx].copy()
    explain_df = X_df.iloc[ex_idx].copy()

    feature_names = list(explain_df.columns)

    if MODEL_NAME in {"clinical_xgb", "clinical_rf", "clinical_dt"}:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(explain_df)

    elif MODEL_NAME == "clinical_logreg":
        explainer = shap.LinearExplainer(model, background_df)
        shap_values = explainer.shap_values(explain_df)

    elif MODEL_NAME in {"clinical_svm", "clinical_svm_rbf", "clinical_nb"}:
        def predict_fn(x_np):
            x_np = np.asarray(x_np, dtype=np.float32)
            return get_scores(model, x_np)

        explainer = shap.KernelExplainer(
            predict_fn,
            background_df.to_numpy(dtype=np.float32)
        )
        shap_values = explainer.shap_values(
            explain_df.to_numpy(dtype=np.float32),
            nsamples=512
        )
    else:
        return

    shap_values = np.array(shap_values)
    if shap_values.ndim == 3:
        shap_values = shap_values[0]
    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(1, -1)

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        "mean_shap": shap_values.mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)

    importance_df.to_csv(f"{RUN_NAME}_shap_importance.csv", index=False)

    shap_input = explain_df.to_numpy(dtype=np.float32) if MODEL_NAME in {"clinical_svm", "clinical_svm_rbf", "clinical_nb"} else explain_df

    plt.figure(figsize=(8, 5))
    shap.summary_plot(
        shap_values,
        shap_input,
        feature_names=feature_names,
        plot_type="bar",
        show=False
    )
    plt.tight_layout()
    plt.savefig(f"{RUN_NAME}_shap_bar.png", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    shap.summary_plot(
        shap_values,
        shap_input,
        feature_names=feature_names,
        show=False
    )
    plt.tight_layout()
    plt.savefig(f"{RUN_NAME}_shap_summary.png", bbox_inches="tight")
    plt.close()

    

def main():
    set_all_seeds(SEED)

    model_cfg = load_yaml(MODEL_CONFIG_FILE)[MODEL_NAME]
    best_params = BEST_PARAMS_ALL[MODEL_NAME]
    print(f"REMOVE_SPOTMAG = {REMOVE_SPOTMAG}")
    X_train_df, y_train, num_stats = prepare_age_density_ml_dataframe(
        TRAIN_CSV,
        remove_spotmag_rows=REMOVE_SPOTMAG,
        num_stats=None,
    )
    print(f"REMOVE_SPOTMAG = {REMOVE_SPOTMAG}")
    X_test_df, y_test, _ = prepare_age_density_ml_dataframe(
        TEST_CSV,
        remove_spotmag_rows=REMOVE_SPOTMAG,
        num_stats=num_stats,
    )

    X_train_np = X_train_df.to_numpy(dtype=np.float32)
    X_test_np = X_test_df.to_numpy(dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int64)
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

    res = evaluate_ml(
        model=model,
        X_test_np=X_test_np,
        y_test=y_test,
        threshold=THRESH
    )

    avg_val_loss, val_accuracy, val_f1_score, val_auprc, val_auroc, val_precision, val_recall, confusion_matrix, roc_curve_data = res

    best_thresh = plot_roc_curve_with_best_threshold(roc_curve_data, auroc_score=float(val_auroc))
    print(f"Best threshold is: {best_thresh}")

    results_df = pd.DataFrame([[ 
        avg_val_loss.item() if torch.is_tensor(avg_val_loss) else avg_val_loss,
        float(val_accuracy),
        float(val_f1_score),
        float(val_auprc),
        float(val_auroc),
        float(val_precision),
        float(val_recall)
    ]], columns=[
        "avg_val_loss",
        "val_accuracy",
        "val_f1_score",
        "val_auprc",
        "val_auroc",
        "precision",
        "recall"
    ])

    confusion_matrix_df = pd.DataFrame(confusion_matrix.cpu().numpy())

    results_df.to_csv(f"{RUN_NAME}_test_results.csv", index=False)
    confusion_matrix_df.to_csv(f"{RUN_NAME}_confusion_matrix.csv", index=False)

    if RUN_SHAP:
        run_shap_analysis(
            model=model,
            X_df=X_test_df,
            background_size=SHAP_BACKGROUND,
            explain_size=SHAP_EXPLAIN
        )
    

    


if __name__ == "__main__":
    main()


