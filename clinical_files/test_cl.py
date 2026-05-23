import os
import torch
import pandas as pd
from tqdm import tqdm
from ddp_utils import init_distributed, cleanup_distributed, gather_from_ranks
from data_utils import create_dataloader
from model_utils import build_model
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import numpy as np
import shap

from dataset import ClinicalAgeDensityDataset
from model import ClinicalAgeDensityClassifier
from metrics import BinaryMetricsCalculator


# Paths and constants
TRAIN_CSV = "/users/scratch1/s189710/Multimodalny/data/data_buler/train_split_clean_cords_spot.csv"
TEST_CSV = "/users/scratch1/s189710/Multimodalny/data/data_buler/test_split_clean_cords_spot.csv"
CKPT_PATH = "/users/scratch1/s189710/Multimodalny/MIL-PyTorch/2026-04-10_03:55:53_best.pth"

RUN_NAME = "test_no_spot_v1"

BATCH_SIZE = 4
NUM_WORKERS = 16
THRESH = 0.5

RUN_SHAP = True
SHAP_BACKGROUND = 58257
SHAP_EXPLAIN = 58257

REMOVE_SPOTMAG = True

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


# Given a model and validation dataloader, evaluate the model performance on validation set
def validate(model, val_dl, criterion, is_ddp, rank, world_size, device, threshold=0.5):
    metrics_calculator = BinaryMetricsCalculator(threshold=threshold)

    val_loss = 0.0
    outputs_list = []
    targets_list = []

    if rank == 0:
        iterator = tqdm(val_dl, desc="Validation")
    else:
        iterator = val_dl

    model.eval()
    with torch.no_grad():
        for inputs, labels in iterator:
            labels = labels.to(device).to(torch.float32)

            clin_inputs = {
                "age": inputs["age"].to(device),
                "td": inputs["td"].to(device),
            }

            logits = model(clin_inputs).squeeze(1)
            loss = criterion(logits, labels)
            outputs = torch.sigmoid(logits)

            val_loss += loss.item()
            outputs_list.extend(outputs.detach().cpu().tolist())
            targets_list.extend(labels.detach().cpu().tolist())

    gathered_outputs = gather_from_ranks(outputs_list, is_ddp, world_size)
    gathered_targets = gather_from_ranks(targets_list, is_ddp, world_size)
    gathered_losses = gather_from_ranks(val_loss, is_ddp, world_size)

    if rank != 0:
        return None

    gathered_losses = torch.tensor(gathered_losses).flatten()
    gathered_outputs = torch.tensor(gathered_outputs).flatten(0, 1)
    gathered_targets = torch.tensor(gathered_targets).flatten(0, 1)

    torch.save(gathered_outputs, f"{RUN_NAME}_outputs.pth")
    torch.save(gathered_targets, f"{RUN_NAME}_targets.pth")
    
    avg_val_loss = gathered_losses.mean() / len(val_dl)

    val_accuracy, val_f1_score, val_auprc, val_auroc, val_precision, val_recall, confusion_matrix = \
        metrics_calculator.calculate(gathered_outputs, gathered_targets)

    y_true = gathered_targets.to(torch.int8).detach().cpu().numpy()
    y_score = gathered_outputs.detach().cpu().numpy()
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_data = (fpr, tpr, thresholds)

    return avg_val_loss, val_accuracy, val_f1_score, val_auprc, val_auroc, val_precision, val_recall, confusion_matrix, roc_data



    
def collect_shap_tensors(dataset, n_samples):
    n = min(n_samples, len(dataset))
    idxs = np.linspace(0, len(dataset) - 1, n, dtype=int)

    rows = []
    for i in idxs:
        inputs, _ = dataset[i]
        row = torch.tensor([
            inputs["age"].item(),
            float(inputs["td"].item()),
        ], dtype=torch.float32)
        rows.append(row)

    data = torch.stack(rows, dim=0).cpu().numpy()
    return data

def run_shap_analysis(model, dataset, device, background_size=64, explain_size=256):
    base_model = model.module if hasattr(model, "module") else model
    base_model.eval()

    background_data = collect_shap_tensors(dataset, background_size)
    explain_data = collect_shap_tensors(dataset, explain_size)

    def predict_fn(x_np):
        x_np = np.asarray(x_np, dtype=np.float32)

        age_np = x_np[:, 0:1]
        td_np = np.clip(np.round(x_np[:, 1]), 1, 4).astype(np.int64)

        inputs = {
            "age": torch.tensor(age_np, dtype=torch.float32, device=device),
            "td": torch.tensor(td_np, dtype=torch.long, device=device),
        }

        with torch.no_grad():
            logits = base_model(inputs).squeeze(1)
            probs = torch.sigmoid(logits)

        return probs.detach().cpu().numpy()

    explainer = shap.KernelExplainer(predict_fn, background_data)
    shap_values = explainer.shap_values(explain_data, nsamples=512)

    shap_values = np.array(shap_values)
    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(1, -1)
    if shap_values.ndim == 3:
        shap_values = shap_values[0]

    feature_names = ["age_scaled", "td_bin"]

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        "mean_shap": shap_values.mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)

    importance_df.to_csv(f"{RUN_NAME}_shap_importance.csv", index=False)

    plt.figure(figsize=(8, 5))
    shap.summary_plot(
        shap_values,
        explain_data,
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
        explain_data,
        feature_names=feature_names,
        show=False
    )
    plt.tight_layout()
    plt.savefig(f"{RUN_NAME}_shap_summary.png", bbox_inches="tight")
    plt.close()

def main():
    is_ddp, local_rank, rank, world_size = init_distributed()

    if rank == 0:
        print(f"DDP initialized: is_ddp={is_ddp}, world_size={world_size}")
        print(f"Available GPUs: {torch.cuda.device_count()}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"REMOVE_SPOTMAG = {REMOVE_SPOTMAG}")
    train_dataset = ClinicalAgeDensityDataset(
        TRAIN_CSV,
        remove_spotmag_rows=REMOVE_SPOTMAG
    )
    print(f"REMOVE_SPOTMAG = {REMOVE_SPOTMAG}")
    val_dataset = ClinicalAgeDensityDataset(
        TEST_CSV,
        num_stats=train_dataset.num_stats,
        remove_spotmag_rows=REMOVE_SPOTMAG
    )

    val_dataloader, val_sampler = create_dataloader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        sample_type=None,
        num_workers=NUM_WORKERS,
        is_ddp=is_ddp,
        rank=rank,
        world_size=world_size
    )

    model_args = {
        "hidden_dim": 128,
        "depth": 2,
        "dropout": 0.058348996146653224,
        "activation": "relu",
    }

    model = build_model(
        ClinicalAgeDensityClassifier,
        model_args,
        is_ddp=is_ddp,
        rank=rank,
        local_rank=local_rank,
        device=device
    )

    state_dict = torch.load(CKPT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    criterion = torch.nn.BCEWithLogitsLoss()

    res = validate(
        model,
        val_dataloader,
        criterion,
        is_ddp=is_ddp,
        rank=rank,
        world_size=world_size,
        device=device,
        threshold=THRESH
    )

    if res is not None:
        avg_val_loss, val_accuracy, val_f1_score, val_auprc, val_auroc, val_precision, val_recall, confusion_matrix, roc_curve_data = res

        best_thresh = plot_roc_curve_with_best_threshold(roc_curve_data, auroc_score=val_auroc)
        print(f"Best threshold is: {best_thresh}")

        results_df = pd.DataFrame([[
            avg_val_loss,
            val_accuracy,
            val_f1_score,
            val_auprc,
            val_auroc,
            val_precision,
            val_recall
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
                dataset=val_dataset,
                device=device,
                background_size=SHAP_BACKGROUND,
                explain_size=SHAP_EXPLAIN
            )

    if is_ddp:
        cleanup_distributed()


if __name__ == "__main__":
    main()


