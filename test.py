import os
import torch
import json
from torchvision.transforms import v2
from image_patcher import ImagePatcher
from dataset import MILDataset
from data_utils import collate_fn
from metrics import BinaryMetricsCalculator, MulticlassMetricsCalculator
from tqdm import tqdm
from argparse import ArgumentParser
import pandas as pd
from ddp_utils import init_distributed, cleanup_distributed, gather_from_ranks
from data_utils import create_dataloader
from model_utils import build_model
from torch.utils.data import random_split
import torch.nn.functional as F
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import numpy as np
import albumentations as A
import yaml
from typing import Dict


def parse_args():
    parser = ArgumentParser(description="Train Attention-based MIL Model")
    parser.add_argument("--data-csv", type=str, required=True, help="Path to the dataset csv file")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--num-epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate for optimizer")
    parser.add_argument("--patch-size", type=int, default=224, help="Size of image patches")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training")
    parser.add_argument("--overlap", type=float, default=0.5, help="Overlap size between patches")
    parser.add_argument("--attention-dim", type=int, default=128, help="Dimension of attention layer")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--log-wandb", action="store_true", help="Whether to log training to Weights & Biases")
    parser.add_argument("--class-selection", action="store_true", help="If used classes are defined based on classes.json config file")
    parser.add_argument("--sample-type", type=str, default=None, required=False, help="Method used for balancing the dataset. Can be 'oversample', 'undersample' or None")
    parser.add_argument("--output-dim", type=int, required=False, default=-1, help="Number of output neurons. Should be equal to n_classes for classification problem or 1 for binary classification.")
    parser.add_argument("--ckpt-path", type=str, required=False, default=None, help="Path to model weights used for evaluation")
    parser.add_argument("--thresh", type=float, required=False, default=0.5, help="Cutoff threshold value")
    return parser.parse_args()


# Parses train config defined as yaml file
def parse_test_config() -> Dict:
    with open("config/test_config.yaml", "r") as f:
        test_config = yaml.safe_load(f)
    return test_config


def plot_roc_curve_with_best_threshold(roc_data, auroc_score=None):
    """
    Rysuje krzywą ROC i zaznacza optymalny próg (Youden's J statistic).
    """
    fpr, tpr, thresholds = roc_data
    
    # Obliczanie statystyki Youdena (J = TPR - FPR)
    # Szukamy punktu, który jest najdalej od linii losowej (przekątnej)
    J = tpr - fpr
    ix = np.argmax(J)
    best_thresh = thresholds[ix]
    best_tpr = tpr[ix]
    best_fpr = fpr[ix]

    plt.figure(figsize=(8, 6))
    
    # Rysowanie krzywej
    label = 'ROC Curve'
    if auroc_score is not None:
        label += f' (AUC = {auroc_score:.4f})'
        
    plt.plot(fpr, tpr, label=label, color='blue')
    
    # Zaznaczenie najlepszego punktu
    plt.scatter(best_fpr, best_tpr, marker='o', color='red', s=100, label=f'Best Threshold: {best_thresh:.4f}')
    
    # Linia losowego zgadywania
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
    
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    # Dodatkowe opisy na wykresie
    plt.annotate(f'Thresh={best_thresh:.2f}\nTPR={best_tpr:.2f}\nFPR={best_fpr:.2f}', 
                 xy=(best_fpr, best_tpr), 
                 xytext=(best_fpr + 0.1, best_tpr - 0.1),
                 arrowprops=dict(facecolor='black', shrink=0.05))
    
    plt.savefig("ROC_curve.png")
    
    return best_thresh


# Given a model and validation dataloader, evaluate the model performance on validation set
def validate(model, val_dl, criterion, output_dim, is_ddp, rank, world_size, device, threshold=0.5):
    # Initialize validation dataloader with correct number of classes
    if output_dim == 1:
        metrics_calculator = BinaryMetricsCalculator(threshold=threshold)
    else:
        metrics_calculator = MulticlassMetricsCalculator(num_classes=output_dim)

    val_loss = 0.0 # Track validation loss
    outputs_list = []
    targets_list = []
    losses_list = []

    if rank == 0:
        iterator = tqdm(val_dl, desc="Validation")
    else:
        iterator = val_dl

    model.eval()
    with torch.no_grad():
        for features, labels, masks, bags_length, instances_idx, instances_cords in iterator:
            # Move data to device
            features = features.to(device)
            labels = labels.to(device)
            masks = masks.to(device)

            # Model forward pass
            outputs = model(features, masks, bags_length)

            # If binary classification use sigmoid, multiclass use softmax
            if output_dim == 1:
                outputs = F.sigmoid(outputs)
                labels = labels.to(torch.float32)
            else:
                # For multiclass, keep logits - softmax applied internally by metrics
                outputs = F.softmax(outputs, dim=-1)
        
            loss = criterion(outputs, labels)

            losses_list.append(loss.item())
            val_loss += loss.item()
            outputs_list.extend(outputs.detach().cpu().tolist())
            targets_list.extend(labels.detach().cpu().tolist())

    gathered_outputs = gather_from_ranks(outputs_list, is_ddp, world_size)
    gathered_targets = gather_from_ranks(targets_list, is_ddp, world_size)
    # gathered_losses = gather_from_ranks(losses_list, is_ddp, world_size)
    gathered_losses = gather_from_ranks(val_loss, is_ddp, world_size)

    if rank != 0:
        return None
    
    gathered_losses = torch.tensor(gathered_losses).flatten()
    gathered_outputs = torch.tensor(gathered_outputs).flatten(0, 1)
    gathered_targets = torch.tensor(gathered_targets).flatten(0, 1)

    # Save outputs and targets tensors as a backup
    torch.save(gathered_outputs, "outputs.pth")
    torch.save(gathered_targets, "targets.pth")

    avg_val_loss = gathered_losses.mean() / len(val_dl)

    val_accuracy, val_f1_score, val_auprc, val_auroc, val_precision, val_recall, confusion_matrix = metrics_calculator.calculate(gathered_outputs, gathered_targets)

    # Obliczenie ROC przy pomocy sklearn
    y_true = gathered_targets.to(torch.int8).detach().cpu().numpy()
    y_score = gathered_outputs.detach().cpu().numpy()
    
    # Funkcja zwraca: (fpr, tpr, thresholds)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_data = (fpr, tpr, thresholds)

    return avg_val_loss, val_accuracy, val_f1_score, val_auprc, val_auroc, val_precision, val_recall, confusion_matrix, roc_data

test_config = parse_test_config()

def main():
    # Setup distributed data processing
    is_ddp, local_rank, rank, world_size = init_distributed()

    if rank == 0:
        print(f"DDP initialized: is_ddp={is_ddp}, world_size={world_size}")
        print(f"Available GPUs: {torch.cuda.device_count()}")

    args = parse_args()
    device = test_config["device"]

    # Define image transformations
    transform = A.Compose([
        A.ToTensorV2(),
    ])

    # Create patcher used for splitting images into patches
    patcher = ImagePatcher(patch_size=test_config["patch_size"], overlap=test_config["overlap"])

    # Select subset of classes
    if args.class_selection:
        with open("config/classes.json", "r") as f:
            selected_classes = json.load(f)
    else:
        selected_classes = None

    val_dataset = MILDataset(dataset_csv=args.data_csv, image_patcher=patcher, dirs_with_classes=selected_classes, transform=transform)

    val_dataloader, val_sampler = create_dataloader(val_dataset, batch_size=test_config["batch_size"], shuffle=False, sample_type=None, num_workers=test_config["num_workers"], is_ddp=is_ddp, rank=rank, world_size=world_size)

    n_classes = len(val_dataset.classes)

    if test_config["output_dim"] == -1:
        if n_classes == 2:
            output_dim = 1
        else:
            output_dim = n_classes
    else:
        output_dim = test_config["output_dim"]

    if args.ckpt_path is not None:
        print(f"Using checkpoint from: {args.ckpt_path}")
        state_dict = torch.load(args.ckpt_path, weights_only=True, map_location=device)
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    else:
        state_dict = None

    # Initialize model, loss function, and optimizer
    model = build_model(output_dim=output_dim, att_dim=test_config["attention_dim"], is_ddp=is_ddp, rank=rank, local_rank=local_rank, device=device, state_dict=state_dict)
    model.load_state_dict(state_dict)

    if output_dim == 1:
        criterion = torch.nn.BCELoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    # Calculate validation metrics
    res = validate(
        model, 
        val_dataloader, 
        criterion,
        output_dim=output_dim,
        is_ddp=is_ddp,
        rank=rank,
        world_size=world_size,
        device=device,
        threshold=args.thresh)
    
    if res is not None:
        avg_val_loss, val_accuracy, val_f1_score, val_auprc, val_auroc, val_precision, val_recall, confusion_matrix, roc_curve = res

        best_thresh = plot_roc_curve_with_best_threshold(roc_curve)
        print(f"Best threshold is: {best_thresh}")

        # Convert metrics to pandas data frames
        results_df = pd.DataFrame([[
            avg_val_loss, 
            val_accuracy, 
            val_f1_score, 
            val_auprc, 
            val_auroc, 
            val_precision, 
            val_recall]], 
            columns=[
                "avg_val_loss", 
                "val_accuracy", 
                "val_f1_score", 
                "val_auprc", 
                "val_auroc", 
                "precision", 
                "recall"])
        confusion_matrix_df = pd.DataFrame(confusion_matrix)
        
        # Save results to csv files
        results_df.to_csv("test_results.csv")
        confusion_matrix_df.to_csv("confusion_matrix.csv")


    # Distributed data processing cleanup
    if is_ddp:
        cleanup_distributed()


if __name__ == "__main__":
    main()


