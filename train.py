import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from image_patcher import ImagePatcher
from dataset import MILDataset
from metrics import BinaryMetricsCalculator
from ddp_utils import init_distributed, cleanup_distributed, gather_from_ranks
from data_utils import create_dataloader
from model_utils import build_model
from tqdm import tqdm
import torch.nn.functional as F
from time import gmtime, strftime
import albumentations as A
import cv2
import optuna
from optuna.samplers import TPESampler
import torch.distributed as dist
from functools import partial
import numpy as np
import random
import matplotlib.pyplot as plt

"""
TODO: HERE IMPORT YOUR DATASET AND MODEL CLASSES
"""
from dataset import YourDataset
from model import YourModelClass


SEED = 42

# Set seeds
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# TODO: Change those values
LOG_NAME = strftime("%Y-%m-%d_%H:%M:%S", gmtime()) # Log name used for saving model and logging to wandb
NUM_EPOCHS = 5
NUM_TRIALS = 8
AVG_METHOD = "macro"  # Averaging method for calculating metrics. Macro, micro or None (to get separate metrics for each class)
NUM_WORKERS = 16
BATCH_SIZE = 2
LR = 1e-4 # Learning rate
WEIGHT_DECAY = 1e-5 # Weight decay for optimizer


def save_first_n_images(dataloader, n=5, save_dir="debug_images"):
    os.makedirs(save_dir, exist_ok=True)
    count = 0
    for batch in dataloader:
        images = batch[0] if isinstance(batch, (list, tuple)) else batch
        for i in range(images.size(0)):
            img = images[i].detach().cpu().numpy()

            # If image has shape (C, H, W), transpose to (H, W, C)
            if img.shape[0] <= 4:
                img = np.transpose(img, (1, 2, 0))
                
            # Min-max normalization
            img_min = img.min()
            img_max = img.max()
            img_norm = (img - img_min) / (img_max - img_min + 1e-8)

            # If single channel, squeeze last dim
            if img_norm.shape[-1] == 1:
                img_norm = img_norm.squeeze(-1)
            
            plt.imsave(os.path.join(save_dir, f"img_{count+1}.png"), img_norm, cmap='gray' if img_norm.ndim == 2 else None)
            count += 1
            if count >= n:
                return


# Given a model and validation dataloader, evaluate the model performance on validation set
def validate(model, val_dl, criterion, is_ddp, rank, world_size, device):
    # Initialize validation dataloader with correct number of classes
    metrics_calculator = BinaryMetricsCalculator()

    val_loss = 0.0 # Track validation loss
    outputs_list = []
    targets_list = []

    if rank == 0:
        iterator = tqdm(val_dl, desc="Validation")
    else:
        iterator = val_dl

    model.eval()
    with torch.no_grad():
        for inputs, labels in iterator:
            # Move data to device
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                # Model forward pass
                logits = model(inputs).squeeze(1)

                # If binary classification use sigmoid and transform labels to float
                labels = labels.to(torch.float32)
        
                # Criterion is already set to BCEWithLogitsLoss so we are passing logits
                loss = criterion(logits, labels)

                # Use sigmoid just for metrics calculation
                outputs = F.sigmoid(logits)

            val_loss += loss.item()
            outputs_list.extend(outputs.detach().cpu().tolist())
            targets_list.extend(labels.detach().cpu().tolist())

    # Gather outputs, targets and losses from all ranks to calculate metrics on the whole validation set
    gathered_outputs = gather_from_ranks(outputs_list, is_ddp, world_size)
    gathered_targets = gather_from_ranks(targets_list, is_ddp, world_size)
    gathered_losses = gather_from_ranks(val_loss, is_ddp, world_size)

    if rank != 0:
        return None
    
    # Convert gathered lists to tensors and flatten them
    gathered_losses = torch.tensor(gathered_losses).flatten()
    gathered_outputs = torch.tensor(gathered_outputs).flatten(0, 1)
    gathered_targets = torch.tensor(gathered_targets).flatten(0, 1)

    # Get average validation loss
    avg_val_loss = torch.tensor(gathered_losses.mean() / len(val_dl))

    # Calculate validation metrics
    val_accuracy, val_f1_score, val_auprc, val_auroc, val_precision, val_recall, _ = metrics_calculator.calculate(gathered_outputs, gathered_targets)
    return avg_val_loss, val_accuracy, val_f1_score, val_auprc, val_auroc, val_precision, val_recall, gathered_outputs, gathered_targets


# Train the model
def train(model: torch.nn.Module, 
          train_dl: DataLoader, 
          val_dl: DataLoader, 
          train_sampler: Sampler, 
          criterion: nn.Module, 
          optimizer: torch.optim.Optimizer, 
          device: str, 
          num_epochs: int, 
          is_ddp: bool, 
          rank: int, 
          world_size: int, 
          log_name: str):
    # Initialize variables to track best model
    best_val_auprc = 0.0
    best_weights = model.state_dict()
    
    # Use correct metrics calculator for classification problem
    metrics_calculator = BinaryMetricsCalculator()
    
    for epoch in range(num_epochs):
        if rank == 0:
            print(f"Epoch {epoch+1}/{num_epochs} started")
        if is_ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        epoch_loss = 0.0 # Track training epoch loss
        outputs_list = []
        targets_list = []

        if rank == 0:
            iterator = tqdm(train_dl, desc=f"Epoch {epoch+1}/{num_epochs} - Training")
        else:
            iterator = train_dl

        scaler = torch.amp.GradScaler()

        model.train()
        for inputs, labels in iterator:
            optimizer.zero_grad() # Zero the gradients

            # Move data to device
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                # Model and criterion forward pass
                logits = model(inputs).squeeze(1)

                labels = labels.to(torch.float32)
                
                # Criterion is already set to BCEWithLogitsLoss so we are passing logits
                loss = criterion(logits, labels)

                # Use sigmoid just for metrics calculation
                outputs = F.sigmoid(logits)


            # Model optimization step
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            outputs_list.extend(outputs.detach().cpu().tolist())
            targets_list.extend(labels.detach().cpu().tolist())

        # Calculate train metrics
        avg_train_loss = torch.tensor(epoch_loss / len(train_dl))
        train_accuracy, train_f1_score, train_auprc, train_auroc, train_precision, train_recall, _ = metrics_calculator.calculate(outputs_list, targets_list)

        # Calculate validation metrics
        res = validate(
            model, 
            val_dl, 
            criterion,
            is_ddp=is_ddp,
            rank=rank,
            world_size=world_size,
            device=device)
        
        if res is not None:
            avg_val_loss, val_accuracy, val_f1_score, val_auprc, val_auroc, val_precision, val_recall, val_outputs, val_targets = res

            # Print epoch summary
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            print(f"\tTrain Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Train F1 Score: {train_f1_score:.4f}, Train AUPRC: {train_auprc:.4f}, Train AUROC: {train_auroc:.4f}, Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f}")
            print(f"\tVal Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Val F1 Score: {val_f1_score:.4f}, Val AUPRC: {val_auprc:.4f}, Val AUROC: {val_auroc:.4f}, Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}")

            if val_auprc > best_val_auprc:
                best_val_auprc = val_auprc
                torch.save(model.state_dict(), f"{log_name}_best.pth")
                best_weights = model.state_dict()


    print("Model training complete and saved.")
    model.load_state_dict(best_weights)
    torch.save(model.state_dict(), f"{log_name}_last.pth")

    return best_val_auprc


def main():
    # Setup distributed data processing
    is_ddp, local_rank, rank, world_size = init_distributed()

    if rank == 0:
        print(f"DDP initialized: is_ddp={is_ddp}, world_size={world_size}")
        print(f"Available GPUs: {torch.cuda.device_count()}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Define image transformations
    val_transform = A.Compose([
        A.Resize(224, 224),
        A.ToTensorV2(),
    ])
    
    train_transform = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5,
        ),

        A.Downscale(
            scale_range=(0.7, 0.9),
            interpolation_pair={
                "downscale": cv2.INTER_AREA,
                "upscale":   cv2.INTER_LINEAR,
            },
            p=0.3,
        ),

        A.Affine(
            scale=(0.95, 1.05),
            translate_percent={"x": 0.03, "y": 0.03},
            rotate=(-7, 7),
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
            fit_output=False,
            keep_ratio=True,
            p=0.5,
        ),

        A.ElasticTransform(
            alpha=20.0,
            sigma=5.0,
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.2,
        ),

        A.GridDistortion(
            num_steps=5,
            distort_limit=0.2,
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.2,
        ),

        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(0.03, 0.10),
            hole_width_range=(0.03, 0.10),
            fill=0,
            p=0.3,
        ),
    ], seed=SEED)


    # TODO: Those values are just an example
    your_train_args = "data/train_split.csv"
    your_val_args = "data/val_split.csv"
    your_model_args = []

    # Create dataset and dataloader
    train_dataset = YourDataset(your_train_args, transform=train_transform)
    train_dataloader, train_sampler = create_dataloader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, is_ddp=is_ddp, rank=rank, world_size=world_size, seed=SEED)

    val_dataset = YourDataset(your_val_args, transform=val_transform)
    val_dataloader, val_sampler = create_dataloader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, is_ddp=is_ddp, rank=rank, world_size=world_size, seed=SEED)

    save_first_n_images(train_dataloader, n=5, save_dir=f"train_images_rank{rank}")
    save_first_n_images(val_dataloader, n=5, save_dir=f"val_images_rank{rank}")

    # Initialize model, loss function, and optimizer
    model = build_model(YourModelClass, your_model_args, is_ddp=is_ddp, rank=rank, local_rank=local_rank, device=device)

    criterion = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Train the model
    best_val_auprc = train(
        model, 
        train_dataloader, 
        val_dataloader, 
        train_sampler, 
        criterion, 
        optimizer, 
        device, 
        num_epochs=NUM_EPOCHS,
        is_ddp=is_ddp,
        rank=rank,
        world_size=world_size, 
        log_name=LOG_NAME)
    
    # Distributed data processing cleanup
    if is_ddp:
        cleanup_distributed()

if __name__ == "__main__":
    main()