import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from image_patcher import ImagePatcher
from metrics import BinaryMetricsCalculator
from ddp_utils import init_distributed, cleanup_distributed, gather_from_ranks
from data_utils import create_dataloader
from model_utils import build_model
from tqdm import tqdm
import torch.nn.functional as F
from time import gmtime, strftime
import albumentations as A
import cv2
import torch.distributed as dist
from functools import partial
import numpy as np
import random
import matplotlib.pyplot as plt
import yaml
from logger import get_logger, log_metric
import wandb
import copy


"""
TODO: HERE IMPORT YOUR DATASET AND MODEL CLASSES
"""
from dataset import AllImagesDataset as DatasetClass
from model import StandardImageModel, AttentionMILModel

BACKBONE = "resnet18"
MODEL_CONFIG_FILE = "config/model_config.yaml" # Path to yaml file with model params
TRAIN_CONFIG_FILE = "config/train_config.yaml" # Path to yaml file with final training args

SEED = 42

DEBUG = True
LOG_WANDB = False

# Set seeds
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# TODO: Change those values
LOG_NAME = f'{BACKBONE}_final_{strftime("%Y-%m-%d_%H:%M:%S", gmtime())}' # Log name used for saving model and logging to wandb
NUM_EPOCHS = 5
NUM_TRIALS = 8
AVG_METHOD = "macro"  # Averaging method for calculating metrics. Macro, micro or None (to get separate metrics for each class)
NUM_WORKERS = 8

is_mil = BACKBONE.endswith("_mil")

if is_mil:
    ModelClass = AttentionMILModel
else:
    ModelClass = StandardImageModel


def load_yaml(yaml_path):
    with open(yaml_path, 'r') as file:
        return yaml.safe_load(file)


def save_first_n_images(dataloader, n=5, save_dir="debug_images"):
    os.makedirs(save_dir, exist_ok=True)
    count = 0
    for batch in dataloader:
        if is_mil:
            orig_imgs = batch[-1]
            # Handle both single image and list of images
            if isinstance(orig_imgs, torch.Tensor):
                orig_imgs = [orig_imgs]
            for img in orig_imgs:
                img = img.detach().cpu().numpy()
                if img.shape[0] <= 4:
                    img = np.transpose(img, (1, 2, 0))
                img_min = img.min()
                img_max = img.max()
                img_norm = (img - img_min) / (img_max - img_min + 1e-8)
                if img_norm.shape[-1] == 1:
                    img_norm = img_norm.squeeze(-1)
                plt.imsave(os.path.join(save_dir, f"img_{count+1}.png"), img_norm, cmap='gray' if img_norm.ndim == 2 else None)
                count += 1
                if count >= n:
                    return
        else:
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            for i in range(images.size(0)):
                img = images[i].detach().cpu().numpy()
                if img.shape[0] <= 4:
                    img = np.transpose(img, (1, 2, 0))
                img_min = img.min()
                img_max = img.max()
                img_norm = (img - img_min) / (img_max - img_min + 1e-8)
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
        for batch in iterator:
            # Check if data is for standard model or MIL
            if len(batch) == 2:
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)
            else:
                inputs, labels, masks, max_bag_length, instances_idx, instances_cords, orig_img = batch
                inputs = inputs.to(device)
                labels = labels.to(device)
                masks = masks.to(device)


            with torch.autocast(device_type="cuda", dtype=torch.float16):
                # Model forward pass
                if is_mil:
                    logits = model(inputs, masks, max_bag_length)
                else:
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
          log_name: str,
          logger: wandb.Run):
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
        for batch in iterator:
            # Check if data is for standard model or MIL
            if len(batch) == 2:
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)
            else:
                inputs, labels, masks, max_bag_length, instances_idx, instances_cords, orig_img = batch
                inputs = inputs.to(device)
                labels = labels.to(device)
                masks = masks.to(device)

            optimizer.zero_grad() # Zero the gradients

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                # Model and criterion forward pass
                if is_mil:
                    logits = model(inputs, masks, max_bag_length)
                else:
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

        gathered_outputs = gather_from_ranks(outputs_list, is_ddp, world_size)
        gathered_targets = gather_from_ranks(targets_list, is_ddp, world_size)
        gathered_losses = gather_from_ranks(epoch_loss, is_ddp, world_size)

        if rank == 0:   
            # Convert gathered lists to tensors and flatten them
            gathered_losses = torch.tensor(gathered_losses).flatten()
            gathered_outputs = torch.tensor(gathered_outputs).flatten(0, 1)
            gathered_targets = torch.tensor(gathered_targets).flatten(0, 1)

            # Calculate train metrics
            avg_train_loss = torch.tensor(gathered_losses.mean() / len(train_dl))
            train_accuracy, train_f1_score, train_auprc, train_auroc, train_precision, train_recall, _ = metrics_calculator.calculate(gathered_outputs, gathered_targets)

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

            if logger is not None:
                log_metric(logger, avg_train_loss, "train_loss")
                log_metric(logger, train_accuracy, "train_accuracy")
                log_metric(logger, train_f1_score, "train_f1_score")
                log_metric(logger, train_auprc, "train_auprc")
                log_metric(logger, train_auroc, "train_auroc")
                log_metric(logger, train_precision, "train_precision")
                log_metric(logger, train_recall, "train_recall")
                log_metric(logger, avg_val_loss, "val_loss")
                log_metric(logger, val_accuracy, "val_accuracy")
                log_metric(logger, val_f1_score, "val_f1_score")
                log_metric(logger, val_auprc, "val_auprc")
                log_metric(logger, val_auroc, "val_auroc")
                log_metric(logger, val_precision, "val_precision")
                log_metric(logger, val_recall, "val_recall")

            if val_auprc > best_val_auprc:
                best_val_auprc = val_auprc
                torch.save(model.state_dict(), f"{log_name}_best.pth")
                best_weights = copy.deepcopy(model.state_dict())


    print("Model training complete and saved.")
    model.load_state_dict(best_weights)
    torch.save(model.state_dict(), f"{log_name}_last.pth")

    return best_val_auprc


def objective(is_ddp, rank, world_size, local_rank, device):
    params = {}

    params = load_yaml(TRAIN_CONFIG_FILE)[BACKBONE]

    model_cfg = load_yaml(MODEL_CONFIG_FILE)[BACKBONE]

    if is_ddp:
        object_list = [params]
        dist.broadcast_object_list(object_list, src=0)
        params = object_list[0]

    # Define image transformations
    val_transform = A.Compose([
        A.ToTensorV2(),
    ])
    
    train_transform = A.Compose([
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
        A.ToTensorV2()
    ], seed=SEED)

    if is_mil:
        patcher = ImagePatcher(patch_size=params["patch_size"], overlap=params["overlap"])
    else:
        patcher = None


    # TODO: Those values are just an example
    your_train_args = "data/train_split_clean_cords_spot.csv"
    your_val_args = "data/val_split_clean_cords_spot.csv"
    your_model_args = model_cfg
    your_model_args["params"] = params

    # Create dataset and dataloader
    if is_mil:
        train_dataset = DatasetClass(your_train_args, transform=train_transform, image_patcher=patcher)
    else:
        train_dataset = DatasetClass(your_train_args, transform=train_transform)
    
    train_dataloader, train_sampler = create_dataloader(
        train_dataset, 
        batch_size=params['batch_size'], 
        shuffle=True, 
        sample_type="oversample", 
        num_workers=NUM_WORKERS, 
        is_ddp=is_ddp, 
        rank=rank, 
        world_size=world_size, 
        seed=SEED, 
        is_mil=is_mil
        )

    if is_mil:
        val_dataset = DatasetClass(your_val_args, transform=val_transform, image_patcher=patcher)
    else:
        val_dataset = DatasetClass(your_val_args, transform=val_transform)
    
    val_dataloader, val_sampler = create_dataloader(
        val_dataset, 
        batch_size=params['batch_size'], 
        shuffle=False, 
        sample_type=None, 
        num_workers=NUM_WORKERS, 
        is_ddp=is_ddp, 
        rank=rank, 
        world_size=world_size, 
        seed=SEED, 
        is_mil=is_mil
        )

    if DEBUG and rank == 0:
        save_first_n_images(train_dataloader, n=15, save_dir=f"train_images_{LOG_NAME}")
        save_first_n_images(val_dataloader, n=15, save_dir=f"val_images_{LOG_NAME}")

    if LOG_WANDB and rank == 0 and is_ddp:     # If distributed, only log from rank 0
        wandb_logger = get_logger(params)
        wandb_logger.log_model(path="model.py", name="attention_mil_model")
    elif LOG_WANDB and not is_ddp:    # Non-distributed logging
        wandb_logger = get_logger(params)
        wandb_logger.log_model(path="model.py", name="attention_mil_model")
    else:
        wandb_logger = None

    # Initialize model, loss function, and optimizer
    model = build_model(ModelClass, your_model_args, is_ddp=is_ddp, rank=rank, local_rank=local_rank, device=device)

    criterion = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

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
        log_name=LOG_NAME,
        logger=wandb_logger)


def main():
    # Setup distributed data processing
    is_ddp, local_rank, rank, world_size = init_distributed()

    if rank == 0:
        print(f"DDP initialized: is_ddp={is_ddp}, world_size={world_size}")
        print(f"Available GPUs: {torch.cuda.device_count()}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    objective(is_ddp, rank, world_size, local_rank, device)
    
    # Distributed data processing cleanup
    if is_ddp:
        cleanup_distributed()

if __name__ == "__main__":
    main()