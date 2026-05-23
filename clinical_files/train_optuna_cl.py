import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from metrics import BinaryMetricsCalculator
from ddp_utils import init_distributed, cleanup_distributed, gather_from_ranks
from data_utils import create_dataloader
from model_utils import build_model
from tqdm import tqdm
import torch.nn.functional as F
from time import gmtime, strftime
import optuna
from optuna.samplers import TPESampler
import torch.distributed as dist
from functools import partial
import numpy as np
import random
import yaml

"""
TODO: HERE IMPORT YOUR DATASET AND MODEL CLASSES
"""
from dataset import ClinicalAgeDensityDataset as YourDataset
from model import ClinicalAgeDensityClassifier as YourModelClass

MODEL_NAME = "clinical"
OPTUNA_PARAMS_FILE = "config/optuna_config.yaml"
MODEL_CONFIG_FILE = "config/model_config.yaml"

SEED = 42



# Set seeds
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# TODO: Change those values
LOG_NAME = "clinical_" + strftime("%Y-%m-%d_%H:%M:%S", gmtime()) # Log name used for saving model and logging to wandb
NUM_EPOCHS = 15
NUM_TRIALS = 100
AVG_METHOD = "macro"  # Averaging method for calculating metrics. Macro, micro or None (to get separate metrics for each class)
NUM_WORKERS = 16

REMOVE_SPOTMAG = True  # True = usuwasz spotmagi


def load_yaml(yaml_path):
    with open(yaml_path, 'r') as file:
        return yaml.safe_load(file)




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
            labels = labels.to(device)

            clin_inputs = {
                "age": inputs["age"].to(device),
                "td": inputs["td"].to(device),
            }

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(clin_inputs).squeeze(1)

                labels = labels.to(torch.float32)
                loss = criterion(logits, labels)
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
            optimizer.zero_grad()

            labels = labels.to(device)
           

            clin_inputs = {
                "age": inputs["age"].to(device),
                "td": inputs["td"].to(device),
            }

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(clin_inputs).squeeze(1)

                labels = labels.to(torch.float32)
                loss = criterion(logits, labels)
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


def objective(trial, is_ddp, rank, world_size, local_rank, device):
    params = {}

    search_space_cfg = load_yaml(OPTUNA_PARAMS_FILE)[MODEL_NAME]
    model_cfg = load_yaml(MODEL_CONFIG_FILE)[MODEL_NAME]

    # TODO: Modify this so that is has all the parameters you want to optimize
    if rank == 0:
        
        for param_name, config in search_space_cfg.items():
            suggest_method = getattr(trial, config['type'])
            params[param_name] = suggest_method(name=param_name, **config['kwargs'])

    trial_number = trial.number if rank == 0 else None
    if is_ddp:
        object_list = [params]
        dist.broadcast_object_list(object_list, src=0)
        params = object_list[0]

    
    


    # TODO: Those values are just an example
    your_train_args = "/users/scratch1/s189710/Multimodalny/data/data_buler/train_split_clean_cords_spot.csv"
    your_val_args = "/users/scratch1/s189710/Multimodalny/data/data_buler/val_split_clean_cords_spot.csv"
    print(f"REMOVE_SPOTMAG = {REMOVE_SPOTMAG}")
    # Create dataset and dataloader
    train_dataset = YourDataset(
        your_train_args,
        num_stats=None,
        remove_spotmag_rows=REMOVE_SPOTMAG,
    )
    print(f"Train dataset size: {len(train_dataset)}")
    your_model_args = dict(model_cfg)

    your_model_args["hidden_dim"] = params["hidden_dim"]
    your_model_args["depth"] = params["depth"]
    your_model_args["dropout"] = params["dropout"]
    your_model_args["activation"] = params["activation"]



    train_dataloader, train_sampler = create_dataloader(
        train_dataset,
        batch_size=params["batch_size"],
        shuffle=True,
        sample_type="oversample",
        num_workers=NUM_WORKERS,
        is_ddp=is_ddp,
        rank=rank,
        world_size=world_size,
        seed=SEED
    )

    val_dataset = YourDataset(
        your_val_args,
        num_stats=train_dataset.num_stats,
        remove_spotmag_rows=REMOVE_SPOTMAG,
    )
    print(f"Val dataset size: {len(val_dataset)}")
    val_dataloader, val_sampler = create_dataloader(
        val_dataset,
        batch_size=params["batch_size"],
        shuffle=False,
        sample_type=None,
        num_workers=NUM_WORKERS,
        is_ddp=is_ddp,
        rank=rank,
        world_size=world_size,
        seed=SEED
    )

    

    # Initialize model, loss function, and optimizer
    model = build_model(YourModelClass, your_model_args, is_ddp=is_ddp, rank=rank, local_rank=local_rank, device=device)

    criterion = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    trial_log_name = f"{MODEL_NAME}_{LOG_NAME}_trial_{trial_number}"

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
        log_name=trial_log_name)
    
    if rank == 0:
        return -best_val_auprc
    else:
        return 0.0


def main():
    # Setup distributed data processing
    is_ddp, local_rank, rank, world_size = init_distributed()

    sampler = TPESampler(seed=SEED) 

    if rank == 0:
        print(f"DDP initialized: is_ddp={is_ddp}, world_size={world_size}")
        print(f"Available GPUs: {torch.cuda.device_count()}")
        study = optuna.create_study(sampler=sampler)
    else:
        study = None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Creates a function with some arguments already filled. We use it because optuna want function with one argument (trial)
    objective_with_args = partial(
        objective, 
        is_ddp=is_ddp, 
        rank=rank, 
        world_size=world_size, 
        local_rank=local_rank, 
        device=device
    )

    if rank == 0:
        study.optimize(objective_with_args, n_trials=NUM_TRIALS)
    else:
        for _ in range(NUM_TRIALS):
            try:
                objective_with_args(None) 
            except Exception as e:
                print(f"Rank {rank} failed: {e}")
                break
    
    # Distributed data processing cleanup
    if is_ddp:
        cleanup_distributed()

    if rank == 0:
        print("\n=== Best parameters: ===")
        print(study.best_params)
        print("\n=== Study: ===")
        print("Best trials:", study.best_trials)
        print("Best value:", study.best_value)

if __name__ == "__main__":
    main()