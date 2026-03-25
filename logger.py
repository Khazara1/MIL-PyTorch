import torch
import wandb

def get_logger(train_config):
    # Start a new wandb run to track this script.
    wandb_logger = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="kubawilk63-politechnika-gda-ska",
        # Set the wandb project where this run will be logged.
        project="MIL-Breast-Cancer",
        # Track hyperparameters and run metadata.
        config=train_config,
    )
    return wandb_logger


def log_metric(logger, metric: torch.Tensor, metric_name: str):
    assert isinstance(metric, torch.Tensor), f"Expected metric to be torch.Tensor, found {metric_name} of type {type(metric)}"

    if metric.ndim != 0:
        # Log separate metric for each class
        for class_id in range(metric.size()[0]):
            logger.log({f"{metric_name}_{class_id}": metric[class_id]})
    else:
        logger.log({metric_name: metric})