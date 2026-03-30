# Setup
1. Install `uv`
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
2. Sync packages and python version
```bash
uv sync
```

# Hyperparameter search with optuna
1. Hyperparameters that will be optimized using Optuna should be set in `config/optuna_config.yaml` file. All the parameters that your model takes should also be set in `config/model_config.yaml` file.
2. After the configs setup run:
```bash
torchrun --nproc_per_node=4 train_optuna.py
```
to train the model. You should set the correct `BACKBONE` inside this script so that the correct model is trained.

# Final training
1. After optimizing the hyperparameters you should set them inside `config/train_config.yaml` file.
2. To run the final training, set the correct model `BACKBONE` inside `train.py` and run:
```bash
torchrun --nproc_per_node=4 train.py
```