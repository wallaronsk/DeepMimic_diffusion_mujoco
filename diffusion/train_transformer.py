import torch
import os
import sys
import mlflow
from datetime import datetime

from diffuser.models.transformer_temporal_new import TransformerMotionModel
from diffuser.utils.transformer_training import TransformerTrainer
from data_loaders.motion_dataset_v2 import MotionDataset

def train_with_timesteps(n_timesteps):
    # MLflow setup
    experiment_name = f"transformer_diffusion_{n_timesteps}_steps"
    experiment_name += "_512_2_8_optimized"  # Updated name to reflect optimized config
    run_name = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Check if experiment exists, if not create it
    try:
        mlflow.create_experiment(experiment_name)
    except Exception:
        pass
    
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=run_name):
        print(f"\n{'='*50}")
        print(f"Starting training with {n_timesteps} diffusion steps")
        print(f"{'='*50}\n")
        
        # Updated parameters based on tuning results
        params = {
            "model_dim": 512,          # Keeping this at 512 as it wasn't in the analysis
            "n_heads": 2,              # Best performing value
            "n_layers": 8,             # Best performing value
            "dropout": 0.15,           # Keeping original value as it wasn't varied
            "n_timesteps": n_timesteps,
            "batch_size": 32,          # Best performing value
            "learning_rate": 1e-4,     # Best performing value
            "ema_decay": 0.995,        # Keeping original value
            "mask_ratio": 0.05,        # Best performing value
            "warmup_steps": 10000,       # From tuning
            "total_steps": 100000,     # Keeping original value for full training
            "smooth_loss_weight": 0.1   # Best performing value
        }
        mlflow.log_params(params)
        
        # Dataset setup
        dataset = MotionDataset("data/motions/humanoid3d_walk.txt", shuffle=True)
        
        # Model parameters
        horizon = dataset[0].trajectories.shape[0]
        transition_dim = dataset[0].trajectories.shape[1]

        print(f"\nStarting training with optimized configuration:")
        print("horizon: ", horizon)
        print("transition_dim: ", transition_dim)
        for k, v in params.items():
            print(f"{k}: {v}")
        
        # Log dataset info
        mlflow.log_params({
            "sequence_length": horizon,
            "state_dimension": transition_dim
        })
        
        # Create model with optimized parameters
        model = TransformerMotionModel(
            horizon=horizon,
            transition_dim=transition_dim,
            dim=params["model_dim"],
            nhead=params["n_heads"],
            num_layers=params["n_layers"],
            dropout=params["dropout"],
            n_timesteps=params["n_timesteps"],
            beta_schedule='linear',
            smooth_loss_weight=params["smooth_loss_weight"]
        ).cuda()
        
        # Print model architecture and number of parameters
        print(model)
        print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

        # Create trainer with optimized parameters
        trainer = TransformerTrainer(
            diffusion_model=model,
            dataset=dataset,
            ema_decay=params["ema_decay"],
            mask_ratio=params["mask_ratio"],
            train_batch_size=params["batch_size"],
            train_lr=params["learning_rate"],
            warmup_steps=params["warmup_steps"],
            results_folder=f'./logs/transformer_walk_{n_timesteps}_steps_optimized',
            mlflow_run=mlflow.active_run()
        )
        
        # Train the model
        trainer.train(n_train_steps=params["total_steps"])

def main():
    # Enable CuDNN benchmark for potential speed improvements on fixed-size inputs
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # MLflow setup
    mlflow.set_tracking_uri("file:./mlruns")
    
    # Different numbers of timesteps to try
    timesteps_to_try = [1000]
    
    for n_timesteps in timesteps_to_try:
        train_with_timesteps(n_timesteps)
        torch.cuda.empty_cache()  # Clear GPU memory between runs

if __name__ == '__main__':
    main() 