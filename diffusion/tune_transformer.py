import torch
import mlflow
from datetime import datetime
import itertools
import os

from diffuser.models.transformer_temporal_new import TransformerMotionModel
from diffuser.utils.transformer_training import TransformerTrainer
from data_loaders.motion_dataset_v2 import MotionDataset

def run_experiment(params):
    """
    Run one training experiment with the given hyperparameters.
    """
    # Set up an experiment name based on the parameters.
    experiment_name = f"tuning_transformer_diffusion_{params['n_timesteps']}_steps"
    experiment_name += f"_{params['model_dim']}_{params['n_heads']}_{params['n_layers']}_{params['mask_ratio']}_{params['learning_rate']}"
    
    run_name = f"tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Ensure the MLflow experiment exists.
    try:
        mlflow.create_experiment(experiment_name)
    except Exception:
        pass
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=run_name) as run:
        print(f"\nStarting tuning run {run.info.run_id} with parameters:")
        print(params)
        mlflow.log_params(params)
        
        # Load dataset.
        dataset = MotionDataset("data/motions/humanoid3d_walk.txt", shuffle=True)
        horizon = dataset[0].trajectories.shape[0]
        transition_dim = dataset[0].trajectories.shape[1]
        mlflow.log_params({
            "sequence_length": horizon,
            "state_dimension": transition_dim
        })
        
        # Create the transformer diffusion model.
        model = TransformerMotionModel(
            horizon=horizon,
            transition_dim=transition_dim,
            dim=params["model_dim"],
            nhead=params["n_heads"],
            num_layers=params["n_layers"],
            dropout=params["dropout"],
            n_timesteps=params["n_timesteps"],
            beta_schedule=params["beta_schedule"],
            smooth_loss_weight=params["smooth_loss_weight"]
        ).cuda()
        
        print(model)
        print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
        
        # Create the trainer with the current hyperparameters.
        trainer = TransformerTrainer(
            diffusion_model=model,
            dataset=dataset,
            ema_decay=params["ema_decay"],
            warmup_steps=params["warmup_steps"],
            mask_ratio=params["mask_ratio"],
            train_batch_size=params["batch_size"],
            train_lr=params["learning_rate"],
            results_folder=f'./logs/tuning_transformer_{run.info.run_id}',
            mlflow_run=mlflow.active_run(),
            is_tuning=True
        )
        
        # --- New: Override log_metrics to record loss metrics for tuning ---
        tuning_metrics_history = []
        original_log_metrics = trainer.log_metrics
        def new_log_metrics(loss, infos, grad_norm, lr):
            tuning_metrics_history.append(infos.copy())
            original_log_metrics(loss, infos, grad_norm, lr)
        trainer.log_metrics = new_log_metrics
        
        # Train for a limited number of steps (e.g. 10k steps for tuning).
        trainer.train(n_train_steps=params["total_steps"])
        
        # After training, compute average losses from the last 10 logged metrics.
        if tuning_metrics_history:
            last_entries = tuning_metrics_history[-10:]
            avg_loss_total = sum(item.get("loss_total", 0.0) for item in last_entries) / len(last_entries)
            avg_loss_angle = sum(item.get("loss_angle", 0.0) for item in last_entries) / len(last_entries)
            avg_loss_velocity = sum(item.get("loss_velocity", 0.0) for item in last_entries) / len(last_entries)
        else:
            avg_loss_total = float('nan')
            avg_loss_angle = float('nan')
            avg_loss_velocity = float('nan')

        results = {
            "params": params,
            "avg_loss_total": avg_loss_total,
            "avg_loss_angle": avg_loss_angle,
            "avg_loss_velocity": avg_loss_velocity,
        }
        return results

def main():
    # Define a grid of hyperparameters.
    grid = {
        "model_dim": [256, 512],
        "n_heads": [2, 4],
        "n_layers": [4, 8],
        "dropout": [0.15],
        "n_timesteps": [1000],
        "beta_schedule": ["linear"],  # Could also add "cosine"
        "smooth_loss_weight": [0.1, 0.2],
        "batch_size": [2, 32],
        "learning_rate": [1e-4, 5e-5],
        "ema_decay": [0.995],
        "warmup_steps": [500],
        "mask_ratio": [0.0, 0.05],
        "total_steps": [10000]  # Use fewer steps for tuning experiments.
    }
    
    # Generate all parameter combinations using a grid search.
    keys = list(grid.keys())
    experiments = []
    for values in itertools.product(*(grid[k] for k in keys)):
        exp_params = dict(zip(keys, values))
        experiments.append(exp_params)
    
    print(f"Running {len(experiments)} experiments...")
    
    results_list = []
    for exp_params in experiments:
        result = run_experiment(exp_params)
        results_list.append(result)
        torch.cuda.empty_cache()  # Clear GPU memory between runs if needed.

    # Write all experiment results to a text file.
    results_file = "tuning_results.txt"
    with open(results_file, "w") as f:
        for res in results_list:
            f.write(f"Params: {res['params']}\n")
            f.write(f"Avg Total Loss   : {res['avg_loss_total']}\n")
            f.write(f"Avg Angle Loss   : {res['avg_loss_angle']}\n")
            f.write(f"Avg Velocity Loss: {res['avg_loss_velocity']}\n")
            f.write("------\n")
    print(f"Tuning results saved to {results_file}")

if __name__ == '__main__':
    main() 