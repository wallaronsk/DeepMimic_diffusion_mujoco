import mlflow
import pandas as pd
import numpy as np
from collections import defaultdict

def get_experiment_metrics(experiment_name):
    """Get metrics from all runs in an experiment."""
    mlflow.set_tracking_uri("file:./mlruns")
    
    # Get experiment by name
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"No experiment found with name: {experiment_name}")
        return None
    
    # Get all runs for the experiment
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    return runs

def analyze_experiments():
    """Analyze all tuning experiments and find the best configuration."""
    # Get all experiments
    client = mlflow.tracking.MlflowClient("file:./mlruns")
    experiments = client.search_experiments()
    
    # Filter for tuning experiments
    tuning_experiments = [exp for exp in experiments if exp.name.startswith("tuning_transformer_diffusion")]
    
    all_runs = []
    for exp in tuning_experiments:
        runs = get_experiment_metrics(exp.name)
        if runs is not None and not runs.empty:
            all_runs.append(runs)
    
    if not all_runs:
        print("No tuning experiments found!")
        return
    
    # Combine all runs
    all_runs_df = pd.concat(all_runs, ignore_index=True)
    
    # Extract relevant metrics
    metrics_of_interest = ['metrics.loss_total', 'metrics.loss_angle', 'metrics.loss_velocity']
    
    # Get the parameters we tuned
    param_columns = [col for col in all_runs_df.columns if col.startswith('params.')]
    
    # Create a summary dataframe with both parameters and final metrics
    summary = all_runs_df[param_columns + metrics_of_interest].copy()
    
    # Sort by total loss (assuming lower is better)
    summary = summary.sort_values('metrics.loss_total')
    
    # Get the best run (lowest total loss)
    best_run = summary.iloc[0]
    
    # Print results
    print("\n=== Best Hyperparameter Configuration ===")
    print("\nParameters:")
    for param in param_columns:
        param_name = param.replace('params.', '')
        print(f"{param_name}: {best_run[param]}")
    
    print("\nMetrics:")
    print(f"Total Loss   : {best_run['metrics.loss_total']:.6f}")
    print(f"Angle Loss   : {best_run['metrics.loss_angle']:.6f}")
    print(f"Velocity Loss: {best_run['metrics.loss_velocity']:.6f}")
    
    # Save top 5 configurations to a file
    top_5 = summary.head(5)
    
    with open('best_configs.txt', 'w') as f:
        f.write("=== Top 5 Hyperparameter Configurations ===\n\n")
        
        for idx, run in top_5.iterrows():
            f.write(f"Rank {idx + 1}\n")
            f.write("Parameters:\n")
            for param in param_columns:
                param_name = param.replace('params.', '')
                f.write(f"{param_name}: {run[param]}\n")
            
            f.write("\nMetrics:\n")
            f.write(f"Total Loss   : {run['metrics.loss_total']:.6f}\n")
            f.write(f"Angle Loss   : {run['metrics.loss_angle']:.6f}\n")
            f.write(f"Velocity Loss: {run['metrics.loss_velocity']:.6f}\n")
            f.write("\n" + "="*50 + "\n\n")
    
    print(f"\nTop 5 configurations have been saved to 'best_configs.txt'")
    
    # Additional analysis
    print("\n=== Parameter Impact Analysis ===")
    for param in param_columns:
        param_name = param.replace('params.', '')
        param_values = summary[param].unique()
        if len(param_values) > 1:  # Only analyze parameters that were actually varied
            print(f"\n{param_name}:")
            for value in param_values:
                mean_loss = summary[summary[param] == value]['metrics.loss_total'].mean()
                print(f"  Value {value}: avg loss = {mean_loss:.6f}")

if __name__ == "__main__":
    analyze_experiments() 