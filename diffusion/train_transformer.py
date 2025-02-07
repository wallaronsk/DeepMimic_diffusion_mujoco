import torch
import os
import sys

from diffuser.models.transformer_temporal import TransformerMotionModel
from diffuser.utils.transformer_training import TransformerTrainer
from data_loaders.motion_dataset_v2 import MotionDataset

def main():
    # Dataset setup
    dataset = MotionDataset("data/motions/humanoid3d_cartwheel.txt", shuffle=True)
    
    # Model parameters
    horizon = dataset[0].trajectories.shape[0]  # sequence length
    transition_dim = dataset[0].trajectories.shape[1]  # full state dimension

    print("horizon: ", horizon)
    print("transition_dim: ", transition_dim)
    
    # Create model
    model = TransformerMotionModel(
        horizon=horizon,
        transition_dim=transition_dim,
        cond_dim=transition_dim,  # Using same dim for conditioning
        dim=512,        # Hidden dimension
        nhead=8,        # Number of attention heads
        num_layers=6,   # Number of transformer layers
        dropout=0.1,
        n_timesteps=1000  # Added parameter
    ).cuda()
    
    # Create trainer
    trainer = TransformerTrainer(
        diffusion_model=model,
        dataset=dataset,
        train_batch_size=32,
        train_lr=2e-4,
        warmup_steps=1000,
        results_folder='./logs/transformer_cartwheel'
    )
    
    # Train
    trainer.train(n_train_steps=100000)

if __name__ == '__main__':
    main() 