import os
import sys
import torch
import numpy as np
from collections import namedtuple

from diffuser.models.transformer_temporal import TransformerMotionModel
from data_loaders.motion_dataset_v2 import MotionDataset, Batch
from diffuser.utils.transformer_training import TransformerTrainer

# Setup paths and device
exp_name = "transformer_cartwheel"
savepath = f'logs/{exp_name}'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def load_checkpoint(model, checkpoint_path):
    """Load model from checkpoint"""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    return model

def save_motion(motion_data, output_dir, filename="motion.npy"):
    """Save motion data to numpy file"""
    filepath = os.path.join(output_dir, filename)
    # Extract position data (first 35 dimensions)
    pos_data = motion_data[:, :35]
    # Flip signs for visualization compatibility
    pos_data[:, :2] = -pos_data[:, :2]
    pos_data[:, 4:8] = -pos_data[:, 4:8]
    
    pos_data = pos_data.cpu().numpy()
    np.save(filepath, pos_data)
    print(f"Motion saved as {filename}")

def main():
    # Load dataset to get dimensions
    dataset = MotionDataset("data/motions/humanoid3d_cartwheel.txt", shuffle=True)
    horizon = dataset[0].trajectories.shape[0]
    transition_dim = dataset[0].trajectories.shape[1]

    # Create model with same config as training
    model = TransformerMotionModel(
        horizon=horizon,
        transition_dim=transition_dim,
        cond_dim=transition_dim,
        dim=512,
        nhead=8,
        num_layers=6,
        dropout=0.1,
        n_timesteps=1000
    ).to(device)

    # Load latest checkpoint
    checkpoint_path = os.path.join(savepath, 'state_11000.pt')  # Using checkpoint from epoch 3
    model = load_checkpoint(model, checkpoint_path)
    model.eval()

    # Generate samples
    with torch.no_grad():
        # Get conditioning from first sample in dataset
        batch = dataset[0]
        cond = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.conditions.items()}
        
        # Generate sample
        print("Generating sample...")
        sample = model.sample(
            batch_size=1,
            cond=cond,
            horizon=horizon,
            device=device
        )
        
        # Save the generated motion
        save_motion(
            sample.squeeze(0),
            savepath,
            filename="generated_motion.npy"
        )

if __name__ == "__main__":
    main() 