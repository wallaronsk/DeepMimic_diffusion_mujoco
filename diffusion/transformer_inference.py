import os
import sys
import torch
import numpy as np
from collections import namedtuple

from diffuser.models.transformer_temporal_new import TransformerMotionModel
from data_loaders.motion_dataset_v2 import MotionDataset, Batch
from diffuser.utils.transformer_training import TransformerTrainer

# Setup paths and device
n_timesteps = 1000

training_steps = 100000
exp_name = f"transformer_walk_{n_timesteps}_steps"
savepath = f'logs/{exp_name}'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def load_checkpoint(model, checkpoint_path):
    """Load model from checkpoint"""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    print("Model loaded")
    print("Model number of parameters: {:,}".format(sum(p.numel() for p in model.parameters())))
    return model

def save_motion(motion_data, output_dir, filename="motion.npy"):
    """Save motion data to numpy file"""
    filepath = os.path.join(output_dir, filename)
    # Extract position data (first 35 dimensions)
    pos_data = motion_data[:, :35]
    # Flip signs for visualization compatibility
    pos_data[:, :2] = -pos_data[:, :2]
    pos_data[:, 4:8] = -pos_data[:, 4:8]
    
    # Reverse the order along the time (sequence) dimension
    pos_data = torch.flip(pos_data, dims=[0])
    
    pos_data = pos_data.cpu().numpy()
    np.save(filepath, pos_data)
    print(f"Motion saved as {filename}")

def main():
    # Setup cudnn benchmark for performance if using GPU
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # Load dataset to get dimensions
    dataset = MotionDataset("data/motions/humanoid3d_walk.txt", shuffle=True)
    horizon = dataset[0].trajectories.shape[0]
    transition_dim = dataset[0].trajectories.shape[1]

    # Create model with same config as training
    model = TransformerMotionModel(
        horizon=horizon,
        transition_dim=transition_dim,
        dim=512,
        nhead=4,
        num_layers=8,
        dropout=0.1,
        n_timesteps=n_timesteps
    ).to(device)

    # Load latest checkpoint
    checkpoint_path = os.path.join(savepath, f'state_{training_steps - 1}.pt')
    model = load_checkpoint(model, checkpoint_path)
    model.eval()

    # Generate samples
    with torch.no_grad():
        print("Generating sample...")
        # Use AMP autocast (only when using GPU) for maximum throughput.
        with torch.cuda.amp.autocast(enabled=(device != 'cpu')):
            # Generate sample without conditioning
            sample = model.sample(
                batch_size=1,
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