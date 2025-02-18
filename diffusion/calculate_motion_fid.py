import torch
from data_loaders.motion_dataset_v2 import MotionDataset
from diffuser.models.transformer_temporal import TransformerMotionModel
from metrics.fid_score import MotionFID

def main():
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load dataset
    dataset = MotionDataset("data/motions/humanoid3d_cartwheel.txt", shuffle=True)
    horizon = dataset[0].trajectories.shape[0]
    transition_dim = dataset[0].trajectories.shape[1]
    
    # Create and load model
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
    
    # Load checkpoint
    checkpoint_path = 'logs/transformer_cartwheel/state_90000.pt'
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    # Initialize FID calculator
    fid_calculator = MotionFID(
        real_dataset=dataset,
        model=model,
        device=device,
        batch_size=2,
        num_samples=10  # Number of samples to use for FID calculation
    )
    
    # Calculate FID score
    fid_score = fid_calculator.compute_fid()
    print(f"FID Score: {fid_score}")

if __name__ == "__main__":
    main() 