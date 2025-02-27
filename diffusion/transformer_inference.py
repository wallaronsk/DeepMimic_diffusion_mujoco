import torch
from diffuser.models.transformer_temporal import TransformerMotionModel
from diffuser.models.diffusion_v4 import DiffusionV4
from data_loaders.motion_dataset_v2 import MotionDataset
import os
import numpy as np

def inference(model, diffusion, device, num_samples):
    model.eval()
    with torch.no_grad():
        x = diffusion.sample(model, num_samples)

    return x

dataset = MotionDataset("data/motions/humanoid3d_walk.txt")
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # find the latest model
    model_path = "models/model_1000.pth"
    if not os.path.exists(model_path):
        print("Model not found, training...")
        train_transformer()
    else:
        print("Model found, loading...")
        model = torch.load(model_path)
        model.to(device)
    joint_dim = dataset[0].trajectories.shape[1]
    frames = dataset[0].trajectories.shape[0]
    diffusion = DiffusionV4(noise_steps=50, beta_start=0.0001, beta_end=0.02, joint_dim=joint_dim, frames=frames, device=device, predict_x0=True)
    x = inference(model, diffusion, device, 1)
    print(x.shape)

first_x = x[0]

# save the motion
def save_motions(sample, output_dir, filename="motion.npy"):
    filepath = os.path.join(output_dir, filename)
    # create filepath if not exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    pos_data = sample[:, :35]
    pos_data = pos_data.squeeze(0).cpu().numpy()
    np.save(filepath, pos_data)
    print(f"Motion saved as {filename}")

savepath = "testoutput"
save_motions(first_x, f"{savepath}/sampled_motions", filename="base-motion2.npy")