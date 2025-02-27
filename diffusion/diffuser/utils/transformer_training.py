from models.transformer_temporal import TransformerTemporal
from models.diffusion_v4 import DiffusionV4
import torch.optim as optim
from data_loaders.motion_dataset_v2 import MotionDataset

def train(args):    
    device = args.device
    model = args.model
    diffusion = args.diffusion
    optimizer = args.optimizer
    dataloader = args.dataloader

    for i in range(args.num_train_steps):
        batch = next(dataloader)
        x_start = batch["trajectories"]
        t = diffusion.sample_timesteps(x_start.shape[0])
        x_noisy, noise = diffusion.noise_sequence(x_start, t)
        predicted_noise = model(x_noisy, t)
        loss = diffusion.training_loss(model, x_start, t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print(f"Step {i} loss: {loss.item()}")

args = {
    "device": "cuda",
    "model": TransformerTemporal(),
    "diffusion": DiffusionV4(),
    "optimizer": optim.Adam(model.parameters(), lr=0.001),
    "dataloader": MotionDataset(src="data/motions/humanoid3d_walk.txt"),
    "num_train_steps": 100
}

train(args)