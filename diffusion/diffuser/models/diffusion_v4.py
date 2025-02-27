import torch
from tqdm import tqdm
import torch.nn as nn

class DiffusionV4:
    def __init__(self, noise_steps, beta_start, beta_end, joint_dim, frames, device="cuda", predict_x0=False):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.joint_dim = joint_dim
        self.frames = frames
        self.device = device
        self.predict_x0 = predict_x0
        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
        
    def noise_sequence(self, x, t):
        """
        Add noise to the input sequence x at time t (forward diffusion process)

        x: (batch_size, joint_dim, frames)
        t: (batch_size)
        """
        x = x.to(self.device)
        t = t.to(self.device)

        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1. - self.alpha_hat[t])[:, None, None]
        epsilon = torch.randn_like(x, device=self.device)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon
    
    def sample_timesteps(self, batch_size):
        return torch.randint(0, self.noise_steps, (batch_size,), device=self.device)
    
    def sample(self, model, n):
        """
        Sample n samples from the model, using the reverse diffusion process
        
        Args:
            model: The model to sample from
            n: Number of samples to generate
            predict_x0: If True, assumes model predicts clean motion x0. If False, assumes model predicts noise.
        """
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, self.frames, self.joint_dim), device=self.device)
            for i in reversed(range(1, self.noise_steps)):
                t = (torch.ones(n, device=self.device) * i).long()
                
                if self.predict_x0:
                    # Model predicts x0 directly
                    predicted_x0 = model(x, t)
                    
                    # Convert predicted x0 to predicted noise
                    alpha_hat = self.alpha_hat[t][:, None, None]
                    sqrt_one_minus_alpha_hat = torch.sqrt(1. - alpha_hat)
                    
                    # Derive predicted noise from predicted x0
                    # ε = (x_t - √(α_t) * x_0) / √(1-α_t)
                    predicted_noise = (x - torch.sqrt(alpha_hat) * predicted_x0) / sqrt_one_minus_alpha_hat
                else:
                    # Model predicts noise directly (original behavior)
                    predicted_noise = model(x, t)
                
                # The rest of the sampling process remains the same
                alpha = self.alpha[t][:, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None]
                beta = self.beta[t][:, None, None]
                
                if i > 1:
                    noise = torch.randn_like(x, device=self.device)
                else:
                    noise = torch.zeros_like(x, device=self.device)
                
                x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * predicted_noise) + torch.sqrt(beta) * noise
        
        model.train()
        return x

    def training_loss(self, model, x_start, t):
        """
        Compute the training loss for the model at one timestep
        
        Args:
            model: The model to train
            x_start: The clean motion data
            t: The timesteps
        """
        x_start = x_start.to(self.device)
        t = t.to(self.device)
        
        x_noisy, noise = self.noise_sequence(x_start, t)
        
        if self.predict_x0:
            # Model predicts the clean motion x0
            predicted_x0 = model(x_noisy, t)
            
            # Calculate the target x0 from the noisy sample and the actual noise
            sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None]
            sqrt_one_minus_alpha_hat = torch.sqrt(1. - self.alpha_hat[t])[:, None, None]
            
            # x0 = (x_noisy - sqrt_one_minus_alpha_hat * noise) / sqrt_alpha_hat
            target_x0 = (x_noisy - sqrt_one_minus_alpha_hat * noise) / sqrt_alpha_hat
            
            loss = nn.functional.mse_loss(predicted_x0, target_x0)
        else:
            # Model predicts the noise (original behavior)
            predicted_noise = model(x_noisy, t)
            loss = nn.functional.mse_loss(predicted_noise, noise)
            
        return loss

