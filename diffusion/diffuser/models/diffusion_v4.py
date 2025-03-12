import torch
from tqdm import tqdm
import torch.nn as nn
import math


class DiffusionV4:
    def __init__(self, noise_steps, beta_start, beta_end, joint_dim, frames, device="cuda", predict_x0=False, 
                 schedule_type="linear", cosine_s=0.008):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.joint_dim = joint_dim
        self.frames = frames
        self.device = device
        self.predict_x0 = predict_x0

        # Noise schedule parameters
        self.schedule_type = schedule_type
        self.cosine_s = cosine_s
        # Data type parameter
        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        if self.schedule_type == "linear":
            # Linear noise schedule (original implementation)
            return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
        elif self.schedule_type == "cosine":
            # Cosine noise schedule (often better for image/motion generation)
            # Implementation based on improved DDPM paper
            steps = self.noise_steps + 1
            x = torch.linspace(0, self.noise_steps, steps)
            alphas_cumprod = torch.cos(((x / self.noise_steps) + self.cosine_s) / (1 + self.cosine_s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            
            # Clamp the betas to be in [beta_start, beta_end]
            return torch.clip(betas, self.beta_start, self.beta_end)
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}. Use 'linear' or 'cosine'.")
    
    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion process: q(x_t | x_0)
        
        Args:
            x_start: The clean data at t=0
            t: The timesteps to diffuse to
            noise: Optional pre-generated noise (if None, will generate noise)
            
        Returns:
            The noisy data at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_start, device=x_start.device)
            
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1. - self.alpha_hat[t])[:, None, None]
        
        return sqrt_alpha_hat * x_start + sqrt_one_minus_alpha_hat * noise
        
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
    
    def sample(self, model, n, custom_frames=None, condition=None):
        """
        Sample n samples from the model, using the reverse diffusion process
        
        Args:
            model: The model to sample from
            n: Number of samples to generate
            custom_frames: Optional custom number of frames to generate (default: self.frames)
            condition: Optional condition to sample from
        """
        model.eval()
        with torch.no_grad():
            # Use custom_frames if provided, otherwise use the default frames
            frames_to_generate = custom_frames if custom_frames is not None else self.frames
            
            # Generate samples with the appropriate dimension based on data_type
            x = torch.randn((n, frames_to_generate, self.joint_dim), device=self.device)
            
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
        Calculate the training loss for the diffusion model.
        
        Args:
            model: The model to compute predictions
            x_start: Starting clean data [batch_size, sequence_length, feature_dim]
            t: Timesteps for diffusion process [batch_size]
            
        Returns:
            Loss value
        """
        # Sample noise
        noise = torch.randn_like(x_start)
        
        # Get noisy samples at timesteps t
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        
        # Get model predictions (either predict_noise or predict_x0)
        if self.predict_x0:
            # Model predicts x0 directly
            predicted_x0 = model(x_noisy, t)
            
            # Calculate noise from predicted x0 for loss
            alpha_hat = self.alpha_hat[t].view(-1, 1, 1)
            beta_hat = 1 - alpha_hat
            
            predicted_noise = (x_noisy - torch.sqrt(alpha_hat) * predicted_x0) / torch.sqrt(beta_hat)
        else:
            # Model predicts noise directly
            predicted_noise = model(x_noisy, t)
            
            # For motion losses, we need x0 too
            alpha_hat = self.alpha_hat[t].view(-1, 1, 1)
            beta_hat = 1 - alpha_hat
            predicted_x0 = (x_noisy - torch.sqrt(beta_hat) * predicted_noise) / torch.sqrt(alpha_hat)
        
        
        loss = nn.functional.mse_loss(predicted_noise, noise)
    
        return loss

