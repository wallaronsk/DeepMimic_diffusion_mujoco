import torch
import torch.nn as nn
import einops
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        return x + self.pe[:x.size(0)]

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class TransformerMotionModel(nn.Module):
    def __init__(
        self,
        horizon,
        transition_dim,
        cond_dim,
        dim=512,
        nhead=8,
        num_layers=6,
        dropout=0.1,
        n_timesteps=1000,
    ):
        super().__init__()
        
        self.horizon = horizon
        self.transition_dim = transition_dim
        self.n_timesteps = n_timesteps
        self.dim = dim  # Store dim for time embeddings
        
        # Add noise schedule parameters
        betas = torch.linspace(0.0001, 0.02, n_timesteps)
        alphas = 1.0 - betas
        self.register_buffer('alphas_cumprod', torch.cumprod(alphas, dim=0))
        self.register_buffer('noise_schedule', alphas)
        
        # Input embedding
        self.input_proj = nn.Linear(transition_dim, dim)
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=nhead,
            dim_feedforward=dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(dim, transition_dim)

    def forward(self, x, cond, time, verbose=False):
        """
        Args:
            x: [batch, horizon, transition_dim]
            cond: conditioning (unused in this implementation)
            time: [batch] time embeddings
        Returns:
            [batch, horizon, transition_dim]
        """
        b, h, t = x.shape
        
        # Project input to model dimension
        x = self.input_proj(x)  # [batch, horizon, dim]
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Time embedding - convert to float and proper dimension
        time_emb = self.time_mlp(time.float().to(x.device))  # [batch, dim]
        time_emb = time_emb.unsqueeze(1).expand(-1, h, -1)  # [batch, horizon, dim]
        
        # Add time embedding
        x = x + time_emb
        
        # Pass through transformer
        x = self.transformer_encoder(x)
        
        # Project back to transition dimension
        x = self.output_proj(x)  # [batch, horizon, transition_dim]
        
        return x

    def loss(self, batch):
        """
        Calculate diffusion loss
        """
        # Extract trajectories from batch and move to GPU
        trajectories = batch[0].cuda()  # Ensure trajectories are on GPU
        B, T, D = trajectories.shape
        
        # Sample random timesteps (on same device as trajectories)
        timesteps = torch.randint(0, self.n_timesteps, (B,), device=trajectories.device)
        
        # Get noise schedule (ensure on same device)
        alpha = self.noise_schedule[timesteps].to(trajectories.device)[:, None, None]
        alpha_hat = self.alphas_cumprod[timesteps].to(trajectories.device)[:, None, None]
        
        # Add noise to trajectories
        noise = torch.randn_like(trajectories)  # Will automatically be on same device as trajectories
        noised_trajectories = torch.sqrt(alpha_hat) * trajectories + torch.sqrt(1 - alpha_hat) * noise
        
        # Predict noise
        pred = self(noised_trajectories, None, timesteps)
        
        # Calculate loss
        loss = torch.nn.functional.mse_loss(pred, noise)
        
        return loss, {'mse': loss.item()}

    @torch.no_grad()
    def sample(self, batch_size=1, cond=None, horizon=None, device=None):
        """
        Generate a sample using the diffusion process
        Args:
            batch_size: number of samples to generate
            cond: conditioning information (e.g. first frame)
            horizon: length of sequence to generate
            device: device to generate on
        Returns:
            torch.Tensor: Generated motion sequence of shape (batch_size, horizon, transition_dim)
        """
        self.eval()
        device = device or next(self.parameters()).device
        horizon = horizon or self.horizon

        # Start with random noise
        x = torch.randn(batch_size, horizon, self.transition_dim, device=device)

        # Get conditioning from first frame if provided
        if cond is not None and 0 in cond:
            x[:, 0] = cond[0].to(device)

        # Reverse diffusion process
        for t in reversed(range(self.n_timesteps)):
            # Create timestep tokens
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Get noise prediction from model
            predicted_noise = self.forward(x, cond=cond, time=t_tensor)
            
            # Get alpha values for current timestep
            alpha = self.noise_schedule[t]
            alpha_hat = self.alphas_cumprod[t]
            
            # Calculate denoised sample
            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = 0

            # Reverse diffusion step
            x = (1 / torch.sqrt(alpha)) * (
                x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * predicted_noise
            ) + torch.sqrt(1 - alpha) * noise

            # Maintain conditioning if provided
            if cond is not None and 0 in cond:
                x[:, 0] = cond[0].to(device)

        return x 