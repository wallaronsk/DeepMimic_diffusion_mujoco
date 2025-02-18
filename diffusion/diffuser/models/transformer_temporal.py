import torch
import torch.nn as nn
import math
from diffuser.losses.kl_loss import KLDivergenceLoss

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [B, seq_len, d_model]
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # x is a tensor of shape [B] containing timestep values
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = x.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class TransformerMotionModel(nn.Module):
    def __init__(
        self,
        horizon,
        transition_dim,
        dim=512,
        nhead=8,
        num_layers=8,
        dropout=0.1,
        n_timesteps=1000,  # Number of steps in the diffusion process (forward and reverse)
        beta_schedule='linear'
    ):
        super().__init__()
        self.horizon = horizon
        self.transition_dim = transition_dim
        self.n_timesteps = n_timesteps  # Store for use in forward/reverse processes
        self.dim = dim

        # Set up noise schedule for n_timesteps steps of diffusion
        beta_start = 0.0001
        beta_end = 0.02
        print(f"Initializing diffusion model with {n_timesteps} timesteps")
        print(f"Beta schedule: {beta_schedule} (start={beta_start}, end={beta_end})")
        
        betas = torch.linspace(beta_start, beta_end, n_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.], dtype=alphas.dtype, device=alphas.device), alphas_cumprod[:-1]])

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))
        
        # Input embedding
        self.input_proj = nn.Linear(transition_dim, dim)

        # Positional encoding with dropout
        self.pos_encoder = PositionalEncoding(dim, dropout=dropout)

        # Timestep embedding (embed_timestep): using a two-layer network as in the reference
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

        # Learned query embeddings for the decoder (one per time step)
        self.query_embed = nn.Parameter(torch.randn(horizon, dim))

        # Transformer Decoder with causal masking
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=nhead,
            dim_feedforward=dim * 2,  # as per reference (512->1024->512 for dim=512)
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output projection
        self.output_proj = nn.Linear(dim, transition_dim)

    def forward(self, x, time, verbose=False):
        # x: [B, horizon, transition_dim]
        B, L, _ = x.shape

        # Encoder: project input and add positional encoding
        encoder_out = self.input_proj(x)  # [B, L, dim]
        encoder_out = self.pos_encoder(encoder_out)

        # Compute time embedding and expand
        time_emb = self.time_mlp(time)  # [B, dim]
        
        # Prepare queries: learned queries plus time embedding
        queries = self.query_embed.unsqueeze(0).expand(B, -1, -1)  # [B, L, dim]
        queries = queries + time_emb.unsqueeze(1)

        # Create causal mask for the decoder (prevent future look-ahead)
        tgt_mask = torch.triu(torch.ones(L, L, device=x.device) * float('-inf'), diagonal=1)

        # Transformer Decoder: queries attend to encoder outputs
        decoded = self.transformer_decoder(tgt=queries, memory=encoder_out, tgt_mask=tgt_mask)

        # Project decoder output back to transition dimension
        out = self.output_proj(decoded)  # [B, L, transition_dim]
        return out

    def loss(self, batch):
        """
        Calculate L-simple loss (MSE) as described in Ho et al.
        Returns loss value and metrics dictionary
        """
        # Extract trajectories from batch
        if isinstance(batch, torch.Tensor):
            trajectories = batch
        elif hasattr(batch, 'trajectories'):
            trajectories = batch.trajectories
        else:
            trajectories = batch[0]

        B, T, D = trajectories.shape
        device = trajectories.device

        # Sample random timesteps for each item in batch
        t = torch.randint(0, self.n_timesteps, (B,), device=device)

        # Generate random noise
        noise = torch.randn_like(trajectories)

        # Get noisy samples at timestep t
        noisy_trajectories = (
            self.sqrt_alphas_cumprod[t, None, None] * trajectories +
            self.sqrt_one_minus_alphas_cumprod[t, None, None] * noise
        )

        # Predict noise using the model
        predicted_noise = self(noisy_trajectories, t)

        # Calculate simple MSE loss between actual and predicted noise
        loss = nn.functional.mse_loss(predicted_noise, noise)

        # Calculate metrics for logging
        with torch.no_grad():
            signal_power = torch.mean(trajectories ** 2)
            noise_power = torch.mean(noise ** 2)
            snr = 10 * torch.log10(signal_power / noise_power)
            mean_timestep = torch.mean(t.float()) / self.n_timesteps
            mse_by_component = torch.mean((predicted_noise - noise) ** 2, dim=(0, 1))
            metrics = {
                'snr_db': snr.item(),
                'avg_timestep': mean_timestep.item(),
                'mse_position': mse_by_component[:3].mean().item(),
                'mse_total': loss.item()
            }

        return loss, metrics

    @torch.no_grad()
    def sample(self, batch_size=1, horizon=None, device=None):
        """
        Generate samples using the reverse diffusion process (DDPM sampling)
        Performs n_timesteps steps of denoising, from t=n_timesteps-1 to t=0
        """
        self.eval()
        device = device or next(self.parameters()).device
        horizon = horizon or self.horizon

        print(f"\nStarting sampling process with {self.n_timesteps} denoising steps")
        
        # Start from pure noise
        x = torch.randn(batch_size, horizon, self.transition_dim, device=device)

        for t in reversed(range(self.n_timesteps)):
            if t % 100 == 0:  # Log progress every 100 steps
                print(f"Denoising step {t}/{self.n_timesteps-1}")
                
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Use AMP autocast if available to maximize throughput
            with torch.cuda.amp.autocast(enabled=(device != "cpu")):
                predicted_noise = self(x, t_batch)

            # Get noise schedule parameters for timestep t
            alpha = self.alphas[t]
            alpha_cumprod = self.alphas_cumprod[t]

            # Add noise if not the final step
            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = 0

            # Update x using the reverse diffusion formula
            x = (1 / torch.sqrt(alpha)) * (
                x - ((1 - alpha) / torch.sqrt(1 - alpha_cumprod)) * predicted_noise
            ) + torch.sqrt(1 - alpha) * noise

        return x

    @torch.no_grad()
    def update_ema(self, ema_model, decay):
        """Update exponential moving average of model parameters."""
        for param, ema_param in zip(self.parameters(), ema_model.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay) 