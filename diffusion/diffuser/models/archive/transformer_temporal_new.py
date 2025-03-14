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
        n_timesteps=1000,  # Number of diffusion steps
        beta_schedule='linear',
        smooth_loss_weight=0.1  # Weight for the velocity smoothness loss
    ):
        super().__init__()
        self.horizon = horizon
        self.transition_dim = transition_dim
        self.n_timesteps = n_timesteps
        self.dim = dim
        self.smooth_loss_weight = smooth_loss_weight  # new parameter

        # Set up noise schedule
        beta_start = 0.0001
        beta_end = 0.02
        print(f"Initializing diffusion model with {n_timesteps} timesteps")
        print(f"Beta schedule: {beta_schedule} (start={beta_start}, end={beta_end})")
        
        if beta_schedule == 'cosine':
            betas = self._cosine_beta_schedule(n_timesteps)
        else:
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
        
        # === Architecture matching the original motion diffusion paper ===

        # input_process: Project input poses (e.g. 263-dim) to model dimension (512)
        self.input_process = nn.Linear(transition_dim, dim)  # poseEmbedding
        
        # sequence_pos_encoder: Positional encoding for the (projected) sequence
        self.sequence_pos_encoder = PositionalEncoding(dim, dropout=dropout)
        
        # embed_timestep: A Timestep Embedder (using sinusoidal embedding + two-layer MLP)
        self.embed_timestep = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        
        # --- New: Learned time embedding for diffusion conditioning ---
        self.learned_time_embed = nn.Embedding(n_timesteps, dim)
        
        # Instead of a separate text-conditioning mechanism, we use learned sequence queries.
        # These will serve as the target (decoder) queries for the transformer decoder.
        self.seq_queries = nn.Parameter(torch.randn(horizon, dim))
        
        # seqTransDecoder: Transformer decoder (with 8 layers by default)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=nhead,
            dim_feedforward=dim * 2,  # e.g. 1024 for dim=512
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.seqTransDecoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # output_process: Final projection back to the original pose dimension (e.g. 263)
        self.output_process = nn.Linear(dim, transition_dim)  # poseFinal

        # --- New: Convolutional block to capture local joint interactions ---
        self.conv_local = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # --- New: Spatial attention refinement block ---
        self.spatial_attn = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        )

    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """Compute betas using a cosine schedule."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, device=device)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * (math.pi / 2)) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clamp(betas, 0.0001, 0.9999)
        return betas

    def forward(self, x, time):
        """
        x: [B, horizon, transition_dim] – the (noisy) pose sequence.
        time: [B] – diffusion timesteps.
        """
        B, L, _ = x.shape

        # Process input through input_process and add positional encoding from sequence_pos_encoder
        h = self.input_process(x)  # [B, L, dim]
        h = self.sequence_pos_encoder(h)
        # --- New: Apply convolutional branch for local interactions ---
        h_conv = self.conv_local(h.transpose(1, 2)).transpose(1, 2)
        h = h + h_conv
        
        # Get time embedding from embed_timestep and add learned embedding: [B, dim]
        time_emb = self.embed_timestep(time) + self.learned_time_embed(time)
        
        # Create queries for the decoder by adding time embedding to learned sequence queries.
        # queries: [B, horizon, dim]
        queries = self.seq_queries.unsqueeze(0).expand(B, -1, -1) + time_emb.unsqueeze(1)
        
        # Create causal mask for the transformer decoder (to prevent future look-ahead)
        tgt_mask = torch.triu(torch.ones(L, L, device=x.device) * float('-inf'), diagonal=1)
        
        # Run the transformer decoder (seqTransDecoder) for denoising.
        decoded = self.seqTransDecoder(tgt=queries, memory=h, tgt_mask=tgt_mask)
        # --- New: Apply spatial attention refinement ---
        decoded_conv = self.spatial_attn(decoded.transpose(1, 2)).transpose(1, 2)
        decoded = decoded + decoded_conv
        
        # Output processing to predict the noise component (will be used to recover angles)
        out = self.output_process(decoded)  # [B, L, transition_dim]
        return out

    def loss(self, batch):
        """
        Compute the loss as the MSE on pose (angle) recovery plus an added smoothness loss
        on the predicted velocities.
        """
        if isinstance(batch, torch.Tensor):
            trajectories = batch
        elif hasattr(batch, 'trajectories'):
            trajectories = batch.trajectories
        else:
            trajectories = batch[0]

        B, T, D = trajectories.shape
        device = trajectories.device

        # Sample random timesteps for each item in the batch.
        t = torch.randint(0, self.n_timesteps, (B,), device=device)

        # Generate random noise of the same shape as trajectories.
        noise = torch.randn_like(trajectories)

        # Create the noisy trajectories according to the diffusion schedule.
        noisy_trajectories = (
            self.sqrt_alphas_cumprod[t, None, None] * trajectories +
            self.sqrt_one_minus_alphas_cumprod[t, None, None] * noise
        )

        # Predict noise using the model.
        predicted_noise = self(noisy_trajectories, t)
        # --- New: Recover predicted angles from predicted noise ---
        predicted_angles = (noisy_trajectories - self.sqrt_one_minus_alphas_cumprod[t, None, None] * predicted_noise) / self.sqrt_alphas_cumprod[t, None, None]
        angle_loss = nn.functional.mse_loss(predicted_angles, trajectories)

        # --- New: Velocity smoothness loss ---
        pred_vel = predicted_angles[:, 1:, :] - predicted_angles[:, :-1, :]
        true_vel = trajectories[:, 1:, :] - trajectories[:, :-1, :]
        velocity_loss = nn.functional.mse_loss(pred_vel, true_vel)

        loss = angle_loss + self.smooth_loss_weight * velocity_loss

        # Compute additional metrics for logging
        with torch.no_grad():
            signal_power = torch.mean(trajectories ** 2)
            noise_power = torch.mean(noise ** 2)
            snr = 10 * torch.log10(signal_power / noise_power)
            mean_timestep = torch.mean(t.float()) / self.n_timesteps
            metrics = {
                'snr_db': snr.item(),
                'avg_timestep': mean_timestep.item(),
                'loss_angle': angle_loss.item(),
                'loss_velocity': velocity_loss.item(),
                'loss_total': loss.item()
            }
        return loss, metrics

    @torch.no_grad()
    def sample(self, batch_size=1, horizon=None, device=None):
        """
        Generate samples using the reverse (denoising) diffusion process.
        """
        self.eval()
        device = device or next(self.parameters()).device
        horizon = horizon or self.horizon

        print(f"\nStarting sampling process with {self.n_timesteps} denoising steps")
        
        # Initialize from pure noise.
        x = torch.randn(batch_size, horizon, self.transition_dim, device=device)

        for t in reversed(range(self.n_timesteps)):
            if t % 100 == 0:
                print(f"Denoising step {t}/{self.n_timesteps-1}")
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

            with torch.cuda.amp.autocast(enabled=(device != "cpu")):
                predicted_noise = self(x, t_batch)

            alpha = self.alphas[t]
            alpha_cumprod = self.alphas_cumprod[t]

            noise = torch.randn_like(x) if t > 0 else 0
            x = (1 / torch.sqrt(alpha)) * (
                    x - ((1 - alpha) / torch.sqrt(1 - alpha_cumprod)) * predicted_noise
                ) + torch.sqrt(1 - alpha) * noise
        return x

    @torch.no_grad()
    def update_ema(self, ema_model, decay):
        """
        Update the exponential moving average (EMA) for stable training.
        """
        for param, ema_param in zip(self.parameters(), ema_model.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay) 