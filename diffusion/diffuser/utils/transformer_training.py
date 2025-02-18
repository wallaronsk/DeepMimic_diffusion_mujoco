import os
import copy
import torch
import numpy as np
from .timer import Timer
import mlflow

def cycle(dl):
    while True:
        for data in dl:
            yield data

class EMA():
    """Exponential Moving Average for model parameters"""
    def __init__(self, beta=0.995):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ema_model, current_model):
        for current_params, ema_params in zip(current_model.parameters(), ema_model.parameters()):
            old_weight, new_weight = ema_params.data, current_params.data
            ema_params.data = old_weight * self.beta + (1 - self.beta) * new_weight

    def update_beta(self, beta):
        self.beta = beta

class TransformerTrainer:
    def __init__(
        self,
        diffusion_model,
        dataset,
        train_batch_size=32,
        train_lr=1e-4,
        log_freq=100,
        results_folder='./results',
        ema_decay=0.995,
        mask_ratio=0.15,
        mlflow_run=None
    ):
        self.model = diffusion_model
        self.dataset = dataset
        self.batch_size = train_batch_size
        self.log_freq = log_freq
        self.results_folder = results_folder
        self.mask_ratio = mask_ratio
        
        # Create EMA model
        self.ema = EMA(beta=ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.ema_model.eval()
        
        # Create dataloader with non-blocking GPU transfers
        self.dataloader = cycle(torch.utils.data.DataLoader(
            self.dataset,
            batch_size=train_batch_size,
            num_workers=4,
            shuffle=True,
            pin_memory=True
        ))
        
        # Initialize optimizer without warmup
        self.optimizer = torch.optim.AdamW(
            diffusion_model.parameters(),
            lr=train_lr,
            betas=(0.9, 0.999),
            weight_decay=0.01
        )
        
        # --- New: Set up learning rate warmup parameters ---
        self.warmup_steps = 10000
        self.lr_scheduler = None  # Will be initialized in train()
        
        self.step = 0
        os.makedirs(results_folder, exist_ok=True)
        self.mlflow_run = mlflow_run
        
        # Initialize metric tracking
        self.loss_history = []
        self.lr_history = []
        self.grad_norm_history = []

        # Add noise schedule tracking
        self.register_noise_schedule_metrics()

        # Enable AMP (Automatic Mixed Precision) if CUDA is available
        self.use_amp = torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

    def register_noise_schedule_metrics(self):
        """Register noise schedule related metrics for visualization"""
        self.noise_metrics = {
            'mean_snr': [],  # Signal-to-Noise Ratio
            'mean_noise_level': [],
        }

    def get_noise_level(self, t):
        """Calculate noise level at timestep t"""
        return self.model.sqrt_one_minus_alphas_cumprod[t]

    def apply_frame_masking(self, batch):
        """Apply random frame masking to the batch"""
        if isinstance(batch, dict):
            trajectories = batch[0]
        else:
            trajectories = batch
            
        B, T, D = trajectories.shape
        device = trajectories.device
        
        # Determine number of frames to mask
        num_masks = int(T * self.mask_ratio)
        
        # Create mask for each sequence in batch
        masks = []
        for _ in range(B):
            # Randomly select frames to mask
            mask_indices = torch.randperm(T)[:num_masks]
            mask = torch.ones(T, device=device, dtype=torch.bool)
            mask[mask_indices] = False
            masks.append(mask)
            
        # Stack masks and expand to match trajectory dimensions
        masks = torch.stack(masks)  # [B, T]
        masks = masks.unsqueeze(-1).expand(-1, -1, D)  # [B, T, D]
        
        # Apply masking (replace masked frames with zeros)
        masked_trajectories = trajectories * masks
        
        if isinstance(batch, dict):
            batch[0] = masked_trajectories
            return batch
        return masked_trajectories

    def _lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / self.warmup_steps
        else:
            decay_steps = self.total_steps - self.warmup_steps
            # Linearly decay from 1.0 to 0.1 (i.e. 1e-4 to 1e-5)
            return max(0.1, 1 - (step - self.warmup_steps) / decay_steps * 0.9)

    def log_metrics(self, loss, infos, grad_norm, lr):
        """Log metrics to MLflow"""
        if self.mlflow_run:
            metrics = {
                'loss': loss,
                'learning_rate': lr,
                'gradient_norm': grad_norm,
                **infos
            }
            mlflow.log_metrics(metrics, step=self.step)
            
            # Store in history for rolling averages
            self.loss_history.append(loss)
            self.lr_history.append(lr)
            self.grad_norm_history.append(grad_norm)
            
            # Calculate and log rolling averages (over last 100 steps)
            window = 100
            if len(self.loss_history) >= window:
                rolling_metrics = {
                    'loss_avg_100': np.mean(self.loss_history[-window:]),
                    'grad_norm_avg_100': np.mean(self.grad_norm_history[-window:])
                }
                mlflow.log_metrics(rolling_metrics, step=self.step)

    def get_checkpoint_steps(self, n_train_steps):
        """Calculate 5 evenly distributed checkpoint steps including start and end"""
        return [int(step) for step in np.linspace(0, n_train_steps - 1, 5)]

    def get_checkpoint_steps_only_end(self, n_train_steps):
        """Only save checkpoint at the end of the training"""
        return [n_train_steps - 1]
    
    def debug_model_devices(self):
        """Print out which device each model parameter is on."""
        print("===== Debug: Model Parameter Device Info =====")
        for name, param in self.model.named_parameters():
            print(f"{name}: {param.device}")
        print("===============================================")

    def debug_memory_usage(self):
        """Print GPU memory allocation details (if CUDA is available)."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 ** 2)
            reserved = torch.cuda.memory_reserved() / (1024 ** 2)
            print(f"===== Debug: GPU Memory Usage (MB): Allocated={allocated:.2f}, Reserved={reserved:.2f} =====")
    
    def train(self, n_train_steps):
        timer = Timer()
        
        # Initialize learning rate scheduler now that total steps are known.
        self.total_steps = n_train_steps
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda step: self._lr_lambda(step))
        
        # Debug: print model devices and current memory usage before training
        if self.step == 0:
            self.debug_model_devices()
            self.debug_memory_usage()
        
        # Calculate checkpoint steps at the start
        checkpoint_steps = self.get_checkpoint_steps_only_end(n_train_steps)
        print(f"Will save checkpoints at steps: {checkpoint_steps}")
        
        for step in range(n_train_steps):
            self.model.train()
            self.ema_model.eval()
            
            # Get batch and move to GPU with non-blocking transfers
            batch = next(self.dataloader)[0]
            if isinstance(batch, dict):
                batch = {k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v 
                         for k, v in batch.items()}
            else:
                batch = batch.cuda(non_blocking=True)
            
            # Apply frame masking
            masked_batch = self.apply_frame_masking(copy.deepcopy(batch))
            
            self.optimizer.zero_grad()
            # Forward pass with AMP autocasting if enabled
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    loss, infos = self.model.loss(masked_batch)
            else:
                loss, infos = self.model.loss(masked_batch)
            
            # Track noise schedule metrics
            with torch.no_grad():
                t = infos.get('timesteps', torch.tensor([0], device=batch.device))  # Ensure device consistency
                self.noise_metrics['mean_noise_level'].append(
                    self.get_noise_level(t).mean().item()
                )
                if 'snr_db' in infos:
                    self.noise_metrics['mean_snr'].append(infos['snr_db'])
            
            # Backward pass with AMP scaling if enabled
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            
            # --- New: Update learning rate scheduler after the optimizer step ---
            self.lr_scheduler.step()
            
            # Update EMA model
            self.ema.update_model_average(self.ema_model, self.model)
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log metrics
            self.log_metrics(
                loss=loss.item(),
                infos=infos,
                grad_norm=grad_norm.item(),
                lr=current_lr
            )
            
            # Console logging
            if self.step % self.log_freq == 0:
                infos_str = ' | '.join([f'{key}: {val:8.4f}' for key, val in infos.items()])
                print(f'{self.step}: {loss:8.4f} | {infos_str} | lr: {current_lr:8.6f} | grad_norm: {grad_norm:8.4f} | t: {timer():8.4f}')
                # Optionally, print GPU memory usage every log frequency steps
                self.debug_memory_usage()
            
            # Save checkpoint if current step is in checkpoint_steps
            if step in checkpoint_steps:
                self.save()
                # Log model checkpoint to MLflow
                if self.mlflow_run:
                    checkpoint_path = os.path.join(self.results_folder, f'state_{self.step}.pt')
                    mlflow.log_artifact(checkpoint_path)
            
            # Log noise schedule metrics periodically
            if self.step % self.log_freq == 0:
                self.log_noise_metrics()
            
            self.step += 1

    def save(self):
        """Save model checkpoint"""
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema_model': self.ema_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        savepath = os.path.join(self.results_folder, f'state_{self.step}.pt')
        torch.save(data, savepath)
        print(f'Saved checkpoint {self.step} to {savepath}')

        # Clean up old checkpoints to maintain exactly 5
        self._cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self):
        """Maintain only the 5 most recent checkpoints"""
        checkpoints = []
        for filename in os.listdir(self.results_folder):
            if filename.startswith('state_') and filename.endswith('.pt'):
                step = int(filename.replace('state_', '').replace('.pt', ''))
                checkpoints.append((step, filename))
        
        # Sort by step number
        checkpoints.sort()
        
        # If we have more than 5 checkpoints, remove the oldest ones
        if len(checkpoints) > 5:
            for _, filename in checkpoints[:-5]:
                filepath = os.path.join(self.results_folder, filename)
                try:
                    os.remove(filepath)
                    print(f'Removed old checkpoint: {filepath}')
                except Exception as e:
                    print(f'Failed to remove checkpoint {filepath}: {e}')

    def load(self, step):
        """Load model from checkpoint"""
        loadpath = os.path.join(self.results_folder, f'state_{step}.pt')
        data = torch.load(loadpath)
        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema_model'])
        self.optimizer.load_state_dict(data['optimizer'])

    def log_noise_metrics(self):
        """Log noise schedule related metrics"""
        if self.mlflow_run:
            metrics = {
                'noise/mean_snr': np.mean(self.noise_metrics['mean_snr'][-100:]),
                'noise/mean_level': np.mean(self.noise_metrics['mean_noise_level'][-100:]),
            }

            mlflow.log_metrics(metrics, step=self.step)
