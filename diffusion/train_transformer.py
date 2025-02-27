from diffuser.models.transformer_temporal import TransformerMotionModel
from diffuser.models.diffusion_v4 import DiffusionV4
import torch.optim as optim
from data_loaders.motion_dataset_v2 import MotionDataset
from torch.utils.data import DataLoader
from itertools import cycle
import torch
import os
import logging
from typing import Dict, Any, Optional


class DiffusionTrainer:
    """
    A trainer class for diffusion models on motion data.
    """
    def __init__(
        self,
        dataset_path: str,
        model_config: Dict[str, Any],
        diffusion_config: Dict[str, Any],
        training_config: Dict[str, Any],
        optimizer_config: Dict[str, Any],
        save_path: str,
    ):
        """
        Initialize the trainer with configurations.
        
        Args:
            dataset_path: Path to the motion dataset
            model_config: Configuration for the transformer model
            diffusion_config: Configuration for the diffusion model
            training_config: Configuration for training (batch size, steps, etc.)
            optimizer_config: Configuration for the optimizer
            save_path: Path to save trained models
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")
        
        # Set up dataset and dataloader
        self.dataset = MotionDataset(dataset_path)
        self.batch_size = training_config.get("batch_size", 4)
        self.dataloader = self._create_dataloader()
        
        # Set up model
        self.model = self._create_model(model_config)
        
        # Set up diffusion
        self.diffusion = self._create_diffusion(diffusion_config)
        
        # Set up optimizer
        self.optimizer = self._create_optimizer(optimizer_config)
        
        # Training settings
        self.num_train_steps = training_config.get("num_train_steps", 1000)
        self.log_interval = training_config.get("log_interval", 100)
        
        # Save settings
        self.save_path = save_path
        self.save_interval = training_config.get("save_interval", self.num_train_steps)
        
        # Create save directory if it doesn't exist
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
    
    def _create_dataloader(self):
        """Create a cyclic dataloader from the dataset."""
        return cycle(DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=1,
            shuffle=True,
            pin_memory=True
        ))
    
    def _create_model(self, config: Dict[str, Any]):
        """Create and initialize the transformer model."""
        model = TransformerMotionModel(
            input_dim=self.dataset[0].trajectories.shape[1],
            latent_dim=config.get("latent_dim", 256),
            n_heads=config.get("n_heads", 4),
            num_layers=config.get("num_layers", 8),
            dropout=config.get("dropout", 0.1),
            ff_size=config.get("ff_size", 1024)
        ).to(self.device)
        
        # Load pretrained weights if provided
        if "pretrained_path" in config and config["pretrained_path"]:
            model.load_state_dict(torch.load(config["pretrained_path"]))
            logging.info(f"Loaded pretrained model from {config['pretrained_path']}")
            
        return model
    
    def _create_diffusion(self, config: Dict[str, Any]):
        """Create the diffusion model."""
        return DiffusionV4(
            noise_steps=config.get("noise_steps", 50),
            beta_start=config.get("beta_start", 0.0001),
            beta_end=config.get("beta_end", 0.02),
            joint_dim=self.dataset[0].trajectories.shape[1],
            frames=self.dataset[0].trajectories.shape[0],
            device=self.device,
            predict_x0=config.get("predict_x0", False)
        )
    
    def _create_optimizer(self, config: Dict[str, Any]):
        """Create the optimizer for the model."""
        return optim.AdamW(
            self.model.parameters(),
            lr=config.get("lr", 0.001),
            weight_decay=config.get("weight_decay", 0.0001),
            eps=config.get("eps", 1e-8)
        )
    
    def train(self):
        """Execute the training loop."""
        self.model.to(self.device)  # Ensure model is on the correct device
        
        for step in range(self.num_train_steps):
            # Get batch and move to device
            batch = next(self.dataloader)
            x_start = batch.trajectories.to(self.device)
            
            # Print shape information on first step
            if step == 0:
                logging.info(f"Input shape: {x_start.shape}")
            
            # Sample timesteps and compute loss
            t = self.diffusion.sample_timesteps(x_start.shape[0])
            loss = self.diffusion.training_loss(self.model, x_start, t)
            
            # Optimization step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Logging
            if step % self.log_interval == 0:
                logging.info(f"Step {step}/{self.num_train_steps} | Loss: {loss.item():.6f}")
            
            # Save checkpoint
            if (step + 1) % self.save_interval == 0 or step == self.num_train_steps - 1:
                self._save_checkpoint(step + 1)
    
    def _save_checkpoint(self, step: int):
        """Save a model checkpoint."""
        checkpoint_path = os.path.join(
            self.save_path, 
            f"model_step_{step}_noise_{self.diffusion.noise_steps}.pth"
        )
        torch.save(self.model.state_dict(), checkpoint_path)
        logging.info(f"Saved checkpoint to {checkpoint_path}")
    
    def generate_samples(self, num_samples: int = 1):
        """Generate samples from the trained model."""
        self.model.eval()
        samples = self.diffusion.sample(self.model, num_samples)
        return samples


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Configuration
    model_config = {
        "latent_dim": 256,
        "n_heads": 4,
        "num_layers": 8,
        "dropout": 0.1,
        "ff_size": 1024,
        "pretrained_path": None  # Optional path to pretrained weights
    }
    
    diffusion_config = {
        "noise_steps": 50,
        "beta_start": 0.0001,
        "beta_end": 0.02,
        "predict_x0": True
    }
    
    training_config = {
        "batch_size": 4,
        "num_train_steps": 100,
        "log_interval": 10,
        "save_interval": 100
    }
    
    optimizer_config = {
        "lr": 0.001,
        "weight_decay": 0.0001,
        "eps": 1e-8
    }
    
    # Create trainer
    trainer = DiffusionTrainer(
        dataset_path="data/motions/humanoid3d_walk.txt",
        model_config=model_config,
        diffusion_config=diffusion_config,
        training_config=training_config,
        optimizer_config=optimizer_config,
        save_path="models/transformer_motion_model"
    )
    
    # Run training
    trainer.train()