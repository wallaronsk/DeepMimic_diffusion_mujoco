from diffuser.models.transformer_temporal import TransformerMotionModel
from diffuser.models.transformer_local_attention import LocalTransformer as TransformerLocalAttention
from diffuser.models.diffusion_v4 import DiffusionV4
from diffuser.models.temporal_v2 import TemporalUnet
import torch.optim as optim
from data_loaders.motion_dataset_v2 import MotionDataset
from torch.utils.data import DataLoader
from itertools import cycle
import torch
import os
import logging
from typing import Dict, Any, Optional
import datetime
import time
import copy
import argparse
import itertools
import json
from pathlib import Path

class EMA:
    def __init__(self, beta):
        self.beta = beta
        self.step = 0
        self.ema_start = 2000
        
    def update_model_average(self, ema_model, model):
        for current_params, ema_params in zip(model.parameters(), ema_model.parameters()):
            old_weight, new_weight = ema_params.data, current_params.data
            ema_params.data = self.update_average(old_weight, new_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new
    
    def step_ema(self, ema_model, model):
        if self.step < self.ema_start:
            self.step += 1
            return
        self.step += 1
        self.update_model_average(ema_model, model)
    
    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


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
        data_type: str = 'both',
        architecture: str = 'transformer',
    ):
        """
        Initialize the trainer with configurations.
        
        Args:
            dataset_path: Path to the motion dataset
            model_config: Configuration for the model
            diffusion_config: Configuration for the diffusion model
            training_config: Configuration for training (batch size, steps, etc.)
            optimizer_config: Configuration for the optimizer
            save_path: Path to save trained models
            data_type: Type of data to use - 'positions', 'velocities', or 'both'
            architecture: Type of model architecture - 'transformer' or 'temporal'
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_type = data_type
        self.architecture = architecture
        logging.info(f"Using device: {self.device}")
        logging.info(f"Data type: {self.data_type}")
        logging.info(f"Architecture: {self.architecture}")
        
        # Set up dataset and dataloader
        self.dataset = MotionDataset(dataset_path, data_type=self.data_type, shuffle=False)
        self.batch_size = training_config.get("batch_size", 4)
        self.dataloader = self._create_dataloader()
        
        # Set up model
        self.model = self._create_model(model_config)
        # print number of parameters in the model
        logging.info(f"Number of parameters in the model: {sum(p.numel() for p in self.model.parameters())}")

        # Set up EMA
        self.ema = EMA(beta=0.995)
        self.ema_model = copy.deepcopy(self.model).eval().requires_grad_(False).to(self.device)
        
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
        
        # Generate a timestamp for this training run
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        logging.info(f"Training run timestamp: {self.timestamp}")
        
        # Create save directory if it doesn't exist
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        scheduler_type = optimizer_config.get("scheduler_type", "ExponentialLR")
        
        # calculate the learning rate decay factor based on start and end learning rates

        if scheduler_type == "cosine":
            # Add learning rate scheduler
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=self.num_train_steps,
                eta_min=1e-5
            )
        elif scheduler_type == "linear":
            self.scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer, 
                start_factor=1, 
                end_factor=5e-1, 
                total_iters=self.num_train_steps)
        elif scheduler_type == "exponential":
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=0.99998
            )
    
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
        """Create and initialize the model based on the selected architecture."""
        input_dim = self.dataset[0].trajectories.shape[1]
        sequence_length = self.dataset[0].trajectories.shape[0]
        print(config)
        if self.architecture == 'transformer':
            model = TransformerMotionModel(
                input_dim=input_dim,
                latent_dim=config.get("latent_dim", 256),
                n_heads=config.get("n_heads", 4),
                num_layers=config.get("num_layers", 8),
                dropout=config.get("dropout", 0.1),
                dim_feedforward=config.get("dim_feedforward", 512)
            ).to(self.device)
        elif self.architecture == 'temporal':
            # Create TemporalUnet model
            model = TemporalUnet(
                horizon=sequence_length,
                transition_dim=input_dim,
                cond_dim=0,  # No conditioning by default
                dim=config.get("channel_dim", 128),
                dim_mults=config.get("dim_mults", (1, 2, 4, 8)),
                attention=config.get("attention", False)
            ).to(self.device)
        elif self.architecture == 'local_attention':
            model = TransformerLocalAttention(
                dim=config.get("dim", 512),
                depth=config.get("depth", 6),
                causal=config.get("causal", False),
                local_attn_window_size=config.get("local_attn_window_size", 4),
                max_seq_len=sequence_length,
                input_dim=input_dim
            ).to(self.device)
        else:
            raise ValueError(f"Unsupported architecture: {self.architecture}")
            
        # Load pretrained weights if provided
        if "pretrained_path" in config and config["pretrained_path"]:
            model.load_state_dict(torch.load(config["pretrained_path"])['model_state_dict'])
            logging.info(f"Loaded pretrained model from {config['pretrained_path']}")
            
        return model
    
    def _create_diffusion(self, config: Dict[str, Any]):
        """Create the diffusion model."""
        return DiffusionV4(
            noise_steps=config.get("noise_steps", 50),
            beta_start=config.get("beta_start", 1e-4),
            beta_end=config.get("beta_end", 0.02),
            joint_dim=self.dataset[0].trajectories.shape[1],  # Use actual joint dim from dataset
            frames=self.dataset[0].trajectories.shape[0],
            device=self.device,
            predict_x0=config.get("predict_x0", False),
            schedule_type=config.get("schedule_type", "linear"),
            cosine_s=config.get("cosine_s", 0.008),
        )
    
    def _create_optimizer(self, config: Dict[str, Any]):
        """Create the optimizer for the model."""
        optimizer_type = config.get("optimizer_type", "adamw").lower()
        print(f"Using optimizer type: {optimizer_type}")
        
        if optimizer_type == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=config.get("lr", 2e-4),
                weight_decay=config.get("weight_decay", 1e-4),
                eps=config.get("eps", 1e-8),
                betas=config.get("betas", [0.9, 0.995])
            )
        elif optimizer_type == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=config.get("lr", 2e-4),
                weight_decay=config.get("weight_decay", 1e-4),
                eps=config.get("eps", 1e-8),
                betas=config.get("betas", [0.9, 0.995])
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}. Choose either 'adam' or 'adamw'.")
    
    def train(self):
        """Execute the training loop."""
        self.model.to(self.device)  # Ensure model is on the correct device
        
        # Dictionary to track training metrics
        training_metrics = {
            "steps": [],
            "losses": [],
            "final_loss": None,
            "best_loss": float('inf'),
            "checkpoint_paths": [],
            "best_model_path": None,  # Add path for best model
            "best_model_step": None   # Add step for best model
        }
        
        # Variables to track the best model in the last 15% of training
        best_model_state = None
        best_loss_final_phase = float('inf')
        final_phase_start = int(self.num_train_steps * 0.85)  # Last 15% of training
        
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

            self.ema.step_ema(self.ema_model, self.model)
            
            # Learning rate step
            self.scheduler.step()
            
            # Track metrics
            if step % self.log_interval == 0 or step == self.num_train_steps - 1:
                current_loss = loss.item()
                training_metrics["steps"].append(step)
                training_metrics["losses"].append(current_loss)
                
                # Check if this is the best loss so far
                if current_loss < training_metrics["best_loss"]:
                    training_metrics["best_loss"] = current_loss
                
                # Track best model in final phase (last 15% of training)
                if step >= final_phase_start and current_loss < best_loss_final_phase:
                    best_loss_final_phase = current_loss
                    # Save model state (make a deep copy to avoid reference issues)
                    best_model_state = copy.deepcopy(self.model.state_dict())
                    training_metrics["best_model_step"] = step
                    logging.info(f"New best model in final phase at step {step} with loss {best_loss_final_phase:.6f}")
            
            # Logging
            if step % self.log_interval == 0:
                log_message = f"Step {step}/{self.num_train_steps} | Loss: {loss.item():.6f}"
                
                # Add information about final phase tracking if applicable
                if step >= final_phase_start:
                    log_message += f" | Final Phase Best Loss: {best_loss_final_phase:.6f}"
                
                logging.info(log_message)
            
            if self.save_interval is not None:
                # Save checkpoint
                if (step + 1) % self.save_interval == 0 or step == self.num_train_steps - 1:
                    checkpoint_paths = self._save_checkpoint(step + 1, loss.item())
                    training_metrics["checkpoint_paths"].append(checkpoint_paths)
            else:
                # save checkpoint at the end of training only   
                if step == self.num_train_steps - 1:
                    checkpoint_paths = self._save_checkpoint(step + 1, loss.item())
                    training_metrics["checkpoint_paths"].append(checkpoint_paths)
        
        # Save final metrics
        training_metrics["final_loss"] = training_metrics["losses"][-1] if training_metrics["losses"] else None
        
        # Save the best model from the final phase if one was found
        if best_model_state is not None:
            best_model_path = self._save_best_model(best_model_state, best_loss_final_phase, training_metrics["best_model_step"])
            training_metrics["best_model_path"] = best_model_path
            logging.info(f"Saved best model from final phase with loss {best_loss_final_phase:.6f}")
        
        # Save training metrics to JSON
        metrics_path = os.path.join(self.save_path, "training_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(training_metrics, f, indent=4)
        
        logging.info(f"Training metrics saved to {metrics_path}")
        
        return training_metrics
    
    def _save_checkpoint(self, step: int, loss: float = None):
        """Save a model checkpoint with timestamp and configuration details."""
        # Create a model ID with key configuration details
        predict_x0_tag = "x0" if self.diffusion.predict_x0 else "eps"
        arch_tag = self.architecture
        
        model_id = f"{self.timestamp}_{arch_tag}_{predict_x0_tag}"
        
        # Create filename with step information and loss if provided
        if loss is not None:
            filename = f"model_{model_id}_step{step}_loss{loss:.6f}_ns{self.diffusion.noise_steps}.pth"
        else:
            filename = f"model_{model_id}_step{step}_ns{self.diffusion.noise_steps}.pth"
        checkpoint_path = os.path.join(self.save_path, filename)
        
        # Save model state
        torch.save({
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,  # Include loss in the saved state
            'architecture': self.architecture,  # Save the architecture type
            'timestamp': self.timestamp,
            'predict_x0': self.diffusion.predict_x0,
            'noise_steps': self.diffusion.noise_steps,
            'schedule_type': self.diffusion.schedule_type,
            'cosine_s': self.diffusion.cosine_s
        }, checkpoint_path)
        
        logging.info(f"Saved checkpoint to {checkpoint_path}")

        # print the model size
        model_size = sum(p.numel() for p in self.model.parameters())
        logging.info(f"Model size: {model_size}")
        
        # Save EMA model state with a different naming scheme
        ema_filename = f"model_ema_{model_id}_step{step}_ns{self.diffusion.noise_steps}.pth"
        ema_checkpoint_path = os.path.join(self.save_path, ema_filename)
        
        # Save EMA model state
        torch.save({
            'step': step,
            'model_state_dict': self.ema_model.state_dict(),
            'architecture': self.architecture,  # Save the architecture type
            'timestamp': self.timestamp,
            'predict_x0': self.diffusion.predict_x0,
            'noise_steps': self.diffusion.noise_steps,
            'schedule_type': self.diffusion.schedule_type,
            'cosine_s': self.diffusion.cosine_s
        }, ema_checkpoint_path)
        
        logging.info(f"Saved EMA checkpoint to {ema_checkpoint_path}")
        
        return {"model_path": checkpoint_path, "ema_model_path": ema_checkpoint_path}
    
    def _save_best_model(self, model_state_dict, loss, step):
        """Save the best model from the final training phase."""
        # Create a model ID with key configuration details
        predict_x0_tag = "x0" if self.diffusion.predict_x0 else "eps"
        arch_tag = self.architecture
        
        model_id = f"{self.timestamp}_{arch_tag}_{predict_x0_tag}"
        
        # Create filename with best model information
        filename = f"best_model_{model_id}_step{step}_loss{loss:.6f}.pth"
        best_model_path = os.path.join(self.save_path, filename)
        
        # Save model state
        torch.save({
            'step': step,
            'model_state_dict': model_state_dict,
            'loss': loss,
            'architecture': self.architecture,
            'timestamp': self.timestamp,
            'predict_x0': self.diffusion.predict_x0,
            'noise_steps': self.diffusion.noise_steps,
            'schedule_type': self.diffusion.schedule_type,
            'cosine_s': self.diffusion.cosine_s
        }, best_model_path)
        
        logging.info(f"Saved best model from final phase to {best_model_path}")
        
        return best_model_path
    
    def generate_samples(self, num_samples: int = 1):
        """Generate samples from the trained model."""
        self.model.eval()
        samples = self.diffusion.sample(self.model, num_samples)
        return samples


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train diffusion model with hyperparameter tuning")
    parser.add_argument("--dataset", type=str, default="data/motions/humanoid3d_walk.txt", help="Path to dataset")
    parser.add_argument("--architecture", type=str, default="local_attention", 
                        choices=["transformer", "temporal", "local_attention"],  # Updated choices
                        help="Architecture to use for the model")
    parser.add_argument("--experiments_dir", type=str, default="experiments", help="Directory to save all experiments")
    parser.add_argument("--sweep", action="store_true", help="Run hyperparameter sweep")
    parser.add_argument("--config", type=str, help="Configuration file for hyperparameter sweep")
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    if args.architecture == "local_attention":
        default_local_attention_config = {
            "dim": 512,
            "depth": 6,
            "causal": False,
            "local_attn_window_size": 512,
            "max_seq_len": 69,
            "input_dim": 69
        }
        default_model_config = default_local_attention_config
    else:
        default_model_config = {
            "latent_dim": 1024,
            "n_heads": 8,
            "num_layers": 4,
            "dropout": 0.25,
            "dim_feedforward": 32,
            "pretrained_path": None,
            # TemporalUnet specific parameters
            "channel_dim": 128,
            "dim_mults": (1, 2, 4, 8),
            "attention": False,
        }
    
    default_diffusion_config = {
        "noise_steps": 1000,
        "beta_start": 1e-4,
        "beta_end": 0.02,
        "predict_x0": True,
        "schedule_type": "cosine",
        "cosine_s": 0.008,
    }
    
    default_training_config = {
        "batch_size": 64,
        "num_train_steps": 1000,
        "log_interval": 100,
        "save_interval": None
    }
    
    default_optimizer_config = {
        "lr": 1e-4,
        "weight_decay": 0,
        "betas": [0.9, 0.98],
        "optimizer_type": "adamw",
        "scheduler_type": "linear"
    }
    
    def run_single_experiment(model_config, diffusion_config, training_config, optimizer_config, 
                            dataset_path, architecture, experiment_name):
        """Run a single training experiment with the given configurations."""
        # Create a unique experiment directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(args.experiments_dir, f"{experiment_name}_{timestamp}")
        os.makedirs(save_dir, exist_ok=True)
        
        # Save configuration for reference
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            config = {
                "model_config": model_config,
                "diffusion_config": diffusion_config,
                "training_config": training_config,
                "optimizer_config": optimizer_config,
                "dataset_path": dataset_path,
                "architecture": architecture,
                "timestamp": timestamp
            }
            json.dump(config, f, indent=4)
        
        # Create trainer
        trainer = DiffusionTrainer(
            dataset_path=dataset_path,
            model_config=model_config,
            diffusion_config=diffusion_config,
            training_config=training_config,
            optimizer_config=optimizer_config,
            save_path=save_dir,
            architecture=architecture
        )
        
        # Run training
        logging.info(f"Starting experiment: {experiment_name}")
        training_metrics = trainer.train()
        logging.info(f"Finished experiment: {experiment_name}")
        
        # Return experiment results including metrics
        return {
            "save_dir": save_dir,
            "final_loss": training_metrics["final_loss"],
            "best_loss": training_metrics["best_loss"],
            "experiment_name": experiment_name,
            "timestamp": timestamp
        }

    if args.sweep and args.config:
        # Load hyperparameter sweep configuration
        with open(args.config, "r") as f:
            sweep_config = json.load(f)
        
        # Extract hyperparameter ranges
        model_params = sweep_config.get("model_params", {})
        diffusion_params = sweep_config.get("diffusion_params", {})
        training_params = sweep_config.get("training_params", {})
        optimizer_params = sweep_config.get("optimizer_params", {})
        architecture_params = sweep_config.get("architecture_params", {"architecture": ["transformer"]})
        
        # Create parameter grid
        model_grid = list(dict(zip(model_params.keys(), values)) 
                        for values in itertools.product(*model_params.values()))
        diffusion_grid = list(dict(zip(diffusion_params.keys(), values))
                            for values in itertools.product(*diffusion_params.values()))
        training_grid = list(dict(zip(training_params.keys(), values))
                            for values in itertools.product(*training_params.values()))
        optimizer_grid = list(dict(zip(optimizer_params.keys(), values))
                            for values in itertools.product(*optimizer_params.values()))
        architecture_grid = list(architecture_params.get("architecture", ["transformer"]))
        
        # If any grid is empty, use a single default configuration
        if not model_grid:
            model_grid = [{}]
        if not diffusion_grid:
            diffusion_grid = [{}]
        if not training_grid:
            training_grid = [{}]
        if not optimizer_grid:
            optimizer_grid = [{}]
        
        # Run experiments for all combinations
        experiment_results = []
        experiment_count = len(model_grid) * len(diffusion_grid) * len(training_grid) * len(optimizer_grid) * len(architecture_grid)
        logging.info(f"Running {experiment_count} experiments with different hyperparameters")
        
        # Create experiment directory if it doesn't exist
        Path(args.experiments_dir).mkdir(exist_ok=True)
        
        # Variables to track best experiment and interim summaries
        best_experiment = None
        summary_counter = 0
        
        experiment_idx = 0
        for architecture in architecture_grid:
            for model_params, diffusion_params, training_params, optimizer_params in itertools.product(
                model_grid, diffusion_grid, training_grid, optimizer_grid
            ):
                experiment_idx += 1
                # Create configurations by updating defaults with sweep parameters
                curr_model_config = default_model_config.copy()
                curr_model_config.update(model_params)
                
                curr_diffusion_config = default_diffusion_config.copy()
                curr_diffusion_config.update(diffusion_params)
                
                curr_training_config = default_training_config.copy()
                curr_training_config.update(training_params)
                
                curr_optimizer_config = default_optimizer_config.copy()
                curr_optimizer_config.update(optimizer_params)
                
                # Create experiment name summarizing key parameters
                experiment_name = f"exp_{experiment_idx}"
                experiment_name += f"_{architecture}"
                if "latent_dim" in model_params:
                    experiment_name += f"_ld{model_params['latent_dim']}"
                if "n_heads" in model_params and architecture == "transformer":
                    experiment_name += f"_nh{model_params['n_heads']}"
                if "num_layers" in model_params:
                    experiment_name += f"_nl{model_params['num_layers']}"
                if "noise_steps" in diffusion_params:
                    experiment_name += f"_ns{diffusion_params['noise_steps']}"
                if "lr" in optimizer_params:
                    experiment_name += f"_lr{optimizer_params['lr']}"
                
                # When building experiment names, add a condition for temporal model
                if architecture == "temporal":
                    if "latent_dim" in model_params:
                        experiment_name += f"_ld{model_params['latent_dim']}"
                    if "attention" in model_params:
                        experiment_name += f"_attn{int(model_params['attention'])}"
                    if "dim_mults" in model_params:
                        experiment_name += f"_dm{'-'.join(map(str, model_params['dim_mults']))}"
                
                logging.info(f"Running experiment {experiment_idx}/{experiment_count}: {experiment_name}")
                
                # Run the experiment
                experiment_result = run_single_experiment(
                    curr_model_config, curr_diffusion_config, curr_training_config, curr_optimizer_config,
                    args.dataset, architecture, experiment_name
                )
                
                # Add configuration details to the result
                experiment_result.update({
                    "model_config": curr_model_config,
                    "diffusion_config": curr_diffusion_config,
                    "training_config": curr_training_config,
                    "optimizer_config": curr_optimizer_config,
                    "architecture": architecture,
                    "experiment_index": experiment_idx
                })
                
                experiment_results.append(experiment_result)
                
                # Update best experiment if needed
                if best_experiment is None or experiment_result["best_loss"] < best_experiment["best_loss"]:
                    best_experiment = experiment_result
                
                # Save interim summary every 10 experiments
                summary_counter += 1
                if summary_counter >= 10 or experiment_idx == experiment_count:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    interim_summary_path = os.path.join(args.experiments_dir, f"interim_summary_{timestamp}_exp{experiment_idx}.json")
                    
                    # Create summary with current results and best experiment
                    interim_summary = {
                        "experiments_completed": experiment_idx,
                        "total_experiments": experiment_count,
                        "best_experiment": best_experiment,
                        "all_results": experiment_results
                    }
                    
                    # Save interim summary
                    with open(interim_summary_path, "w") as f:
                        json.dump(interim_summary, f, indent=4)
                    
                    logging.info(f"Saved interim summary after {experiment_idx} experiments to {interim_summary_path}")
                    summary_counter = 0  # Reset counter
        
        # Save final summary of all experiments
        final_summary_path = os.path.join(args.experiments_dir, f"final_summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(final_summary_path, "w") as f:
            final_summary = {
                "experiments_completed": experiment_count,
                "total_experiments": experiment_count,
                "best_experiment": best_experiment,
                "all_results": experiment_results
            }
            json.dump(final_summary, f, indent=4)
        
        logging.info(f"Completed all {experiment_count} experiments. Final summary saved to {final_summary_path}")
        logging.info(f"Best experiment: {best_experiment['experiment_name']} with loss {best_experiment['best_loss']:.6f}")
        logging.info(f"Best experiment directory: {best_experiment['save_dir']}")
    
    else:
        # Run a single experiment with default configuration
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        predict_x0_tag = "predict_x0" if default_diffusion_config["predict_x0"] else "predict_eps"
        arch_tag = args.architecture
        experiment_name = f"{arch_tag}_{predict_x0_tag}"
        
        experiment_result = run_single_experiment(
            default_model_config, default_diffusion_config, default_training_config, default_optimizer_config,
            args.dataset, args.architecture, experiment_name
        )
        
        logging.info(f"Single experiment completed with best loss: {experiment_result['best_loss']:.6f}")