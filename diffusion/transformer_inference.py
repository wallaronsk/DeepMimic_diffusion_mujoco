import torch
from diffuser.models.transformer_temporal import TransformerMotionModel
from diffuser.models.transformer_local_attention import LocalTransformer as TransformerLocalAttention
from diffuser.models.diffusion_v4 import DiffusionV4
from diffuser.models.temporal_v2 import TemporalUnet
from data_loaders.motion_dataset_v2 import MotionDataset
import os
import numpy as np
import logging
from typing import Dict, Any, Optional, Union, List, Tuple
import datetime
import json


class DiffusionInference:
    """
    A class for running inference with trained diffusion models on motion data.
    """
    def __init__(
        self,
        model_path: str,
        dataset_path: str,
        diffusion_config: Dict[str, Any],
        model_config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
        data_type: Optional[str] = None,
        architecture: Optional[str] = None
    ):
        """
        Initialize the inference engine.
        
        Args:
            model_path: Path to the trained model checkpoint
            dataset_path: Path to the motion dataset (needed for dimensions)
            diffusion_config: Configuration for the diffusion process
            model_config: Optional configuration for model architecture
            device: Device to run inference on (defaults to CUDA if available)
            data_type: Type of data to use - 'positions', 'velocities', or 'both'
                       If None, will be loaded from checkpoint if available,
                       otherwise defaults to 'both'
            architecture: Type of model architecture - 'transformer' or 'temporal'
                          If None, will be loaded from checkpoint if available,
                          otherwise defaults to 'transformer'
        """
        # Set device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load checkpoint first to get data_type and architecture if available
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
            
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Determine data_type - prioritize explicitly provided value, then checkpoint, then default
        if data_type is not None:
            self.data_type = data_type
            logging.info(f"Using provided data_type: {self.data_type}")
        else:
            # Extract data_type from checkpoint or use default
            checkpoint_data_type = self._get_from_checkpoint(checkpoint, 'data_type')
            if checkpoint_data_type:
                self.data_type = checkpoint_data_type
                logging.info(f"Using data_type from checkpoint: {self.data_type}")
            else:
                self.data_type = 'both'
                logging.info(f"No data_type found in checkpoint. Using default: {self.data_type}")

        # Determine architecture - prioritize explicitly provided value, then checkpoint, then default
        if architecture is not None:
            self.architecture = architecture
            logging.info(f"Using provided architecture: {self.architecture}")
        else:
            # Extract architecture from checkpoint or use default
            checkpoint_architecture = self._get_from_checkpoint(checkpoint, 'architecture')
            if checkpoint_architecture:
                self.architecture = checkpoint_architecture
                logging.info(f"Using architecture from checkpoint: {self.architecture}")
            else:
                self.architecture = 'transformer'
                logging.info(f"No architecture found in checkpoint. Using default: {self.architecture}")
        
        # Set up dataset with correct data_type
        self.dataset = MotionDataset(dataset_path, data_type=self.data_type)
        self.joint_dim = self.dataset[0].trajectories.shape[1]
        self.frames = self.dataset[0].trajectories.shape[0]
        
        # Now load the model with the correct input dimensions
        self.model = self._load_model(model_path, model_config, checkpoint)
        
        # Set up diffusion
        self.diffusion = self._create_diffusion(diffusion_config)
    
    def _load_model(self, model_path: str, model_config: Optional[Dict[str, Any]], checkpoint: Optional[Dict] = None) -> torch.nn.Module:
        """
        Load the trained model from a checkpoint.
        
        Args:
            model_path: Path to the model checkpoint
            model_config: Configuration for the model architecture (if loading state_dict)
            checkpoint: Checkpoint dictionary if available
            
        Returns:
            The loaded model
        """
        logging.info(f"Loading model from {model_path} with architecture {self.architecture}")
        
        # Determine input dimensions from dataset
        input_dim = self.dataset[0].trajectories.shape[1]
        sequence_length = self.dataset[0].trajectories.shape[0]
        
        # Create model instance based on architecture
        if model_config is None:
            model_config = {}
            
        if self.architecture == 'transformer':
            model = TransformerMotionModel(
                input_dim=input_dim,
                latent_dim=model_config.get("latent_dim", 256),
                n_heads=model_config.get("n_heads", 4),
                num_layers=model_config.get("num_layers", 8),
                dropout=model_config.get("dropout", 0.1),
                dim_feedforward=model_config.get("dim_feedforward", 512)
            ).to(self.device)
        elif self.architecture == 'temporal':
            # Create TemporalUnet model
            model = TemporalUnet(
                horizon=sequence_length,
                transition_dim=input_dim,
                cond_dim=0,  # No conditioning by default
                dim=model_config.get("channel_dim", 128),
                dim_mults=model_config.get("dim_mults", (1, 2, 4, 8)),
                attention=model_config.get("attention", False)
            ).to(self.device)
        elif self.architecture == 'local_attention':
            model = TransformerLocalAttention(
                input_dim=model_local_attention_config.get("input_dim", 69),
                max_seq_len=model_local_attention_config.get("max_seq_len", 69),
                dim=model_local_attention_config.get("dim", 512),
                depth=model_local_attention_config.get("depth", 6),
                local_attn_window_size=model_local_attention_config.get("local_attn_window_size", 4),
                causal=False
            ).to(self.device)
        else:
            raise ValueError(f"Unsupported architecture: {self.architecture}")
            
        # Load model weights from checkpoint
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
                
        return model
    
    def _create_diffusion(self, config: Dict[str, Any]) -> DiffusionV4:
        """
        Create the diffusion model.
        
        Args:
            config: Configuration for the diffusion model
            
        Returns:
            The configured diffusion model
        """
        return DiffusionV4(
            noise_steps=config.get("noise_steps", 1000),
            beta_start=config.get("beta_start", 1e-4),
            beta_end=config.get("beta_end", 0.02),
            joint_dim=self.joint_dim,
            frames=self.frames,
            device=self.device,
            predict_x0=config.get("predict_x0", True),
            schedule_type=config.get("schedule_type", "cosine"),
            cosine_s=config.get("cosine_s", 0.008),
        )
    
    def generate_samples(self, num_samples: int = 1, custom_frames: Optional[int] = None) -> torch.Tensor:
        """
        Generate motion samples using the trained model.
        
        Args:
            num_samples: Number of motion samples to generate
            custom_frames: Optional number of frames to generate
                           (overrides the default from the model)
                           
        Returns:
            Tensor of generated motion samples
        """
        self.model.eval()
        with torch.no_grad():
            samples = self.diffusion.sample(self.model, num_samples, custom_frames)
        return samples
    
    def save_motions(
        self, 
        samples: torch.Tensor, 
        output_dir: str, 
        filenames: Optional[List[str]] = None,
        joint_indices: Optional[List[int]] = None
    ) -> List[str]:
        """
        Save generated motion samples to disk.
        
        Args:
            samples: Tensor of motion samples to save
            output_dir: Directory to save the motions in
            filenames: Optional list of filenames for each sample
            joint_indices: Optional indices of joints to save (default: first 35 joints)
            
        Returns:
            List of paths where the motions were saved
        """
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Default joint indices if not provided
        if joint_indices is None:
            joint_indices = list(range(35))  # Default to first 35 joints
            
        # Generate default filenames if not provided
        if filenames is None:
            filenames = [f"motion_{i}.npy" for i in range(samples.shape[0])]
        elif len(filenames) < samples.shape[0]:
            # Extend filenames if not enough provided
            for i in range(len(filenames), samples.shape[0]):
                filenames.append(f"motion_{i}.npy")
                
        saved_paths = []
        for i, filename in enumerate(filenames[:samples.shape[0]]):
            filepath = os.path.join(output_dir, filename)
            
            # Extract the specified joints
            pos_data = samples[i, :, joint_indices].cpu().numpy()
            # print the shape of the motion data
            logging.info(f"Motion data shape: {pos_data.shape}")
            
            # Save the motion data
            np.save(filepath, pos_data)
            saved_paths.append(filepath)
            logging.info(f"Motion saved as {filepath}")
            
        return saved_paths

    def _get_from_checkpoint(self, checkpoint, key, default=None):
        """Helper to safely extract a value from a checkpoint dictionary"""
        if isinstance(checkpoint, dict) and key in checkpoint:
            return checkpoint[key]
        return default

def compare_models(
    model_paths: List[str],
    dataset_path: str,
    output_dir: str,
    num_samples: int = 1,
    custom_frames: Optional[int] = None,
    diffusion_config: Optional[Dict[str, Any]] = None,
    model_config: Optional[Dict[str, Any]] = None,
    device: Optional[str] = None,
    data_type: Optional[str] = None,
    architecture: Optional[str] = None
) -> Dict[str, List[str]]:
    """
    Compare multiple models by generating motions from each model.
    
    Args:
        model_paths: List of paths to model files to compare
        dataset_path: Path to the dataset used for the models
        output_dir: Directory to save generated motions
        num_samples: Number of samples to generate from each model
        custom_frames: Optional custom number of frames for generation
        diffusion_config: Configuration for the diffusion model (optional)
        model_config: Configuration for the model (optional)
        device: Device to run inference on (optional)
        data_type: Type of data to generate (optional)
        architecture: Type of model architecture (optional)
        
    Returns:
        Dictionary mapping model names to lists of motion file paths
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Motion output directory
    motion_dir = os.path.join(output_dir, "motions")
    if not os.path.exists(motion_dir):
        os.makedirs(motion_dir)
    
    # Default diffusion config if not provided
    if diffusion_config is None:
        diffusion_config = {
            "noise_steps": 50,
            "beta_start": 1e-4,
            "beta_end": 0.02,
            "predict_x0": True
        }
    
    # Dictionary to store the results
    results = {}
    
    # Process each model
    logging.info(f"Comparing {len(model_paths)} models...")
    
    for model_idx, model_path in enumerate(model_paths):
        model_name = os.path.basename(model_path)
        model_name = os.path.splitext(model_name)[0]
        logging.info(f"Processing model {model_idx+1}/{len(model_paths)}: {model_name}")
        
        # Initialize the diffusion inference model
        inference = DiffusionInference(
            model_path=model_path,
            dataset_path=dataset_path,
            diffusion_config=diffusion_config,
            model_config=model_config,
            device=device,
            data_type=data_type,
            architecture=architecture
        )
        
        # Generate samples
        samples = inference.generate_samples(num_samples=num_samples, custom_frames=custom_frames)
        
        # Create a subdirectory for this model
        model_motion_dir = os.path.join(motion_dir, model_name)
        if not os.path.exists(model_motion_dir):
            os.makedirs(model_motion_dir)
            
        # Save the motions
        motion_paths = inference.save_motions(
            samples=samples,
            output_dir=model_motion_dir
        )
        
        # Store results
        results[model_name] = {
            "model_path": model_path,
            "motion_paths": motion_paths,
            "architecture": inference.architecture
        }
    
    logging.info(f"Completed model comparison. Results saved to {output_dir}")
    return results


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Configuration
    dataset_path = "data/motions/humanoid3d_walk.txt"
    model_dir = "models/transformer_motion_model_no_motion_losses"
    output_dir = "test_output/sampled_motions"
    specific_model = "experiments/local_attention_predict_x0_20250312_115057/best_model_20250312_115058_local_attention_x0_step1800_loss0.002409.pth"
    # Find the latest model
    model_path = specific_model
    if model_path is None:
        logging.error(f"No model found in {model_dir}. Please train a model first.")
        exit(1)

    # Extract model information for the output filename
    model_name = os.path.basename(model_path)
    # Remove file extension
    model_name = os.path.splitext(model_name)[0]
    
    # Model configuration (only needed if loading state_dict)
    model_config = {
        "latent_dim": 1024,
        "n_heads": 8,
        "num_layers": 4,
        "dropout": 0.25,
        "dim_feedforward": 32,
        # TemporalUnet specific parameters (will be used if architecture is 'temporal')
        "dim_mults": (1, 2, 4, 8),
        "attention": False,
        "channel_dim": 128
    }

    model_local_attention_config = {
        "dim": 512,
        "depth": 6,
        "causal": False,
        "local_attn_window_size": 4,
        "max_seq_len": 69,
        "input_dim": 69
    }
    
    # Diffusion configuration
    diffusion_config = {
        "noise_steps": 1000,
        "beta_start": 1e-4,
        "beta_end": 0.02,
        "predict_x0": True,
        "schedule_type": "linear",
        "cosine_s": 0.008
    }
    
    # Create experiment subfolder with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(output_dir, f"experiment_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    custom_frames = None # Set to None to use default length from dataset
    data_type = 'both'
    architecture = 'local_attention'  # Set to None to use architecture from checkpoint, or specify 'transformer'
    
    # Save experiment metadata
    metadata = {
        "timestamp": timestamp,
        "model_path": model_path,
        "model_config": model_config,
        "diffusion_config": diffusion_config,
        "dataset_path": dataset_path,
        "custom_frames": custom_frames,
        "data_type": data_type,
        "architecture": architecture
    }
    with open(os.path.join(experiment_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
    
    # Create inference engine
    inference_engine = DiffusionInference(
        model_path=model_path,
        dataset_path=dataset_path,
        diffusion_config=diffusion_config,
        model_config=model_config,
        data_type=data_type,
        architecture=architecture
    )
    
    # Generate samples
    num_samples = 1
    
    samples = inference_engine.generate_samples(num_samples, custom_frames)
    logging.info(f"Generated {num_samples} samples with shape {samples.shape}")
    
    # Save samples
    saved_paths = inference_engine.save_motions(
        samples=samples,
        output_dir=experiment_dir,
        filenames=[f"motion_{model_name}_{inference_engine.architecture}_frames_{custom_frames or inference_engine.frames}_{i}.npy" 
                  for i in range(num_samples)]
    )
    
    logging.info(f"Saved {len(saved_paths)} motion samples to {experiment_dir}")

    # ADD COMPARISON OF MULTIPLE MODELS
    # Define an array of model paths to compare
    model_paths_to_compare = [
    ]
    
    # Only proceed with comparison if we have multiple models
    if len(model_paths_to_compare) > 1:
        logging.info(f"Comparing {len(model_paths_to_compare)} models...")
        
        # Create a subdirectory for comparison results
        comparison_dir = os.path.join(experiment_dir, "model_comparison")
        if not os.path.exists(comparison_dir):
            os.makedirs(comparison_dir)
        
        # Run the comparison
        comparison_results = compare_models(
            model_paths=model_paths_to_compare,
            dataset_path=dataset_path,
            output_dir=comparison_dir,
            num_samples=1,  # Generate 1 sample per model
            custom_frames=custom_frames,
            diffusion_config=diffusion_config,
            model_config=model_config,
            data_type=data_type,
            architecture=architecture
        )
        
        logging.info(f"Model comparison completed. Results saved to {comparison_dir}")
        
        # Print paths to motion files
        for model_name, result in comparison_results.items():
            for i, motion_path in enumerate(result["motion_paths"]):
                logging.info(f"Model {model_name} ({result['architecture']}), Sample {i+1}: {motion_path}")
