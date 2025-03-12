import torch
import numpy as np
from scipy import linalg
from torch.utils.data import DataLoader
from tqdm import tqdm

class MotionFID:
    def __init__(self, real_dataset, model, device='cuda', batch_size=128, num_samples=512):
        """
        Initialize FID calculator for motion data
        Args:
            real_dataset: Dataset containing real motion data
            model: Trained motion model for generating samples
            device: Device to run calculations on
            batch_size: Batch size for processing
            num_samples: Number of samples to generate for FID calculation
        """
        self.real_dataset = real_dataset
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.num_samples = num_samples
        
    def calculate_activation_statistics(self, data):
        """
        Calculate mean and covariance statistics of the data
        """
        # print(f"Processing batch of shape {data.shape}...")
        # Flatten the temporal dimension
        features = data.reshape(data.shape[0], -1)
        # print(f"Flattened to shape {features.shape}")
        
        # Calculate statistics on GPU
        # print("Computing mean...")
        mu = torch.mean(features, dim=0)
        # Center the features
        # print("Computing covariance...")
        features = features - mu.unsqueeze(0)
        # Calculate covariance
        sigma = (features.T @ features) / (features.shape[0] - 1)
        # print(f"Statistics computed: mean shape {mu.shape}, covariance shape {sigma.shape}")
        
        return mu.cpu().numpy(), sigma.cpu().numpy()
    
    def calculate_fid(self, mu1, sigma1, mu2, sigma2):
        """
        Calculate FID score between two distributions
        """
        # print("Converting arrays to tensors...")
        # Convert numpy arrays to torch tensors on GPU
        mu1 = torch.from_numpy(mu1).to(self.device)
        mu2 = torch.from_numpy(mu2).to(self.device)
        sigma1 = torch.from_numpy(sigma1).to(self.device)
        sigma2 = torch.from_numpy(sigma2).to(self.device)
        
        eps = 1e-6
        
        # print("Computing mean difference...")
        # Calculate squared difference between means
        diff = mu1 - mu2
        diff_square = torch.dot(diff, diff)
        
        # print("Computing matrix square root (this might take a while)...")
        # Calculate matrix sqrt using SVD
        # ensure sigma1 and sigma2 are same dtype
        sigma1 = sigma1.to(torch.float32)
        sigma2 = sigma2.to(torch.float32)
        product = sigma1 @ sigma2
        U, s, Vh = torch.linalg.svd(product)
        sqrt_s = torch.sqrt(torch.clamp(s, min=eps))
        covmean = U @ torch.diag(sqrt_s) @ Vh
        
        # print("Computing final FID score...")
        tr_covmean = torch.trace(covmean)
        
        fid = (diff_square + 
               torch.trace(sigma1) + 
               torch.trace(sigma2) - 
               2 * tr_covmean)
        
        return fid.cpu().item()
    
    def get_real_activations(self):
        """
        Get activation statistics for real data
        """
        all_features = []
        dataloader = DataLoader(self.real_dataset, batch_size=self.batch_size, shuffle=True)
        
        # print("Processing real data...")
        for batch in tqdm(dataloader):
            trajectories = batch.trajectories.to(self.device)
            all_features.append(trajectories)
            
            if len(all_features) * self.batch_size >= self.num_samples:
                break
                
        all_features = torch.cat(all_features, dim=0)[:self.num_samples]
        return self.calculate_activation_statistics(all_features)
    
    def get_generated_activations(self):
        """
        Get activation statistics for generated data
        """
        all_features = []
        num_batches = (self.num_samples + self.batch_size - 1) // self.batch_size
        
        # print("Generating samples...")
        for _ in tqdm(range(num_batches)):
            # Get conditioning from a random real sample
            idx = np.random.randint(len(self.real_dataset))
            batch = self.real_dataset[idx]
            cond = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                   for k, v in batch.conditions.items()}
            
            # Generate samples
            with torch.no_grad():
                samples = self.model.sample(
                    batch_size=self.batch_size,
                    cond=cond,
                    device=self.device
                )
            
            all_features.append(samples)
            
        all_features = torch.cat(all_features, dim=0)[:self.num_samples]
        return self.calculate_activation_statistics(all_features)
    
    def compute_fid(self):
        """
        Compute FID score between real and generated distributions
        """
        # print("\nStep 1: Computing statistics for real data...")
        # Get statistics for real data
        mu_real, sigma_real = self.get_real_activations()
        
        # print("\nStep 2: Computing statistics for generated data...")
        # Get statistics for generated data
        mu_gen, sigma_gen = self.get_generated_activations()
        
        # print("\nStep 3: Computing FID score...")
        # Calculate FID
        fid_score = self.calculate_fid(mu_real, sigma_real, mu_gen, sigma_gen)
        
        return fid_score 