import torch
from data_loaders.motion_dataset_v2 import MotionDataset
from torch.utils.data import Subset
import numpy as np
from tqdm import tqdm
import time
class RealDataFID:
    def __init__(self, dataset, device='cuda', batch_size=32, num_samples=1000):
        """
        Initialize FID calculator for comparing real data with itself
        """
        self.dataset = dataset
        self.device = device
        self.batch_size = batch_size
        self.num_samples = num_samples
        
    def calculate_activation_statistics(self, data):
        """
        Calculate mean and covariance statistics of the data
        """
        print(f"Processing batch of shape {data.shape}...")
        features = data.reshape(data.shape[0], -1)
        print(f"Flattened to shape {features.shape}")
        
        print("Computing mean...")
        mu = torch.mean(features, dim=0)
        print("Computing covariance...")
        features = features - mu.unsqueeze(0)
        sigma = (features.T @ features) / (features.shape[0] - 1)
        print(f"Statistics computed: mean shape {mu.shape}, covariance shape {sigma.shape}")
        
        return mu.cpu().numpy(), sigma.cpu().numpy()
    
    def calculate_fid(self, mu1, sigma1, mu2, sigma2):
        """
        Calculate FID score between two distributions
        """
        print("Converting arrays to tensors...")
        mu1 = torch.from_numpy(mu1).to(self.device)
        mu2 = torch.from_numpy(mu2).to(self.device)
        sigma1 = torch.from_numpy(sigma1).to(self.device)
        sigma2 = torch.from_numpy(sigma2).to(self.device)
        
        eps = 1e-6
        
        print("Computing mean difference...")
        diff = mu1 - mu2
        diff_square = torch.dot(diff, diff)
        start = time.time()
        print("Computing matrix square root...")
        product = sigma1 @ sigma2
        U, s, Vh = torch.linalg.svd(product)
        sqrt_s = torch.sqrt(torch.clamp(s, min=eps))
        covmean = U @ torch.diag(sqrt_s) @ Vh
        end = time.time()
        print("Time taken: {}".format(end - start))
        print("Computing final FID score...")
        tr_covmean = torch.trace(covmean)
        
        fid = (diff_square + 
               torch.trace(sigma1) + 
               torch.trace(sigma2) - 
               2 * tr_covmean)
        
        return fid.cpu().item()
    
    def get_subset_data(self, indices):
        """
        Get data from a subset of the dataset
        """
        all_features = []
        subset = Subset(self.dataset, indices)
        
        print(f"Processing {len(subset)} samples...")
        for idx in tqdm(range(len(subset))):
            batch = subset[idx]
            trajectories = batch.trajectories.to(self.device)
            all_features.append(trajectories.unsqueeze(0))
            
        all_features = torch.cat(all_features, dim=0)
        return self.calculate_activation_statistics(all_features)
    
    def compute_fid(self):
        """
        Compute FID score between two different subsets of real data
        """
        # Create two non-overlapping sets of indices
        all_indices = np.arange(len(self.dataset))
        np.random.shuffle(all_indices)
        split_idx = self.num_samples
        
        indices1 = all_indices[:split_idx]
        indices2 = all_indices[split_idx:split_idx*2]
        
        print("\nStep 1: Computing statistics for first subset...")
        mu1, sigma1 = self.get_subset_data(indices1)
        
        print("\nStep 2: Computing statistics for second subset...")
        mu2, sigma2 = self.get_subset_data(indices2)
        
        print("\nStep 3: Computing FID score...")
        fid_score = self.calculate_fid(mu1, sigma1, mu2, sigma2)
        
        return fid_score

def main():
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load dataset
    dataset = MotionDataset("data/motions/humanoid3d_cartwheel.txt", shuffle=True)
    
    # Initialize FID calculator
    fid_calculator = RealDataFID(
        dataset=dataset,
        device=device,
        batch_size=2,
        num_samples=10  # Small number for testing
    )
    
    # Calculate FID score
    fid_score = fid_calculator.compute_fid()
    print(f"\nFID Score between two subsets of real data: {fid_score}")
    print("Note: This score should be very close to 0 since we're comparing the same distribution with itself")

if __name__ == "__main__":
    main() 