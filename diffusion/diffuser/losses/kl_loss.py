import torch
import torch.nn.functional as F

class KLDivergenceLoss:
    """
    Computes KL divergence loss for diffusion models.
    This implementation is optimized for temporal sequences.
    """
    def __init__(self, eps=1e-6):
        self.eps = eps

    def compute_q_posterior_mean_variance(self, x_start, x_t, t, alphas_cumprod):
        """
        Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0)
        """
        posterior_mean_coef1 = (
            torch.sqrt(alphas_cumprod[t-1]) * (1. - alphas_cumprod[t])
            / (1. - alphas_cumprod[t-1])
        )
        posterior_mean_coef2 = (
            torch.sqrt(1. - alphas_cumprod[t]) * alphas_cumprod[t-1]
            / (1. - alphas_cumprod[t-1])
        )
        
        posterior_mean = posterior_mean_coef1[..., None, None] * x_start + \
                        posterior_mean_coef2[..., None, None] * x_t
                        
        posterior_variance = (1. - alphas_cumprod[t-1]) / (1. - alphas_cumprod[t]) * \
                           (1. - alphas_cumprod[t] / alphas_cumprod[t-1])
                           
        return posterior_mean, posterior_variance

    def __call__(self, model_output, x_start, x_t, t, alphas_cumprod):
        """
        Compute KL divergence between true posterior and predicted distribution
        
        Args:
            model_output: Predicted noise from the model
            x_start: Original clean data
            x_t: Noisy data at timestep t
            t: Current timestep
            alphas_cumprod: Cumulative product of alpha values
            
        Returns:
            KL divergence loss and additional metrics
        """
        # Get true posterior parameters
        true_mean, true_variance = self.compute_q_posterior_mean_variance(
            x_start, x_t, t, alphas_cumprod
        )
        
        # Get predicted distribution parameters
        # Convert model's noise prediction to mean prediction
        pred_mean = (
            (torch.sqrt(alphas_cumprod[t])[..., None, None] * x_t - 
             (1 - alphas_cumprod[t])[..., None, None] * model_output)
            / torch.sqrt(1. - alphas_cumprod[t])[..., None, None]
        )
        
        # Compute KL divergence
        kl_div = 0.5 * (
            torch.log(true_variance[..., None, None] + self.eps) - 
            torch.log(alphas_cumprod[t][..., None, None] + self.eps) + 
            (alphas_cumprod[t][..., None, None] + 
             (true_mean - pred_mean).pow(2)) / 
            (true_variance[..., None, None] + self.eps) - 1.0
        )
        
        # Average over all dimensions except batch
        kl_div = kl_div.mean(dim=(1, 2))
        
        # Compute additional metrics
        mean_diff = (true_mean - pred_mean).abs().mean()
        var_diff = (true_variance - alphas_cumprod[t]).abs().mean()
        
        metrics = {
            'kl_loss': kl_div.mean().item(),
            'mean_diff': mean_diff.item(),
            'var_diff': var_diff.item()
        }
        
        return kl_div.mean(), metrics 