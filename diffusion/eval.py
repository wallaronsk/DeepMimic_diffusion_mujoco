import numpy as np
from scipy import linalg
from diffuser.models.diffusion_v4 import DiffusionV4
from diffuser.models.transformer_temporal import TransformerMotionModel
from diffuser.models.transformer import LocalTransformer as TransformerLocalAttention
from diffuser.models.temporal_v2 import TemporalUnet
from metrics.fid_score import MotionFID
import torch
import time
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
motion_path = "data/motions/humanoid3d_walk_with_vels.npy"

def slice_motion_sample(sample, window_size, step_size=10):
    # sample [nframes, nfeats]
    windows = []
    max_offset = sample.shape[0] - window_size + 1
    for offset_i in np.arange(max_offset)[0::step_size]:
        windows.append(sample[offset_i:offset_i+window_size].unsqueeze(0))
    return torch.cat(windows, dim=0)


class CustomEvalWrapper():
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.pose_embeddings = None
    
    @torch.no_grad()
    def get_motion_embeddings(self, samples, optional="unet"):
        if optional == "unet":
            return samples
        else:
            return self.pose_embeddings(samples)


def get_sample(path, device):
    try:
        return torch.tensor(np.load(path), device=device)
    except Exception as e:
        print("get_sample error: {}".format(e))

def generate_eval_samples(model, diffusion, num_samples):
    # (nsamples, nframes,nfeats)
    start_time = time.time()
    samples = diffusion.sample(model, num_samples)
    end_time = time.time()
    sampling_time = end_time - start_time
    sampling_rate = num_samples / sampling_time
    print(f"Sampling time: {sampling_time} seconds")
    print(f"Sampling rate: {sampling_rate} samples/second")
    return samples, sampling_rate  # Return both samples and sampling rate

def calc_inter_diversity(eval_wrapper, samples):
    motion_emb = eval_wrapper.get_motion_embeddings(samples).cpu().numpy()
    dist = linalg.norm(motion_emb[:samples.shape[0]//2] - motion_emb[samples.shape[0]//2:], axis=1)  # FIXME - will not work for odd bs
    return dist.mean()

def calc_sifid(eval_wrapper, gen_samples, gt_sample, window_size=10):
    gt_slices = slice_motion_sample(gt_sample, window_size)
    motion_fid = MotionFID(gt_slices, eval_wrapper.model, device=device)

    def get_stats(_samples):
        _mu, _cov = motion_fid.calculate_activation_statistics(_samples)
        return _mu, _cov

    sifid = []

    gt_mu, gt_cov = get_stats(gt_slices)

    for sampe_i in range(gen_samples.shape[0]):
        gen_slices = slice_motion_sample(gen_samples[sampe_i], window_size)
        gen_mu, gen_cov = get_stats(gen_slices)
        sifid.append(motion_fid.calculate_fid(gt_mu, gt_cov, gen_mu, gen_cov))

    return np.array(sifid).mean()


def calc_intra_diversity(eval_wrapper, samples, window_size=10):
    max_offset = samples.shape[1] - window_size
    dist = []
    for sample_i in range(samples.shape[0]):
        offsets = np.random.randint(max_offset, size=2)
        window0 = samples[[sample_i], offsets[0]:offsets[0]+window_size, :]
        window1 = samples[[sample_i], offsets[1]:offsets[1]+window_size, :]
        motion_emb = eval_wrapper.get_motion_embeddings(torch.cat([window0, window1]).float()).cpu().numpy()
        dist.append(linalg.norm(motion_emb[0] - motion_emb[1]))
    return np.array(dist).mean()


def evaluate(model, diffusion, eval_wrapper, num_samples_limit, replication_times):

    results = {}
    sampling_rates = []  # Add this to track sampling rates

    for window_size in [10]:
        print(f'===Starting [window_size={window_size}]===')
        intra_diversity = []
        gt_intra_diversity = []
        intra_diversity_gt_diff = []
        inter_diversity = []
        sifid = []
        for rep_i in range(replication_times):
            gt_samples = get_sample(motion_path, device)
            gen_samples, sampling_rate = generate_eval_samples(model, diffusion, num_samples_limit)
            sampling_rates.append(sampling_rate)  # Store sampling rate
            print(f'===REP[{rep_i}]===')
            _intra_diversity = calc_intra_diversity(eval_wrapper, gen_samples, window_size=window_size)
            intra_diversity.append(_intra_diversity)
            print('intra_diversity [{:.3f}]'.format(_intra_diversity))
            _gt_intra_diversity = calc_intra_diversity(eval_wrapper, torch.tile(gt_samples[None], (gen_samples.shape[0], 1, 1)), window_size=window_size)
            gt_intra_diversity.append(_gt_intra_diversity)
            print('gt_intra_diversity [{:.3f}]'.format(_gt_intra_diversity))
            _intra_diversity_gt_diff = abs(_intra_diversity - _gt_intra_diversity)
            intra_diversity_gt_diff.append(_intra_diversity_gt_diff)
            print('intra_diversity_gt_diff [{:.3f}]'.format(_intra_diversity_gt_diff))
            _inter_diversity = calc_inter_diversity(eval_wrapper, gen_samples)
            inter_diversity.append(_inter_diversity)
            print('inter_diversity [{:.3f}]'.format(_inter_diversity))
            _sifid = calc_sifid(eval_wrapper, gen_samples, gt_samples, window_size=window_size)
            sifid.append(_sifid)
            print('SiFID [{:.3f}]'.format(_sifid))

        results[window_size] = {
            'intra_diversity': {'mean': np.mean(intra_diversity), 'std': np.std(intra_diversity)},
            'gt_intra_diversity': {'mean': np.mean(gt_intra_diversity), 'std': np.std(gt_intra_diversity)},
            'intra_diversity_gt_diff': {'mean': np.mean(intra_diversity_gt_diff), 'std': np.std(intra_diversity_gt_diff)},
            'inter_diversity': {'mean': np.mean(inter_diversity), 'std': np.std(inter_diversity)},
            'sifid': {'mean': np.mean(sifid), 'std': np.std(sifid)},
            'sampling_rate': {'mean': np.mean(sampling_rates), 'std': np.std(sampling_rates)}  # Add sampling rate stats
        }

        print(f'===Summary [window_size={window_size}]===')
        print('intra_diversity [{:.3f}±{:.3f}]'.format(results[window_size]['intra_diversity']['mean'], results[window_size]['intra_diversity']['std']))
        print('gt_intra_diversity [{:.3f}±{:.3f}]'.format(results[window_size]['gt_intra_diversity']['mean'], results[window_size]['gt_intra_diversity']['std']))
        print('intra_diversity_gt_diff [{:.3f}±{:.3f}]'.format(results[window_size]['intra_diversity_gt_diff']['mean'], results[window_size]['intra_diversity_gt_diff']['std']))
        print('inter_diversity [{:.3f}±{:.3f}]'.format(results[window_size]['inter_diversity']['mean'], results[window_size]['inter_diversity']['std']))
        print('SiFID [{:.3f}±{:.3f}]'.format(results[window_size]['sifid']['mean'], results[window_size]['sifid']['std']))
        print('Sampling Rate [{:.3f}±{:.3f}] samples/second'.format(results[window_size]['sampling_rate']['mean'], results[window_size]['sampling_rate']['std']))

    return results

model_arch = "local_attention"
input_dim = 69
model_config = {
    "input_dim": input_dim,
    "latent_dim": 512,
    "n_heads": 4,
    "num_layers": 8,
    "dropout": 0.1,
    # UNET
    "channel_dim": 128,
    "channel_mult": [1, 2, 4, 8],
    "attention": False,
}

model_local_attention_config = {
    "dim": 512,
    "depth": 6,
    "local_attn_window_size": 4,
    "max_seq_len": 69,
    "input_dim": 69
}

diffusion_config = {
    "noise_steps": 1000,
    "beta_start": 1e-4,
    "beta_end": 0.02,
    "joint_dim": input_dim,
    "predict_x0": True,
    "schedule_type": "cosine",
    "cosine_s": 0.008
}

def main():
    print("Using model architecture: {}".format(model_arch))
    # fixseed(args.seed)
    num_samples_limit = 50  # 100
    replication_times = 5  # 5
    if model_arch == "transformer":
        model = TransformerMotionModel(
                input_dim=model_config.get("input_dim", 35),
                latent_dim=model_config.get("latent_dim", 128),
                n_heads=model_config.get("n_heads", 16),
                num_layers=model_config.get("num_layers", 4),
                dropout=model_config.get("dropout", 0.7)
            ).to(device)
    elif model_arch == "temporal":
        model = TemporalUnet(
            horizon=32, # num frames
            transition_dim=model_config.get("input_dim", 35), # num features
            cond_dim=0,
            dim=model_config.get("channel_dim", 128),
            dim_mults=model_config.get("channel_mult", [1, 2, 4, 8]),
            attention=model_config.get("attention", False)
        ).to(device)
    elif model_arch == "local_attention":
        model = TransformerLocalAttention(
            input_dim=model_local_attention_config.get("input_dim", 69),
            max_seq_len=model_local_attention_config.get("max_seq_len", 69),
            dim=model_local_attention_config.get("dim", 512),
            depth=model_local_attention_config.get("depth", 6),
            causal=False,
            local_attn_window_size=model_local_attention_config.get("local_attn_window_size", 4),
        ).to(device)
    
    pretrained_model_path = "experiments/local_attention_predict_x0_20250312_115057/best_model_20250312_115058_local_attention_x0_step1800_loss0.002409.pth"
    
    if pretrained_model_path:   
        print("Loading pretrained model")
        checkpoint = torch.load(pretrained_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    # print number of model parameters
    print(f"Number of model parameters: {sum(p.numel() for p in model.parameters())}")
    diffusion = DiffusionV4(
            noise_steps=diffusion_config.get("noise_steps", 1000),
            beta_start=diffusion_config.get("beta_start", 1e-4),
            beta_end=diffusion_config.get("beta_end", 0.02),
            joint_dim=diffusion_config.get("joint_dim", input_dim),
            device=device,
            frames=32,
            predict_x0=diffusion_config.get("predict_x0", True),
            schedule_type=diffusion_config.get("schedule_type", "linear"),
            cosine_s=diffusion_config.get("cosine_s", 0.008),
        )
    
    model.eval()  # disable random masking

    eval_wrapper = CustomEvalWrapper(model)
    eval_dict = evaluate(model, diffusion, eval_wrapper, num_samples_limit, replication_times)
    results_folder = r"model_eval_results/local_attention/trained"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    log_file = os.path.join(results_folder, "eval_results_local_attention_trained_1000_steps.log")
    with open(log_file, 'w') as fw:
        fw.write(str(eval_dict))
    print("Saved eval results to {}".format(log_file))
    np.save(log_file.replace('.log', '.npy'), eval_dict)


if __name__ == '__main__':
    main()