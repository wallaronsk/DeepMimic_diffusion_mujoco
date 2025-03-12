import torch
import torch.nn as nn
import math
from abc import abstractmethod
from diffuser.models.qna import FusedQnA

def timestep_embedding(timesteps, embedding_dim, max_period=10000, device=None):
    """
    Source: https://github.com/SinMDM/SinMDM/blob/main/models/mdm_qnanet.py
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x di.m] Tensor of positional embeddings.
    """
    half = embedding_dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if embedding_dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class TimestepBlock(nn.Module):
    """
    Source: https://github.com/SinMDM/SinMDM/blob/main/models/mdm_qnanet.py
    Any module where forward() takes timestep embeddings as a second argument.
    """
    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """
    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class TransformerMotionModel(nn.Module):
    def __init__(self, input_dim, latent_dim=256, n_heads=4, num_layers=8, dropout=0.1, dim_feedforward=1024):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.time_embed_dim = latent_dim
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.dim_feedforward = dim_feedforward
        
        self.pose_embed = nn.Linear(self.input_dim, self.latent_dim)

        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )

        self.transformer = nn.ModuleList([
            nn.TransformerEncoderLayer(self.latent_dim, self.n_heads, self.dim_feedforward, self.dropout)
            for _ in range(self.num_layers)
        ])

        self.final_layer = nn.Linear(self.latent_dim, self.input_dim)

    def forward(self, x, time):
        pose_emb = self.pose_embed(x)
        time_emb = self.time_embed(timestep_embedding(time, self.latent_dim))
        time_emb = time_emb.unsqueeze(1)
        x = pose_emb + time_emb
        
        for layer in self.transformer:
            x = layer(x)

        x = self.final_layer(x)
        return x
