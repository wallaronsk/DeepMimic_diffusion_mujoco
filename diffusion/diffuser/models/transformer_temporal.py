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
    def __init__(self, input_dim, latent_dim=256, n_heads=4, num_layers=8, dropout=0.1, dim_feedforward=1024, max_seq_len=128, num_classes=10, random_mask_prob=0.1, masking_type="causal", max_sequence_length=128):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.time_embed_dim = latent_dim
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.dim_feedforward = dim_feedforward
        self.random_mask_prob = random_mask_prob
        self.masking_type = masking_type
        self.max_sequence_length = max_sequence_length
        self.pose_embed = nn.Linear(self.input_dim, self.latent_dim)

        self.position_embed = nn.Embedding(max_sequence_length, self.latent_dim)

        # self.class_embed = nn.Embedding(num_classes, self.latent_dim)

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

    def forward(self, x, time, y=None):
        '''
        x: the input motion sequence (batch_size, seq_len, input_dim)
        time: the timestep (batch_size,)
        y: the target motion class (batch_size, seq_len, input_dim)
        '''
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        # # If seq_lengths is not provided, assume all sequences are full length
        # if seq_lengths is None:
        #     seq_lengths = torch.full((batch_size,), orig_seq_len, device=x.device)
            
        # Create padding mask (1 for real positions, 0 for padding)
        # padding_mask = torch.zeros((batch_size, self.max_sequence_length), device=x.device)
        # for i in range(batch_size):
        #     padding_mask[i, :seq_lengths[i]] = 1
            
        # # if x.shape[1] != self.max_sequence_length, we need to pad the sequence
        # if x.shape[1] < self.max_sequence_length:
        #     # pad the sequence with zeros
        #     x = torch.nn.functional.pad(x, (0, 0, 0, self.max_sequence_length - x.shape[1]))

        pose_emb = self.pose_embed(x)
        time_emb = self.time_embed(timestep_embedding(time, self.latent_dim))
        time_emb = time_emb.unsqueeze(1)
        x = pose_emb + time_emb

        position_indices = torch.arange(seq_len, device=x.device)
        position_emb = self.position_embed(position_indices)
        
        # Apply padding mask to position embeddings
        # position_emb = position_emb # * padding_mask.unsqueeze(-1)

        # # Create attention mask for transformer based on padding
        # attention_mask = None
        # if self.masking_type == "random":
        #     # Create random mask (1 = keep, 0 = mask)
        #     rand_mask = torch.bernoulli(torch.ones_like(padding_mask) * (1 - self.random_mask_prob))
        #     # Combine with padding mask (we only want to randomly mask real positions)
        #     attention_mask = rand_mask * padding_mask
        # elif self.masking_type == "causal":
        #     # Create causal mask (1 = attend, 0 = don't attend)
        #     causal_mask = torch.tril(torch.ones((x.shape[1], x.shape[1]), device=x.device))
        #     # Apply padding mask to causal mask
        #     attention_mask = causal_mask.unsqueeze(0) * padding_mask.unsqueeze(1)
        # else:
        #     # Just use padding mask
        #     attention_mask = padding_mask.unsqueeze(1)
            
        # Add position embeddings
        x = x + position_emb

        # if y is not None:
        #     y_emb = self.class_embed(y)
        #     y_emb = y_emb.unsqueeze(1)
        #     x = x + y_emb
        
        # Convert attention mask for transformer layers
        # For PyTorch transformer, the attention mask is usually (batch_size, seq_len, seq_len)
        # where 1 means "attend" and 0 means "don't attend"
        # We need to convert our mask format accordingly
        for layer in self.transformer:
            # Apply layer with attention mask
            x = layer(x)

        x = self.final_layer(x)
        
        # # Apply padding mask to output to ensure padded positions remain zeroed
        # x = x * padding_mask.unsqueeze(-1)
        
        return x
