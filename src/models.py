"""
Model definitions for learning voice embeddings.

This module contains a simple baseline model. You should modify and extend
this to create better embeddings that capture temporal patterns and
patient-specific voice characteristics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union


class EmbeddingModel(nn.Module):
    """
    Simple baseline model for learning voice embeddings.
    This is a basic feedforward network that you should improve upon.

    Parameters
    ----------
    input_dim : int
        Dimension of input acoustic features.
    embedding_dim : int, default=64
        Dimension of the learned embedding space.
    hidden_dims : list of int, default=[256, 128]
        Dimensions of hidden layers.
    dropout : float, default=0.3
        Dropout probability for regularization.
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 64,
        hidden_dims: list = [256, 128],
        dropout: float = 0.3,
    ):
        super(EmbeddingModel, self).__init__()

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        # Build the network layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        # Final embedding layer (no activation - embeddings can be negative)
        layers.append(nn.Linear(prev_dim, embedding_dim))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute embeddings.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim).

        Returns
        -------
        embeddings : torch.Tensor
            Embedding tensor of shape (batch_size, embedding_dim).
        """
        return self.encoder(x)

    def get_num_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LinearClassifierHead(nn.Module):
    """
    Simple linear classifier head.

    This is intentionally kept simple to force the embedding model to learn
    rich, discriminative representations. A complex classifier would defeat
    the purpose of learning good embeddings.

    Parameters
    ----------
    embedding_dim : int
        Dimension of input embeddings.
    num_classes : int, default=2
        Number of output classes.
    """

    def __init__(self, embedding_dim: int, num_classes: int = 2):
        super(LinearClassifierHead, self).__init__()
        self.linear = nn.Linear(embedding_dim, num_classes)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for classification.

        Parameters
        ----------
        embeddings : torch.Tensor
            Input embeddings of shape (batch_size, embedding_dim).

        Returns
        -------
        logits : torch.Tensor
            Class logits of shape (batch_size, num_classes).
        """
        return self.linear(embeddings)




# Input : Les 3 derniers audios complets. On les projette, on les flattens, on les passe dans la couche dattention
# contrastive loss force la structure de l'espace latent, 
# la supervised CL permet de séparer les données en fonction de leur classe et éventuellement d'une cohérence de labels.

class MaskedSelfAttentionHead(nn.Module):

    def __init__(self, embedding_dim, head_size, block_size, time_decay: float = 0.01):
        super().__init__()
        self.key = nn.Linear(embedding_dim, head_size, bias=False)
        self.query = nn.Linear(embedding_dim, head_size, bias=False)
        self.value = nn.Linear(embedding_dim, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.time_decay = time_decay  # scales how strongly distant timestamps are downweighted

    def forward(self, x, time_ctx: Union[torch.Tensor, None] = None):
        B, T, C = x.shape
        # x is B, T, C
        k = self.key(x)   # (B, T, H)
        H = k.shape[-1]
        q = self.query(x) # (B, T, H)

        # Raw attention scores
        weights = q @ k.transpose(-2, -1) * H**-0.5  # (B, T, H) @ (B, H, T) -> (B, T, T)

        # Causal mask
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)

        # Temporal decay bias: downweight keys farther in time from the most recent segment
        if time_ctx is not None:
            # time_ctx shape: (B, T), use gap to last timestamp in the context
            gap_to_last = (time_ctx[:, -1].unsqueeze(-1) - time_ctx).clamp(min=0.0)
            # Broadcast to (B, T, T): same decay applied for every query on each key
            time_bias = -self.time_decay * gap_to_last.unsqueeze(1)
            weights = weights + time_bias

        weights = F.softmax(weights, dim=-1)
        v = self.value(x)  # (B, T, H)
        out = weights @ v  # (B, T, H)
        return out

class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, embedding_dim, head_size, block_size, time_decay: float = 0.01):
        super().__init__()
        self.heads = nn.ModuleList([
            MaskedSelfAttentionHead(embedding_dim, head_size, block_size, time_decay=time_decay)
            for _ in range(num_heads)
        ])

    def forward(self, x, time_ctx: torch.Tensor | None = None):
        out = torch.cat([h(x, time_ctx) for h in self.heads], dim=-1)
        return out

# class FeedForward(nn.Module):

#     def __init__(self, embedding_dim):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(embedding_dim, embedding_dim * 4),
#             nn.ReLU(),
#             nn.Linear(embedding_dim * 4, embedding_dim),
#         )

#     def forward(self, x):
#         return self.net(x)
    
class Block(nn.Module):

    def __init__(self, num_heads, embedding_dim, block_size, time_decay: float = 0.01):
        super().__init__()
        head_size = embedding_dim // num_heads
        self.sa_heads = MultiHeadAttention(num_heads, embedding_dim, head_size, block_size, time_decay=time_decay)
        self.ffwd = nn.Sequential(nn.Linear(head_size*num_heads, embedding_dim),
                                  nn.ReLU())
        self.LayerNorm = nn.LayerNorm(head_size*num_heads)

    def forward(self, x, time_ctx: torch.Tensor | None = None):
        x = self.sa_heads(x, time_ctx)
        x = self.LayerNorm(x)
        x = self.ffwd(x)
        return x
    
class Transformer_Decoder(nn.Module):

    """ 
    Pre-Classification model using a Transformer Decoder.

    Parameters 
    -------------
    n_features : number of features
    n_layer : number of blocks (MultiHeadMaskedAttention, LayerNorm, FFN) in the transformer
    embedding_dim : dimension of the embedding of the input
    num_heads : number of heads of attention in each block
    block_size : context size
    dropout : level of dropout
    device 

    Returns
    -------------
    Processed embeding of the input ready for binary classification
    """

    def __init__(self, n_features, n_layer, embedding_dim, num_heads, block_size, dropout, device, time_decay: float = 0.01):
        super().__init__()
        self.embedding = nn.Linear(n_features, embedding_dim)
        self.position_embedding_table = nn.Embedding(block_size, embedding_dim)
        self.blocks = nn.ModuleList([Block(num_heads, embedding_dim, block_size, time_decay=time_decay) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(embedding_dim) # LayerNorm ultime
        self.device = device
        self.block_size = block_size
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, time_ctx: torch.Tensor | None = None):
        B, T = x.shape[0], x.shape[1]
        if time_ctx is not None:
            time_ctx = time_ctx.to(x.device)
        emb = self.embedding(x)  # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device))  # (T, C)
        x = emb + pos_emb
        for block in self.blocks:
            x = block(x, time_ctx)
        x = self.ln_f(x)
        pred = x[:,-1,:].squeeze()
        return pred
    
    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)