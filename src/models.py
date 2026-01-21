"""
Model definitions for learning voice embeddings.

This module contains a simple baseline model. You should modify and extend
this to create better embeddings that capture temporal patterns and patient-specific voice characteristics.
"""

import math
import torch
import torch.nn as nn


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
                    nn.BatchNorm1d(hidden_dim),     # Original norm method
                    # nn.LayerNorm(hidden_dim),     # Newly added for testing
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        # Final embedding layer (no activation - embeddings can be negative)
        layers.append(nn.Linear(prev_dim, embedding_dim))
        # layers.append(nn.LayerNorm(embedding_dim))  # Newly added for testing
        
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


class EmbeddingTransformer(nn.Module):
    """
    Transformer encoder that maps tabular/vector features [B, input_dim] -> embeddings [B, embedding_dim].

    接口保持与原始 EmbeddingModel 一致： __init__(input_dim, embedding_dim=64, hidden_dims=[256, 128], dropout=0.3)
    """
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 64,
        hidden_dims: list = [256, 128],
        dropout: float = 0.0,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        # 解释 hidden_dims：尽量复用你现有的配置习惯
        # hidden_dims[0] -> d_model（token embedding 维度）
        # hidden_dims[1] -> dim_feedforward（FFN 隐层维度），若不给则用 4*d_model
        d_model = int(hidden_dims[0]) if (hidden_dims and len(hidden_dims) >= 1) else 256
        dim_ff = int(hidden_dims[1]) if (hidden_dims and len(hidden_dims) >= 2) else 4 * d_model

        # Transformer 超参（不暴露接口，避免改动其他代码；你要改也只改这里）
        self.num_layers = 2  # 和默认 hidden_dims 长度一致的“轻量基线”
        self.nhead = self._pick_nhead(d_model)

        # 把 786 维向量切成 token：默认每 8 维为一个 token，序列长度约 99（比 786 token 省很多算力）
        self.patch_size = 8
        self.seq_len = math.ceil(input_dim / self.patch_size)
        self.pad_dim = self.seq_len * self.patch_size - input_dim  # 不整除则右侧补 0

        # token 投影：每个 token 是 patch_size 维 -> d_model
        self.token_proj = nn.Linear(self.patch_size, d_model)

        # CLS token + 位置编码（特征本身无“时间顺序”，但维度索引是固定的，因此可用可学习位置编码）
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.seq_len, d_model))

        # Transformer Encoder（LayerNorm 风格）
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=self.nhead,
            dim_feedforward=max(dim_ff, 2 * d_model),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=self.num_layers)

        # 输出到 embedding
        self.out_norm = nn.LayerNorm(d_model)
        self.to_embedding = nn.Linear(d_model, embedding_dim)

        # 初始化（轻量即可）
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    @staticmethod
    def _pick_nhead(d_model: int) -> int:
        # 选择一个能整除 d_model 的 nhead（尽量用 8/4/2/1）, 减少显存占用
        for h in (8, 4, 2, 1):
            if d_model % h == 0:
                return h
        return 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, input_dim]
        return: [B, embedding_dim]
        """
        # 确保形状正确
        if x.dim() != 2 or x.size(1) != self.input_dim:
            raise ValueError(f"Expected x shape [B, {self.input_dim}], got {tuple(x.shape)}")

        B = x.size(0)

        # 右侧 padding 到 patch_size 的整数倍
        if self.pad_dim > 0:
            x = torch.cat([x, x.new_zeros(B, self.pad_dim)], dim=1)  # [B, seq_len*patch_size]

        # 切成 token： [B, seq_len, patch_size]
        x = x.view(B, self.seq_len, self.patch_size)

        # token embedding： [B, seq_len, d_model]
        tok = self.token_proj(x)

        # prepend CLS： [B, 1+seq_len, d_model]
        cls = self.cls_token.expand(B, -1, -1)
        tok = torch.cat([cls, tok], dim=1)

        # 加位置编码
        tok = tok + self.pos_embed

        # Transformer 编码
        h = self.encoder(tok)  # [B, 1+seq_len, d_model]

        # CLS 池化
        cls_h = h[:, 0, :]     # [B, d_model]

        # 输出 embedding
        cls_h = self.out_norm(cls_h)
        emb = self.to_embedding(cls_h)  # [B, embedding_dim]
        return emb
    
    def get_num_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LinearClassifierHead(nn.Module):
    """
    Simple linear classifier head.

    This is intentionally kept simple to force the embedding model to learn rich, discriminative representations.
    A complex classifier would defeat the purpose of learning good embeddings.

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
