"""
Model definitions for learning voice embeddings.

This module contains a simple baseline model. You should modify and extend
this to create better embeddings that capture temporal patterns and patient-specific voice characteristics.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    Same inputs with "EmbeddingModel": __init__(input_dim, embedding_dim=64, hidden_dims=[256, 128], dropout=0.3)
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

        # 解释 hidden_dims: 尽量复用你现有的配置习惯
        # hidden_dims[0] -> d_model (token embedding 维度)
        # hidden_dims[1] -> dim_feedforward (FFN 隐层维度), 若不给则用 4*d_model
        d_model = int(hidden_dims[0]) if (hidden_dims and len(hidden_dims) >= 1) else 256
        dim_ff = int(hidden_dims[1]) if (hidden_dims and len(hidden_dims) >= 2) else 4 * d_model

        # Transformer 超参 (不暴露接口, 避免改动其他代码；你要改也只改这里)
        self.num_layers = 2  # 和默认 hidden_dims 长度一致的“轻量基线”
        self.nhead = self._pick_nhead(d_model)

        # 把 786 维向量切成 token: 默认每 8 维为一个 token, 序列长度约 99 (比 786 token 省很多算力)
        self.patch_size = 8
        self.seq_len = math.ceil(input_dim / self.patch_size)
        self.pad_dim = self.seq_len * self.patch_size - input_dim  # 不整除则右侧补 0

        # token 投影: 每个 token 是 patch_size 维 -> d_model
        self.token_proj = nn.Linear(self.patch_size, d_model)

        # CLS token + 位置编码 (特征本身无“时间顺序”, 但维度索引是固定的, 因此可用可学习位置编码)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.seq_len, d_model))

        # Transformer Encoder (LayerNorm 风格)
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

        # 轻量初始化
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    @staticmethod
    def _pick_nhead(d_model: int) -> int:
        # 选择一个能整除 d_model 的 nhead (尽量用 8/4/2/1), 减少显存占用
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

        # 切成 token:  [B, seq_len, patch_size]
        x = x.view(B, self.seq_len, self.patch_size)

        # token embedding:  [B, seq_len, d_model]
        tok = self.token_proj(x)

        # prepend CLS:  [B, 1+seq_len, d_model]
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


class ContextualEmbeddingTransformer(nn.Module):
    """
    Context-aware Transformer encoder:
    Inputs:
        x_seq               [B, T, input_dim]   (同一病人的上下文段序列, 最后一个 token 是“当前段”)
        delta_t             [B, T]              (当前段 - 各段 的时间差, 最后一个为 0)
        key_padding_mask    [B, T]              (可选: pad 位置为 True)
    Outputs:
        emb                 [B, embedding_dim]

    设计要点: 
    1) 先用 TransformerEncoder 在序列维度上做上下文建模 (token=segment)
    2) 再用一个“只以最后一个 token 作为 Query”的注意力池化, 把上下文汇聚到当前 token
       并且在 attention logits 上加一个与 delta_t 相关的负偏置 (越远惩罚越大)
    3) 输出 embedding 给固定的线性分类头 (不改 classifier)
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 64,
        hidden_dims: list = [256, 128],
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.embedding_dim = int(embedding_dim)

        # 复用 hidden_dims 习惯: 
        # hidden_dims[0] -> d_model
        # hidden_dims[1] -> dim_feedforward
        d_model = int(hidden_dims[0]) if (hidden_dims and len(hidden_dims) >= 1) else 256
        dim_ff = int(hidden_dims[1]) if (hidden_dims and len(hidden_dims) >= 2) else 4 * d_model

        self.d_model = d_model
        self.dim_ff = max(dim_ff, 2 * d_model)
        self.dropout = float(dropout)

        # 内部 Transformer 超参 (可以只改这里, 不影响外部接口)
        self.num_layers = 2
        self.nhead = self._pick_nhead(d_model)

        # 1) segment token 投影: 每个 segment 的 786/781 维 -> d_model
        self.token_proj = nn.Linear(self.input_dim, d_model)

        # 2) 时间差编码: 把 log1p(delta_t) 映射到 d_model, 加到 token embedding (类似“时间位置编码”)
        #    这样即便不改 attention, 也能让模型感知不规则间隔
        self.time_mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(d_model, d_model),
        )

        # 3) 序列位置编码 (可学习): 只表达“上下文窗口内的相对顺序”, 不表达真实间隔; 为兼容变长 T, 这里给上限 max_len (足够大即可)
        self.max_len = 256
        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_len, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # 4) Transformer Encoder (对序列做上下文建模)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_ff,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN: 通常更稳
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=self.num_layers)

        # 5) “时间加权注意力池化”: 只用最后 token 当 Query, 聚合整个序列
        #    用一个可学习的 alpha (每个 head 一个) 控制“时间惩罚强度”
        head_dim = d_model // self.nhead
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        # alpha > 0, 越大表示越“只看近期”. 用 softplus 保证正数. 
        self.alpha = nn.Parameter(torch.zeros(self.nhead))  # 初始 0 -> softplus ~ 0.693

        self.pool_norm = nn.LayerNorm(d_model)
        self.out_norm = nn.LayerNorm(d_model)
        self.to_embedding = nn.Linear(d_model, self.embedding_dim)

        self.emb_dropout = nn.Dropout(self.dropout)

    @staticmethod
    def _pick_nhead(d_model: int) -> int:
        for h in (8, 4, 2, 1):
            if d_model % h == 0:
                return h
        return 1

    def forward(
        self,
        x: torch.Tensor,
        delta_t: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x:
          - [B, T, input_dim](推荐: 序列输入) 或 [B, input_dim](兼容: 视作 T=1)
        delta_t:
          - [B, T], 最后一个为 0(推荐)
          - 若 None, 则默认全 0(相当于不使用时间差)
        key_padding_mask:
          - [B, T], pad 为 True; 若 None 则不 mask
        """
        if x.dim() == 2:        # 兼容旧用法: 单段输入 -> 当作长度 1 序列
            x = x.unsqueeze(1)  # [B, 1, D]

        if x.dim() != 3 or x.size(-1) != self.input_dim:
            raise ValueError(f"Expected x shape [B, T, {self.input_dim}], got {tuple(x.shape)}")

        B, T, _ = x.shape
        if T > self.max_len:
            raise ValueError(f"T={T} exceeds max_len={self.max_len}. Increase max_len in the model.")

        # delta_t 默认全 0
        if delta_t is None:
            delta_t = x.new_zeros((B, T))
        else:
            if delta_t.dim() != 2 or delta_t.shape != (B, T):
                raise ValueError(f"Expected delta_t shape [B, T]=[{B}, {T}], got {tuple(delta_t.shape)}")
            delta_t = delta_t.to(dtype=x.dtype)

        # 1) token embedding
        tok = self.token_proj(x)  # [B, T, d_model]
        tok = self.emb_dropout(tok)

        # 2) 时间差编码 (log1p 缩放, 避免跨度太大)
        #    约定: delta_t>=0；如果给了负值, 这里也能工作, 但语义就不对了
        dt = torch.log1p(torch.clamp(delta_t, min=0.0)).unsqueeze(-1)  # [B, T, 1]
        time_emb = self.time_mlp(dt)  # [B, T, d_model]
        tok = tok + time_emb

        # 3) 序列位置编码 (窗口内相对顺序)
        tok = tok + self.pos_embed[:, :T, :]

        # 4) Transformer 编码 (上下文建模)
        h = self.encoder(tok, src_key_padding_mask=key_padding_mask)  # [B, T, d_model]

        # 5) “只用最后 token 做 Query”的时间加权注意力池化
        #    Query = h_last; Keys/Values = h_all
        h_last = h[:, -1:, :]  # [B, 1, d_model]
        q = self.q_proj(h_last)
        k = self.k_proj(h)
        v = self.v_proj(h)

        # reshape -> [B, nhead, *, head_dim]
        nhead = self.nhead
        head_dim = self.d_model // nhead

        q = q.view(B, 1, nhead, head_dim).transpose(1, 2)  # [B, nhead, 1, head_dim]
        k = k.view(B, T, nhead, head_dim).transpose(1, 2)  # [B, nhead, T, head_dim]
        v = v.view(B, T, nhead, head_dim).transpose(1, 2)  # [B, nhead, T, head_dim]

        # attention logits: [B, nhead, 1, T]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)

        # 时间惩罚 bias: 越远越负 (抑制远处 token 的注意力)
        #       bias = -softplus(alpha_h) * log1p(delta_t)
        #       delta_t 对应每个 key 位置 (序列位置), 形状 [B, 1, 1, T], 广播到各 head
        alpha = F.softplus(self.alpha).view(1, nhead, 1, 1)  # [1, nhead, 1, 1]
        dt_key = torch.log1p(torch.clamp(delta_t, min=0.0)).view(B, 1, 1, T)  # [B, 1, 1, T]
        scores = scores - alpha * dt_key

        # mask pad (pad 位置设为 -inf)
        if key_padding_mask is not None:
            if key_padding_mask.shape != (B, T):
                raise ValueError(f"Expected key_padding_mask shape [{B}, {T}], got {tuple(key_padding_mask.shape)}")
            mask = key_padding_mask.view(B, 1, 1, T)  # True=pad
            scores = scores.masked_fill(mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1)                                # [B, nhead, 1, T]
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        ctx = torch.matmul(attn, v)                                         # [B, nhead, 1, head_dim]
        ctx = ctx.transpose(1, 2).contiguous().view(B, 1, self.d_model)     # [B, 1, d_model]
        ctx = self.o_proj(ctx)                                              # [B, 1, d_model]

        # 融合: 当前 token + 聚合上下文
        fused = self.pool_norm(h_last + ctx).squeeze(1)                     # [B, d_model]

        # 输出 embedding
        fused = self.out_norm(fused)
        emb = self.to_embedding(fused)                                      # [B, embedding_dim]
        return emb

    def get_num_parameters(self) -> int:
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
