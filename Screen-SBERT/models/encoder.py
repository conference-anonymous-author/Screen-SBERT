import math

import torch
import torch.nn as nn


def relative_positional_bucket(delta, num_buckets=32, max_distance=128, log_base=2):
    sign = torch.sign(delta)
    n = torch.abs(delta)
    n = torch.clamp(n, min=1e-6, max=max_distance)
    log_index = (torch.log(n) / math.log(log_base)).floor().long()
    log_index = torch.clamp(log_index, min=0, max=num_buckets - 1)
    return log_index * sign.to(torch.long) + num_buckets - 1


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        num_buckets=32,
        attn_dropout=0.1,
        max_distance=128,
        log_base=2,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim({embed_dim}) must be divisible by num_heads({num_heads})")

        self.num_heads = num_heads
        self.d_head = embed_dim // num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.log_base = log_base

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.rpe_bias_x = nn.Embedding(num_buckets * 2 - 1, num_heads)
        self.rpe_bias_y = nn.Embedding(num_buckets * 2 - 1, num_heads)
        # Start from content-only attention and let RPE grow only if data supports it.
        nn.init.zeros_(self.rpe_bias_x.weight)
        nn.init.zeros_(self.rpe_bias_y.weight)

    def forward(self, query, key, value, coords, mask=None):
        bsz, g_q, dim = query.size()
        g_k = key.size(1)

        q = self.q_proj(query).view(bsz, g_q, self.num_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(key).view(bsz, g_k, self.num_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(value).view(bsz, g_k, self.num_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)

        # 2D Relative Positional Encoding
        coords_q = coords[:, :g_q, :]
        coords_k = coords[:, :g_k, :]
        delta = coords_q[:, :, None, :] - coords_k[:, None, :, :]
        delta_x, delta_y = delta[..., 0], delta[..., 1]

        bucket_x = relative_positional_bucket(delta_x, self.num_buckets, self.max_distance, self.log_base)
        bucket_y = relative_positional_bucket(delta_y, self.num_buckets, self.max_distance, self.log_base)

        bias_x = self.rpe_bias_x(bucket_x)
        bias_y = self.rpe_bias_y(bucket_y)
        rpe_bias = (bias_x + bias_y).permute(0, 3, 1, 2)
        scores = scores + rpe_bias

        if mask is not None:
            key_mask = mask[:, None, None, :].to(dtype=torch.bool)
            # Use a finite large negative value for ONNX / TensorRT friendliness.
            scores = scores.masked_fill(~key_mask, -1.0e4)

        attn = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(bsz, g_q, dim)
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, embed_dim, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        d_ff,
        dropout=0.1,
        num_buckets=32,
        layer_scale_init=0.1,
        attn_dropout=None,
        max_distance=128,
        log_base=2,
    ):
        super().__init__()
        if attn_dropout is None:
            attn_dropout = dropout
        self.self_attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_buckets=num_buckets,
            attn_dropout=attn_dropout,
            max_distance=max_distance,
            log_base=log_base,
        )
        self.ffn = FeedForward(embed_dim, d_ff, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        # LayerScale improves stability for small-data training with deep residual stacks.
        self.attn_scale = nn.Parameter(torch.full((embed_dim,), float(layer_scale_init)))
        self.ffn_scale = nn.Parameter(torch.full((embed_dim,), float(layer_scale_init)))

    def forward(self, x, coords, mask=None):
        x_ = self.norm1(x)
        attn_out = self.self_attn(x_, x_, x_, coords, mask)
        x = x + self.dropout(self.attn_scale * attn_out)

        x_ = self.norm2(x)
        ff_out = self.ffn(x_)
        x = x + self.dropout(self.ffn_scale * ff_out)
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers,
        embed_dim,
        num_heads,
        d_ff,
        dropout=0.1,
        num_buckets=32,
        layer_scale_init=0.1,
        attn_dropout=None,
        max_distance=128,
        log_base=2,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    num_buckets=num_buckets,
                    layer_scale_init=layer_scale_init,
                    attn_dropout=attn_dropout,
                    max_distance=max_distance,
                    log_base=log_base,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, coords, mask=None):
        for layer in self.layers:
            x = layer(x, coords, mask)
        return self.final_norm(x)
