import torch
import torch.nn as nn


class NonlinearProjection(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=None, dropout=0.1, init_scale=0.1):
        super().__init__()
        if hidden_dim is None:
            # Keep a small bottleneck so the module stays close to linear in few-shot settings.
            hidden_dim = max(64, min(input_dim, output_dim) // 4)

        self.norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.delta_scale = nn.Parameter(torch.tensor(float(init_scale)))
        if input_dim == output_dim:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x):
        zero_input = x.abs().sum(dim=-1, keepdim=True) == 0
        delta = self.fc2(self.dropout(self.act(self.fc1(self.norm(x)))))
        out = self.skip(x) + self.delta_scale * delta
        return out.masked_fill(zero_input, 0.0)


class GUIEmbeddingModule(nn.Module):
    def __init__(
        self,
        vision_dim=768,
        text_dim=1024,
        function_dim=1024,
        embed_dim=768,
        width=128,
        height=256,
        function_proj_hidden_dim=None,
        vision_proj_hidden_dim=None,
        text_proj_hidden_dim=None,
        proj_dropout=0.1,
        proj_init_scale=0.1,
    ):
        super().__init__()

        self.width = width
        self.height = height
        self.x0_table = nn.Embedding(width + 1, embed_dim)
        self.y0_table = nn.Embedding(height + 1, embed_dim)
        self.x1_table = nn.Embedding(width + 1, embed_dim)
        self.y1_table = nn.Embedding(height + 1, embed_dim)
        self.w_table = nn.Embedding(width + 1, embed_dim)
        self.h_table = nn.Embedding(height + 1, embed_dim)

        self.function_proj = NonlinearProjection(
            function_dim,
            embed_dim,
            hidden_dim=function_proj_hidden_dim,
            dropout=proj_dropout,
            init_scale=proj_init_scale,
        )
        self.vision_proj = NonlinearProjection(
            vision_dim,
            embed_dim,
            hidden_dim=vision_proj_hidden_dim,
            dropout=proj_dropout,
            init_scale=proj_init_scale,
        )
        self.text_proj = NonlinearProjection(
            text_dim,
            embed_dim,
            hidden_dim=text_proj_hidden_dim,
            dropout=proj_dropout,
            init_scale=proj_init_scale,
        )

    def _to_index(self, value, scale):
        return (value * scale).long().clamp(min=0, max=scale)

    def forward(self, bbox, function_embedding, vision_embedding, text_embedding):
        if (not torch.jit.is_tracing()) and bbox.size(-1) != 4:
            raise ValueError(f"bbox last dim must be 4, got {bbox.size(-1)}")
        bbox = bbox.clamp(0.0, 1.0)
        x0, y0, x1, y1 = bbox.unbind(dim=-1)
        w = (x1 - x0).clamp(0.0, 1.0)
        h = (y1 - y0).clamp(0.0, 1.0)

        coord_tokens = (
            self.x0_table(self._to_index(x0, self.width))
            + self.y0_table(self._to_index(y0, self.height))
            + self.x1_table(self._to_index(x1, self.width))
            + self.y1_table(self._to_index(y1, self.height))
            + self.w_table(self._to_index(w, self.width))
            + self.h_table(self._to_index(h, self.height))
        )

        gui_tokens = (
            coord_tokens
            + self.function_proj(function_embedding)
            + self.text_proj(text_embedding)
            + self.vision_proj(vision_embedding)
        )
        return gui_tokens
