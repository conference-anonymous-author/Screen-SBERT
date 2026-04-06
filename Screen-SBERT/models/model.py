import torch
import torch.nn as nn

from .embedding import GUIEmbeddingModule
from .encoder import TransformerEncoder
from export_onnx import export_onnx_model, make_dummy_inputs, make_trtexec_command
from optuna_train_pipeline import suggest_optuna_params as _suggest_optuna_params


class ScreenSBERT(nn.Module):
    FIXED_VISION_DIM = 768
    FIXED_TEXT_DIM = 1024
    FIXED_FUNCTION_DIM = 1024
    ONNX_INPUT_NAMES = [
        "bbox",
        "function_embedding",
        "vision_embedding",
        "text_embedding",
        "padding_mask",
    ]
    ONNX_OUTPUT_NAMES = ["screen_embedding"]

    def __init__(
        self,
        embed_dim=768,
        num_heads=8,
        num_layers=4,
        d_ff=None,
        dropout=0.1,
        vision_dim=FIXED_VISION_DIM,
        text_dim=FIXED_TEXT_DIM,
        function_dim=FIXED_FUNCTION_DIM,
        width=128,
        height=256,
        num_buckets=32,
        layer_scale_init=0.1,
        attn_dropout=None,
        max_distance=128,
        log_base=2,
        function_proj_hidden_dim=None,
        vision_proj_hidden_dim=None,
        text_proj_hidden_dim=None,
        proj_dropout=0.1,
        proj_init_scale=0.1,
    ):
        super().__init__()

        if d_ff is None:
            d_ff = embed_dim * 2

        self.gui_embedding = GUIEmbeddingModule(
            vision_dim=vision_dim,
            text_dim=text_dim,
            function_dim=function_dim,
            embed_dim=embed_dim,
            width=width,
            height=height,
            function_proj_hidden_dim=function_proj_hidden_dim,
            vision_proj_hidden_dim=vision_proj_hidden_dim,
            text_proj_hidden_dim=text_proj_hidden_dim,
            proj_dropout=proj_dropout,
            proj_init_scale=proj_init_scale,
        )
        self.transformer_encoder = TransformerEncoder(
            num_layers=num_layers,
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

    @classmethod
    def suggest_optuna_params(cls, trial):
        return _suggest_optuna_params(trial)

    @classmethod
    def from_optuna_trial(
        cls,
        trial,
        vision_dim=FIXED_VISION_DIM,
        text_dim=FIXED_TEXT_DIM,
        function_dim=FIXED_FUNCTION_DIM,
    ):
        params = cls.suggest_optuna_params(trial)
        return cls(
            vision_dim=vision_dim,
            text_dim=text_dim,
            function_dim=function_dim,
            **params,
        )

    def make_dummy_inputs(
        self,
        batch_size=1,
        num_gui=32,
        device="cpu",
        dtype=torch.float32,
    ):
        return make_dummy_inputs(
            batch_size=batch_size,
            num_gui=num_gui,
            device=device,
            dtype=dtype,
            function_dim=self.FIXED_FUNCTION_DIM,
            vision_dim=self.FIXED_VISION_DIM,
            text_dim=self.FIXED_TEXT_DIM,
        )

    def export_onnx(
        self,
        onnx_path,
        batch_size=1,
        num_gui=32,
        opset_version=17,
        dynamic_axes=True,
        dtype=torch.float32,
    ):
        export_onnx_model(
            model=self,
            onnx_path=onnx_path,
            input_names=self.ONNX_INPUT_NAMES,
            output_names=self.ONNX_OUTPUT_NAMES,
            batch_size=batch_size,
            num_gui=num_gui,
            opset_version=opset_version,
            use_dynamic_axes=dynamic_axes,
            dtype=dtype,
            function_dim=self.FIXED_FUNCTION_DIM,
            vision_dim=self.FIXED_VISION_DIM,
            text_dim=self.FIXED_TEXT_DIM,
        )

    @classmethod
    def make_trtexec_command(
        cls,
        onnx_path="model.onnx",
        plan_path="model.plan",
        min_batch=1,
        opt_batch=8,
        max_batch=16,
        min_gui=1,
        opt_gui=32,
        max_gui=128,
        use_fp16=True,
    ):
        return make_trtexec_command(
            onnx_path=onnx_path,
            plan_path=plan_path,
            function_dim=cls.FIXED_FUNCTION_DIM,
            vision_dim=cls.FIXED_VISION_DIM,
            text_dim=cls.FIXED_TEXT_DIM,
            min_batch=min_batch,
            opt_batch=opt_batch,
            max_batch=max_batch,
            min_gui=min_gui,
            opt_gui=opt_gui,
            max_gui=max_gui,
            use_fp16=use_fp16,
        )

    def forward(
        self,
        bbox,
        function_embedding,
        vision_embedding,
        text_embedding,
        padding_mask=None,
    ):
        gui_tokens = self.gui_embedding(bbox, function_embedding, vision_embedding, text_embedding)

        bbox_xyxy = bbox[..., :4].clamp(0.0, 1.0)
        center_coords = 0.5 * (bbox_xyxy[..., :2] + bbox_xyxy[..., 2:4])

        if padding_mask is None:
            padding_mask = torch.ones(
                gui_tokens.size(0),
                gui_tokens.size(1),
                dtype=torch.int32,
                device=gui_tokens.device,
            )
        else:
            padding_mask = padding_mask.to(device=gui_tokens.device)

        key_mask = padding_mask.to(dtype=torch.bool)
        enc = self.transformer_encoder(gui_tokens, center_coords, key_mask)

        # Average pooling excluding padding tokens.
        mask = key_mask.to(dtype=enc.dtype).unsqueeze(-1)
        masked_enc = enc * mask
        sum_enc = masked_enc.sum(dim=1)
        valid_token_counts = mask.sum(dim=1)
        screen_emb = sum_enc / valid_token_counts.clamp(min=1)

        return screen_emb
