# ViT Decoder for DINO Feature Reconstruction
# Reconstructs dense DINO features from latent representations

from functools import partial
from typing import Any, Literal, Tuple

import torch
import torch.nn as nn
from torch import Tensor

import sys
from pathlib import Path

# Import layers from dino_quant_tokenizer
sys.path.insert(0, str(Path(__file__).parent.parent / "dino_quant_tokenizer"))
from layers import (
    LayerScale,
    Mlp,
    RMSNorm,
    RopePositionEmbedding,
    SelfAttentionBlock,
    SwiGLUFFN,
)
from layers.utils import named_apply

# === layer dicts ===
ffn_layer_dict = {
    "mlp": Mlp,
    "swiglu": SwiGLUFFN,
    "swiglu32": partial(SwiGLUFFN, align_to=32),
    "swiglu64": partial(SwiGLUFFN, align_to=64),
    "swiglu128": partial(SwiGLUFFN, align_to=128),
}

norm_layer_dict = {
    "layernorm": partial(nn.LayerNorm, eps=1e-6),
    "layernormbf16": partial(nn.LayerNorm, eps=1e-5),
    "rmsnorm": RMSNorm,
}

dtype_dict = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def init_weights_vit(module: nn.Module, name: str = ""):
    if isinstance(module, nn.Linear):
        torch.nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    if isinstance(module, nn.LayerNorm):
        module.reset_parameters()
    if isinstance(module, LayerScale):
        module.reset_parameters()
    if isinstance(module, RMSNorm):
        module.reset_parameters()


class DinoFeatureDecoder(nn.Module):
    """
    ViT Decoder for reconstructing DINO features from latent representations.
    
    Args:
        embed_dim: Input embedding dimension (from encoder)
        dino_dim: Output DINO feature dimension
        num_patches: Number of patches (H, W)
        depth: Number of decoder transformer blocks
        num_heads: Number of attention heads
        ffn_ratio: FFN hidden dimension ratio
        pos_embed_rope_*: RoPE position embedding parameters
        qkv_bias: Use bias in qkv projection
        proj_bias: Use bias in output projection
        ffn_bias: Use bias in FFN
        drop_path_rate: Drop path rate
        layerscale_init: Layer scale initialization value
        norm_layer: Normalization layer type
        ffn_layer: FFN layer type
        mask_k_bias: Mask k bias in attention
        device: Device to place the model
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        dino_dim: int = 768,
        num_patches: Tuple[int, int] = (14, 14),
        depth: int = 8,
        num_heads: int = 12,
        ffn_ratio: float = 4.0,
        pos_embed_rope_base: float = 100.0,
        pos_embed_rope_min_period: float | None = None,
        pos_embed_rope_max_period: float | None = None,
        pos_embed_rope_normalize_coords: Literal["min", "max", "separate"] = "separate",
        pos_embed_rope_shift_coords: float | None = None,
        pos_embed_rope_jitter_coords: float | None = None,
        pos_embed_rope_rescale_coords: float | None = 2,
        pos_embed_rope_dtype: str = "fp32",
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop_path_rate: float = 0.0,
        layerscale_init: float | None = 1e-5,
        norm_layer: str = "layernorm",
        ffn_layer: str = "mlp",
        mask_k_bias: bool = False,
        device: Any | None = None,
    ):
        super().__init__()
        
        # ===== base configs =====
        self.embed_dim = embed_dim
        self.dino_dim = dino_dim
        self.num_patches = num_patches
        self.num_heads = num_heads
        
        norm_layer_cls = norm_layer_dict[norm_layer]
        
        # ===== Input projection (if embed_dim != dino_dim) =====
        if embed_dim != dino_dim:
            self.input_proj = nn.Linear(embed_dim, dino_dim)
        else:
            self.input_proj = nn.Identity()
        
        # ===== rope position embedding =====
        self.rope_embed = RopePositionEmbedding(
            embed_dim=dino_dim,
            num_heads=num_heads,
            base=pos_embed_rope_base,
            min_period=pos_embed_rope_min_period,
            max_period=pos_embed_rope_max_period,
            normalize_coords=pos_embed_rope_normalize_coords,
            shift_coords=pos_embed_rope_shift_coords,
            jitter_coords=pos_embed_rope_jitter_coords,
            rescale_coords=pos_embed_rope_rescale_coords,
            dtype=dtype_dict[pos_embed_rope_dtype],
            device=device,
        )
        
        # ===== decoder blocks =====
        ffn_layer_cls = ffn_layer_dict[ffn_layer]
        self.decoder_blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim=dino_dim,
                    num_heads=num_heads,
                    ffn_ratio=ffn_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    drop_path=drop_path_rate,
                    norm_layer=norm_layer_cls,
                    act_layer=nn.GELU,
                    ffn_layer=ffn_layer_cls,
                    init_values=layerscale_init,
                    mask_k_bias=mask_k_bias,
                    device=device,
                )
                for i in range(depth)
            ]
        )
        
        # ===== output layers =====
        self.norm = norm_layer_cls(dino_dim)
        # Final projection to DINO feature dimension (in case we want different output dim)
        self.head = nn.Linear(dino_dim, dino_dim)
    
    # ===== weight init =====
    def init_weights(self):
        """Initialize weights following ViT conventions"""
        self.rope_embed._init_weights()
        named_apply(init_weights_vit, self)
        # Initialize head
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)
    
    # ===== forward =====
    def forward(self, x: Tensor, num_patches: Tuple[int, int] | None = None) -> Tensor:
        """
        Args:
            x: Latent tokens from encoder, shape [B, L, embed_dim]
            num_patches: Optional number of patches (H, W), if None uses self.num_patches
        
        Returns:
            dino_features: Reconstructed DINO features, shape [B, H, W, dino_dim]
        """
        B, L, C = x.shape
        
        if num_patches is None:
            num_patches = self.num_patches
        
        assert L == num_patches[0] * num_patches[1], \
            f"Input sequence length {L} doesn't match num_patches {num_patches}"
        
        # --- Project to decoder dimension ---
        x = self.input_proj(x)  # [B, L, dino_dim]
        
        # --- Generate rope embeddings ---
        rope_sincos = self.rope_embed(H=num_patches[0], W=num_patches[1])
        
        # --- Pass through decoder blocks ---
        for blk in self.decoder_blocks:
            x = blk(x, rope_sincos)
        
        # --- Normalize and project ---
        x = self.norm(x)
        x = self.head(x)  # [B, L, dino_dim]
        
        # --- Reshape to spatial format ---
        x = x.reshape(B, num_patches[0], num_patches[1], self.dino_dim)  # [B, H, W, dino_dim]
        
        return x
