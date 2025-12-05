# ViT Decoder for Image Reconstruction
# Reconstructs pixel-level images from latent representations

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


class ImageDecoder(nn.Module):
    """
    ViT Decoder for reconstructing images from latent representations.
    
    Args:
        embed_dim: Input embedding dimension (from encoder)
        decoder_dim: Decoder internal dimension
        num_patches: Number of patches (H, W)
        patch_size: Patch size for unpatchifying
        out_chans: Number of output channels (3 for RGB)
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
        decoder_dim: int = 512,
        num_patches: Tuple[int, int] = (14, 14),
        patch_size: int = 16,
        out_chans: int = 3,
        depth: int = 8,
        num_heads: int = 8,
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
        self.decoder_dim = decoder_dim
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.out_chans = out_chans
        self.num_heads = num_heads
        
        norm_layer_cls = norm_layer_dict[norm_layer]
        
        # ===== Input projection =====
        self.input_proj = nn.Linear(embed_dim, decoder_dim)
        
        # ===== rope position embedding =====
        self.rope_embed = RopePositionEmbedding(
            embed_dim=decoder_dim,
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
                    dim=decoder_dim,
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
        self.norm = norm_layer_cls(decoder_dim)
        
        # Project to patch pixels: patch_size^2 * out_chans
        self.predictor = nn.Linear(decoder_dim, patch_size * patch_size * out_chans)
    
    # ===== weight init =====
    def init_weights(self):
        """Initialize weights following ViT conventions"""
        self.rope_embed._init_weights()
        named_apply(init_weights_vit, self)
        # Initialize predictor
        nn.init.trunc_normal_(self.predictor.weight, std=0.02)
        if self.predictor.bias is not None:
            nn.init.zeros_(self.predictor.bias)
    
    # ===== unpatchify =====
    def unpatchify(self, x: Tensor, num_patches: Tuple[int, int]) -> Tensor:
        """
        Convert patch tokens to image.
        
        Args:
            x: Patch tokens, shape [B, H*W, patch_size^2 * C]
            num_patches: Number of patches (H, W)
        
        Returns:
            imgs: Reconstructed images, shape [B, C, H*patch_size, W*patch_size]
        """
        B, L, _ = x.shape
        H, W = num_patches
        p = self.patch_size
        c = self.out_chans
        
        assert L == H * W
        
        # Reshape to [B, H, W, p, p, C]
        x = x.reshape(B, H, W, p, p, c)
        
        # Rearrange to [B, C, H, p, W, p] then [B, C, H*p, W*p]
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()  # [B, C, H, p, W, p]
        imgs = x.reshape(B, c, H * p, W * p)  # [B, C, H*p, W*p]
        
        return imgs
    
    # ===== forward =====
    def forward(self, x: Tensor, num_patches: Tuple[int, int] | None = None) -> Tensor:
        """
        Args:
            x: Latent tokens from encoder, shape [B, L, embed_dim]
            num_patches: Optional number of patches (H, W), if None uses self.num_patches
        
        Returns:
            imgs: Reconstructed images, shape [B, C, H*patch_size, W*patch_size]
        """
        B, L, C = x.shape
        
        if num_patches is None:
            num_patches = self.num_patches
        
        assert L == num_patches[0] * num_patches[1], \
            f"Input sequence length {L} doesn't match num_patches {num_patches}"
        
        # --- Project to decoder dimension ---
        x = self.input_proj(x)  # [B, L, decoder_dim]
        
        # --- Generate rope embeddings ---
        rope_sincos = self.rope_embed(H=num_patches[0], W=num_patches[1])
        
        # --- Pass through decoder blocks ---
        for blk in self.decoder_blocks:
            x = blk(x, rope_sincos)
        
        # --- Normalize and predict ---
        x = self.norm(x)
        x = self.predictor(x)  # [B, L, patch_size^2 * out_chans]
        
        # --- Unpatchify to image ---
        imgs = self.unpatchify(x, num_patches)  # [B, out_chans, H*patch_size, W*patch_size]
        
        return imgs
