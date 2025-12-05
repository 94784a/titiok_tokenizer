# ViT Encoder for Image Encoding
# Based on DinoTransformer architecture

from functools import partial
from typing import Any, Dict, Literal, Tuple

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
    PatchEmbed,
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
    if isinstance(module, PatchEmbed):
        module.reset_parameters()
    if isinstance(module, RMSNorm):
        module.reset_parameters()


class ViTEncoder(nn.Module):
    """
    Vision Transformer Encoder for encoding images to latent representations.
    
    Args:
        img_size: Input image size (H, W)
        patch_size: Patch size for tokenization
        in_chans: Number of input channels (3 for RGB)
        embed_dim: Embedding dimension
        depth: Number of transformer encoder blocks
        num_heads: Number of attention heads
        ffn_ratio: FFN hidden dimension ratio
        num_register_tokens: Number of register tokens
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
        img_size: Tuple[int, int] = (224, 224),
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        ffn_ratio: float = 4.0,
        num_register_tokens: int = 4,
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
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_register_tokens = num_register_tokens
        
        # Calculate num_patches
        self.num_patches = (img_size[0] // patch_size, img_size[1] // patch_size)
        
        norm_layer_cls = norm_layer_dict[norm_layer]
        
        # ===== patch embedding =====
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=None,  # We'll apply norm after adding position embeddings
            flatten_embedding=True,
        )
        
        # ===== rope position embedding =====
        self.rope_embed = RopePositionEmbedding(
            embed_dim=embed_dim,
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
        
        # ===== tokens =====
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if num_register_tokens > 0:
            self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))
        else:
            self.register_tokens = None
        
        # ===== encoder blocks =====
        ffn_layer_cls = ffn_layer_dict[ffn_layer]
        self.encoder_blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim=embed_dim,
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
        self.norm = norm_layer_cls(embed_dim)
    
    # ===== weight init =====
    def init_weights(self):
        """Initialize weights following ViT conventions"""
        self.rope_embed._init_weights()
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.register_tokens is not None:
            nn.init.trunc_normal_(self.register_tokens, std=0.02)
        named_apply(init_weights_vit, self)
    
    # ===== forward =====
    def forward(
        self, 
        x: Tensor, 
        return_intermediates: bool = False,
        return_all_tokens: bool = False,
    ) -> Tensor | Tuple[Tensor, Dict[int, Tensor]]:
        """
        Args:
            x: Input images, shape [B, C, H, W]
            return_intermediates: Whether to return intermediate layer outputs
            return_all_tokens: Whether to return all tokens (including CLS and register tokens)
        
        Returns:
            If return_intermediates:
                (patch_tokens, intermediates_dict)
            Else:
                patch_tokens: shape [B, H*W, C] or [B, 1+num_register+H*W, C] if return_all_tokens
        """
        B, C, H, W = x.shape
        
        # --- Patch embedding ---
        x = self.patch_embed(x)  # [B, H*W, embed_dim]
        
        # --- Add CLS and register tokens ---
        tokens = [self.cls_token.expand(B, -1, -1)]
        if self.register_tokens is not None:
            tokens.append(self.register_tokens.expand(B, -1, -1))
        tokens.append(x)
        x = torch.cat(tokens, dim=1)  # [B, 1+num_register+H*W, embed_dim]
        
        # --- Generate rope for patch tokens only ---
        rope_sincos = self.rope_embed(H=self.num_patches[0], W=self.num_patches[1])
        
        intermediates: Dict[int, Tensor] = {}
        
        # --- Pass through encoder blocks ---
        for i, blk in enumerate(self.encoder_blocks):
            x = blk(x, rope_sincos)
            if return_intermediates:
                # Store only patch tokens in intermediates
                num_special_tokens = 1 + self.num_register_tokens
                intermediates[i] = x[:, num_special_tokens:]
        
        # --- Final normalization ---
        x_norm = self.norm(x)
        
        # --- Return appropriate tokens ---
        if return_all_tokens:
            output_tokens = x_norm
        else:
            # Return only patch tokens (without CLS and register tokens)
            num_special_tokens = 1 + self.num_register_tokens
            output_tokens = x_norm[:, num_special_tokens:]
        
        if return_intermediates:
            return output_tokens, intermediates
        return output_tokens
    
    def load_checkpoint(self, path):
        """Load pretrained weights"""
        path = Path(path)
        assert path.exists(), f"Checkpoint not found: {path}"
        pt = torch.load(str(path), map_location="cpu")
        if "model" in pt:
            pt = pt["model"]
        elif "state_dict" in pt:
            pt = pt["state_dict"]
        pt = {k.replace("module.", "") if "module." in k else k: v for k, v in pt.items()}
        pt = {k: v for k, v in pt.items() if "loss_fn." not in k}
        self.load_state_dict(pt, strict=True)
