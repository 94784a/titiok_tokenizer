# reference https://github.com/facebookresearch/dinov3/tree/main/dinov3/models/vision_transformer.py
from functools import partial
# from tkinter.messagebox import NO
from typing import Any, Dict, Literal, Tuple
from xmlrpc.client import Boolean

import torch
import torch.nn.init
from torch import Tensor, nn
from pathlib import Path

from .layers import (
    LayerScale,
    Mlp,
    PatchEmbed,
    RMSNorm,
    RopePositionEmbedding,
    SelfAttentionBlock,
    SwiGLUFFN,
)
from .layers.utils import named_apply

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


# === main transformer ===
class DinoTransformer(nn.Module):
    def __init__(
        self,
        num_patches: Tuple[int, int] = (27, 48),
        embed_dim: int = 768,
        encoder_depth: int = 12,
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
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_register_tokens = num_register_tokens

        norm_layer_cls = norm_layer_dict[norm_layer]

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
        encoder_ffn_ratio_sequence = [ffn_ratio] * encoder_depth
        self.encoder_blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    ffn_ratio=encoder_ffn_ratio_sequence[i],
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
                for i in range(encoder_depth)
            ]
        )
        self.norm = norm_layer_cls(embed_dim)

    # ===== weight init =====
    def init_weights(self):
        self.rope_embed._init_weights()
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.register_tokens is not None:
            nn.init.trunc_normal_(self.register_tokens, std=0.02)
        named_apply(init_weights_vit, self)

    # ===== forward =====
    def forward(self, x: Tensor,
                num_patches:tuple[int, int] = None,
                additional_tokens: Tensor = None, return_intermediates: bool = False) -> Dict[str, Tensor]:
        """
        Args:
            x: patch token embeddings, shape [B, H*W, C]
            additional_tokens: additional token embeddings, shape [B, L', C]
        """
        B, L, C = x.shape
        if num_patches is None:
            num_patches = self.num_patches
        assert L == num_patches[0] * num_patches[1]

        # --- 拼接 [CLS] + [REGISTER] + patch tokens ---
        tokens = [self.cls_token.expand(B, -1, -1)]
        if self.register_tokens is not None:
            tokens.append(self.register_tokens.expand(B, -1, -1))
        additional_tokens_num = 0
        if additional_tokens is not None:
            _,additional_tokens_num,_ = additional_tokens.shape
            tokens.append(additional_tokens)
        
        tokens.append(x)
        x = torch.cat(tokens, dim=1)

        # --- 生成 rope 只作用于 patch 部分 ---
        rope_sincos = self.rope_embed(H=num_patches[0], W=num_patches[1])
        intermediates: Dict[int, Tensor] = {}
        
        # --- 进入 transformer blocks ---
        for i, blk in enumerate(self.encoder_blocks):
            x = blk(x, rope_sincos)
            if return_intermediates:
                 intermediates[i]=x[:, 1 + self.num_register_tokens + additional_tokens_num:]

        # --- 最后归一化 ---
        x_norm = self.norm(x)
        if return_intermediates:
            return x_norm[:, 1 + self.num_register_tokens + additional_tokens_num:],intermediates
        return x_norm[:, 1 + self.num_register_tokens + additional_tokens_num:]

    def load_checkpoint(self, path):
        path = Path(path)
        assert path.exists()
        pt = torch.load(str(path), map_location="cpu")
        if "model" in pt:
            pt = pt["model"]
        elif "state_dict" in pt:
            pt = pt["state_dict"]
        pt = {k.replace("module.", "") if "module." in k else k: v for k, v in pt.items()}
        pt = {k: v for k, v in pt.items() if "loss_fn." not in k}
        super().load_state_dict(pt, strict=True)



class DinoV3BEncoder(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        REPO_DIR="/high_perf_store2/users/yaoziyang/public_code/dinov3"
        self.img_backbone = torch.hub.load(REPO_DIR, 'dinov3_vitb16', source='local', weights="/high_perf_store2/users/jiangyuncheng/dinov3_ckpt/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth")
        self.patch_size=16
    def _extract_img_feat(self, img):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:
            B, T, N, C, H, W = img.size()
            img = img.reshape(B * T * N, C, H, W)
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                img_feats = self.img_backbone(img,return_all=True)['x_norm_patchtokens']
        img_feats=img_feats.view(B, T,N ,H // self.patch_size, W // self.patch_size,-1)
        return img_feats
    def forward(
        self,
        img_inputs=None,#[B T N C H W]
    ):
        '''
        input :img_inputs [B T N C H W]
        return : img_feats [B,T,N,H',W',C']
        '''
        
        img_feats = self._extract_img_feat(img_inputs)
        B,T,N,H,W,C=img_feats.shape

        return img_feats
