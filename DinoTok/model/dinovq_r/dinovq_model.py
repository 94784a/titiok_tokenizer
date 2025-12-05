# Residual VQ-VAE Model with DINO Supervision
# Image → Encoder → z → VQ → e (semantic codes)
#   Residual: r = z - e (detail)
#   DINO decoder: e → DINO features (semantic)
#   Image decoder: concat(e, r) → image (detail + semantic)

from typing import Any, Dict, Literal, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .vit_encoder import ViTEncoder
from .dino_feature_decoder import DinoFeatureDecoder
from .image_decoder import ImageDecoder
from .vector_quantize_pytorch import VectorQuantize
from .fusion_module import ConcatFusion


class DinoVQModel(nn.Module):
    """
    Residual VQ-VAE with DINO supervision:
    - ViT encoder: Encodes images to latent z
    - Vector quantizer: Quantizes z → e (semantic codes)
    - Residual: r = z - e (detail information)
    - DINO decoder: Decodes e → DINO features (semantic supervision)
    - Image decoder: Decodes concat(e, r) → reconstructed image

    Args:
        # Image and patch config
        img_size: Input image size (H, W)
        patch_size: Patch size for tokenization
        in_chans: Number of input channels (3 for RGB)
        out_chans: Number of output channels (3 for RGB)

        # Encoder config
        encoder_embed_dim: Encoder embedding dimension
        encoder_depth: Number of encoder transformer blocks
        encoder_num_heads: Number of encoder attention heads

        # VQ config
        codebook_size: Number of codes in codebook
        codebook_dim: Codebook embedding dimension (if None, uses encoder_embed_dim)
                      If different from encoder_embed_dim, automatic projection layers are added
        vq_decay: EMA decay for codebook update
        vq_commitment_weight: Weight for commitment loss

        # DINO decoder config
        dino_decoder_depth: Number of DINO decoder transformer blocks
        dino_decoder_num_heads: Number of DINO decoder attention heads
        dino_dim: DINO feature dimension

        # Image decoder config
        image_decoder_dim: Image decoder internal dimension
        image_decoder_depth: Number of image decoder transformer blocks
        image_decoder_num_heads: Number of image decoder attention heads

        # Loss weights
        dino_loss_weight: Weight for DINO semantic loss
        recon_loss_weight: Weight for image reconstruction loss
        vq_loss_weight: Weight for VQ commitment loss

        # Optional: Perceptual loss (LPIPS)
        use_perceptual: Whether to use perceptual loss
        perceptual_weight: Weight for perceptual loss
        vgg_ckpt_path: Path to VGG checkpoint for LPIPS

        # Optional: GAN loss
        use_gan: Whether to use GAN discriminator
        disc_start: Step to start discriminator training
        disc_weight: Weight for discriminator loss
        disc_dim: Discriminator hidden dimension
        disc_num_layers: Number of discriminator layers
        disc_adaptive_weight: Use adaptive weight for adversarial loss

        # Shared config
        ffn_ratio: FFN hidden dimension ratio
        num_register_tokens: Number of register tokens in encoder
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
        # Image and patch config
        img_size: Tuple[int, int] = (224, 224),
        patch_size: int = 16,
        in_chans: int = 3,
        out_chans: int = 3,
        # Encoder config
        encoder_embed_dim: int = 768,
        encoder_depth: int = 12,
        encoder_num_heads: int = 12,
        # VQ config
        codebook_size: int = 8192,
        codebook_dim: int | None = None,  # If None, uses encoder_embed_dim
        vq_decay: float = 0.99,
        vq_commitment_weight: float = 0.25,
        # DINO decoder config
        dino_decoder_depth: int = 8,
        dino_decoder_num_heads: int = 12,
        dino_dim: int = 768,
        # Image decoder config
        image_decoder_dim: int = 512,
        image_decoder_depth: int = 8,
        image_decoder_num_heads: int = 8,
        # Shared config
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

        # Store config
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.encoder_embed_dim = encoder_embed_dim

        # ===== ViT Encoder =====
        self.encoder = ViTEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            ffn_ratio=ffn_ratio,
            num_register_tokens=num_register_tokens,
            pos_embed_rope_base=pos_embed_rope_base,
            pos_embed_rope_min_period=pos_embed_rope_min_period,
            pos_embed_rope_max_period=pos_embed_rope_max_period,
            pos_embed_rope_normalize_coords=pos_embed_rope_normalize_coords,
            pos_embed_rope_shift_coords=pos_embed_rope_shift_coords,
            pos_embed_rope_jitter_coords=pos_embed_rope_jitter_coords,
            pos_embed_rope_rescale_coords=pos_embed_rope_rescale_coords,
            pos_embed_rope_dtype=pos_embed_rope_dtype,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            ffn_bias=ffn_bias,
            drop_path_rate=drop_path_rate,
            layerscale_init=layerscale_init,
            norm_layer=norm_layer,
            ffn_layer=ffn_layer,
            mask_k_bias=mask_k_bias,
            device=device,
        )

        # ===== Vector Quantizer =====
        # If codebook_dim != encoder_embed_dim, VectorQuantize will automatically:
        # 1. Add project_in: encoder_embed_dim → codebook_dim
        # 2. Quantize in codebook_dim space
        # 3. Add project_out: codebook_dim → encoder_embed_dim
        self.quantizer = VectorQuantize(
            dim=encoder_embed_dim,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,  # Can be different from encoder_embed_dim
            heads=1,
            separate_codebook_per_head=False,
            # 距离度量 & 码本更新
            use_cosine_sim=False,  # 先用欧式距离, 更稳定, 也兼容affine_param
            decay=vq_decay,  # EMA更新稍微慢一点
            eps=1e-5,
            kmeans_init=True,  # 用K-means初始化码本
            kmeans_iters=20,
            sync_kmeans=True,
            threshold_ema_dead_code=1,  # 少用的code会被重置,防止死码
            # 输入格式相关
            channel_last=True,  # 输入[B, L, D]
            accept_image_fmap=False,  # 输入不是[B, C, T, H, W]
            # 损失相关
            commitment_weight=vq_commitment_weight,
            commitment_use_cross_entropy_loss=False,  # 先用MSE commitment loss
            orthogonal_reg_weight=1e-3,  # 给码本一点点正交正则
            orthogonal_reg_active_codes_only=True,
            orthogonal_reg_max_codes=512,  # 限制下参与正则的 code 数，加速
            # Gumbel / ST 相关：先不用花活
            stochastic_sample_codes=False,
            sample_codebook_temp=1.0,
            straight_through=False,
            reinmax=False,
            # 码本更新方式：经典 EMA VQ-VAE
            ema_update=True,
            learnable_codebook=False,
            in_place_codebook_optimizer=None,
            # affine param：可以先关着，如果你想 squeeze 性能可以试着打开
            affine_param=False,
            sync_affine_param=False,
            affine_param_batch_decay=0.99,
            affine_param_codebook_decay=0.9,
            # advanced sync update（先不用）
            sync_update_v=0.0,
        )

        # ===== Fusion Module (concat e and r) =====
        self.fusion = ConcatFusion(
            dim=encoder_embed_dim,
            output_dim=encoder_embed_dim,  # Keep same dimension
        )

        # ===== DINO Feature Decoder (takes e only) =====
        self.dino_decoder = DinoFeatureDecoder(
            embed_dim=encoder_embed_dim,
            dino_dim=dino_dim,
            num_patches=self.num_patches,
            depth=dino_decoder_depth,
            num_heads=dino_decoder_num_heads,
            ffn_ratio=ffn_ratio,
            pos_embed_rope_base=pos_embed_rope_base,
            pos_embed_rope_min_period=pos_embed_rope_min_period,
            pos_embed_rope_max_period=pos_embed_rope_max_period,
            pos_embed_rope_normalize_coords=pos_embed_rope_normalize_coords,
            pos_embed_rope_shift_coords=pos_embed_rope_shift_coords,
            pos_embed_rope_jitter_coords=pos_embed_rope_jitter_coords,
            pos_embed_rope_rescale_coords=pos_embed_rope_rescale_coords,
            pos_embed_rope_dtype=pos_embed_rope_dtype,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            ffn_bias=ffn_bias,
            drop_path_rate=drop_path_rate,
            layerscale_init=layerscale_init,
            norm_layer=norm_layer,
            ffn_layer=ffn_layer,
            mask_k_bias=mask_k_bias,
            device=device,
        )

        # ===== Image Decoder (takes fused e + r) =====
        self.image_decoder = ImageDecoder(
            embed_dim=encoder_embed_dim,  # After fusion, same dim
            decoder_dim=image_decoder_dim,
            num_patches=self.num_patches,
            patch_size=patch_size,
            out_chans=out_chans,
            depth=image_decoder_depth,
            num_heads=image_decoder_num_heads,
            ffn_ratio=ffn_ratio,
            pos_embed_rope_base=pos_embed_rope_base,
            pos_embed_rope_min_period=pos_embed_rope_min_period,
            pos_embed_rope_max_period=pos_embed_rope_max_period,
            pos_embed_rope_normalize_coords=pos_embed_rope_normalize_coords,
            pos_embed_rope_shift_coords=pos_embed_rope_shift_coords,
            pos_embed_rope_jitter_coords=pos_embed_rope_jitter_coords,
            pos_embed_rope_rescale_coords=pos_embed_rope_rescale_coords,
            pos_embed_rope_dtype=pos_embed_rope_dtype,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            ffn_bias=ffn_bias,
            drop_path_rate=drop_path_rate,
            layerscale_init=layerscale_init,
            norm_layer=norm_layer,
            ffn_layer=ffn_layer,
            mask_k_bias=mask_k_bias,
            device=device,
        )

    def init_weights(self):
        """Initialize all model weights"""
        self.encoder.init_weights()
        self.dino_decoder.init_weights()
        self.image_decoder.init_weights()
        # VQ and fusion will be initialized by PyTorch defaults

    def encode(
        self, imgs: Tensor, return_intermediates: bool = False
    ) -> Tensor | Tuple[Tensor, Dict]:
        """
        Encode images to latent representations.

        Args:
            imgs: Input images, shape [B, C, H, W]
            return_intermediates: Whether to return intermediate encoder outputs

        Returns:
            z: Latent tokens, shape [B, L, embed_dim]
            (optional) intermediates: Dict of intermediate outputs
        """
        return self.encoder(imgs, return_intermediates=return_intermediates)

    def quantize(self, z: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Quantize latent z to codebook codes e.

        Args:
            z: Latent tokens, shape [B, L, D]

        Returns:
            e: Quantized codes, shape [B,L, D]
            indices: Codebook indices, shape [B, L] or [B, L, H] for multi-head
            commitment_loss: VQ commitment loss
        """
        # VectorQuantize returns a dict with:
        # 'embeddings', 'encodings', 'commitment_loss', 'perplexity', 'avg_usage', 'batch_usage'
        result = self.quantizer(z)

        quantized = result["embeddings"]
        indices = result["encodings"]
        commitment_loss = result["commitment_loss"]

        return quantized, indices, commitment_loss

    def compute_residual(self, z: Tensor, e: Tensor) -> Tensor:
        """
        Compute residual r = z - e.

        Args:
            z: Original latents, shape [B, L, D]
            e: Quantized codes, shape [B, L, D]

        Returns:
            r: Residual, shape [B, L, D]
        """
        return z - e

    def fuse_latents(self, e: Tensor, r: Tensor) -> Tensor:
        """
        Fuse quantized codes e and residual r.

        Args:
            e: Quantized codes, shape [B, L, D]
            r: Residual, shape [B, L, D]

        Returns:
            fused: Fused latents, shape [B, L, D]
        """
        return self.fusion(e, r)

    def decode_dino(self, e: Tensor) -> Tensor:
        """
        Decode quantized codes to DINO features (semantic).

        Args:
            e: Quantized codes, shape [B, L, D]

        Returns:
            dino_features: DINO features, shape [B, H, W, dino_dim]
        """
        return self.dino_decoder(e, num_patches=self.num_patches)

    def decode_image(self, latents: Tensor) -> Tensor:
        """
        Decode fused latents to images.

        Args:
            latents: Fused latents (e + r), shape [B, L, D]

        Returns:
            imgs: Reconstructed images, shape [B, C, H, W]
        """
        return self.image_decoder(latents, num_patches=self.num_patches)

    def forward_encode(self, imgs: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Encode images to quantized codes and indices (inference mode).

        Args:
            imgs: Input images [B, C, H, W]

        Returns:
            e: Quantized codes [B, L, D]
            indices: Codebook indices [B, L]
        """
        z = self.encode(imgs)
        e, indices, _ = self.quantize(z)
        return e, indices

    def forward(
        self,
        imgs: Tensor,
        return_intermediates: bool = False,
        return_all: bool = True,
    ) -> Dict[str, Tensor]:
        """
        Full forward pass: encode → quantize → residual → dual decode.

        Args:
            imgs: Input images, shape [B, C, H, W]
            return_intermediates: Whether to return intermediate encoder outputs
            return_all: Whether to return all intermediate values (z, e, r, indices)

        Returns:
            dict containing:
                - "z": Original latents [B, L, D] (if return_all)
                - "e": Quantized codes [B, L, D] (if return_all)
                - "r": Residual [B, L, D] (if return_all)
                - "indices": Codebook indices (if return_all)
                - "dino_features": DINO features [B, H, W, dino_dim]
                - "reconstructed_imgs": Reconstructed images [B, C, H, W]
                - "vq_loss": VQ commitment loss
                - "intermediates": Dict of intermediate encoder outputs (if return_intermediates)
        """
        outputs = {}

        # 1. Encode: imgs → z
        if return_intermediates:
            z, intermediates = self.encode(imgs, return_intermediates=True)
            outputs["intermediates"] = intermediates
        else:
            z = self.encode(imgs)

        if return_all:
            outputs["z"] = z

        # 2. Quantize: z → e
        vq_out = self.quantizer(z)

        if isinstance(vq_out, dict):
            e = vq_out["embeddings"]
            indices = vq_out["encodings"]
            vq_loss = vq_out["commitment_loss"]
            outputs["perplexity"] = vq_out["perplexity"]
            outputs["avg_usage"] = vq_out["avg_usage"]
        else:
            # Fallback for older VQ versions or if return format changes
            e, indices, vq_loss = vq_out

        if return_all:
            outputs["e"] = e
            outputs["indices"] = indices

        outputs["vq_loss"] = vq_loss

        # 3. Compute residual: r = z - e
        r = self.compute_residual(z, e)

        if return_all:
            outputs["r"] = r

        # 4. DINO decoder: e → DINO features (semantic supervision)
        dino_features = self.decode_dino(e)
        outputs["dino_features"] = dino_features

        # 5. Fuse e and r
        fused = self.fuse_latents(e, r)

        # 6. Image decoder: fused → reconstructed image
        reconstructed_imgs = self.decode_image(fused)
        outputs["reconstructed_imgs"] = reconstructed_imgs

        return outputs
