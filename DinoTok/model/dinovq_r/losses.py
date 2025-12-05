# Enhanced Loss Functions for Residual VQ-VAE with DINO Supervision
# Includes: DINO loss, reconstruction loss, VQ loss, perceptual loss (LPIPS), GAN loss

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

# Import perceptual loss and discriminators from vqvae
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.vqvae.lpips import LPIPS
from model.vqvae.discriminator_patchgan import (
    NLayerDiscriminator as PatchGANDiscriminator,
)


def hinge_d_loss(logits_real, logits_fake):
    """Hinge loss for discriminator"""
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    return 0.5 * (loss_real + loss_fake)


def hinge_gen_loss(logit_fake):
    """Hinge loss for generator"""
    return -torch.mean(logit_fake)


def adopt_weight(weight, global_step, threshold=0, value=0.0):
    """Adopt weight after certain training steps"""
    if global_step < threshold:
        weight = value
    return weight


def cosine_similarity_loss(pred: Tensor, target: Tensor, dim: int = -1) -> Tensor:
    """
    Compute cosine similarity loss for DINO feature alignment.
    Returns 1 - cosine_similarity to minimize distance.

    Args:
        pred: Predicted DINO features, shape [B, H, W, D] or [B, L, D]
        target: Target DINO features, same shape as pred
        dim: Dimension along which to compute similarity

    Returns:
        loss: Scalar tensor, 1 - mean cosine similarity
    """
    # Flatten spatial dimensions if needed
    if pred.ndim == 4:  # [B, H, W, D]
        pred = pred.flatten(1, 2)  # [B, H*W, D]
    if target.ndim == 4:
        target = target.flatten(1, 2)

    # Compute cosine similarity
    cos_sim = F.cosine_similarity(pred, target, dim=dim)  # [B, L] or [B]

    # Return 1 - similarity as loss (to minimize)
    loss = 1 - cos_sim.mean()

    return loss


def mse_reconstruction_loss(pred: Tensor, target: Tensor) -> Tensor:
    """
    Compute MSE loss for image reconstruction.

    Args:
        pred: Reconstructed images, shape [B, C, H, W]
        target: Ground truth images, shape [B, C, H, W]

    Returns:
        loss: Scalar MSE loss
    """
    return F.mse_loss(pred, target)


class DinoVQLoss(nn.Module):
    """
    Combined loss for Residual VQ-VAE with DINO supervision.
    Supports optional perceptual loss (LPIPS) and GAN loss.

    Args:
        dino_loss_weight: Weight for DINO semantic loss
        recon_loss_weight: Weight for image reconstruction loss
        vq_loss_weight: Weight for VQ commitment loss
        use_perceptual: Whether to use perceptual loss (LPIPS)
        perceptual_weight: Weight for perceptual loss
        vgg_ckpt_path: Path to VGG checkpoint for LPIPS
        use_gan: Whether to use GAN loss
        disc_start: Step to start discriminator training
        disc_loss: Discriminator loss type ('hinge')
        disc_weight: Weight for GAN discriminator loss
        disc_dim: Discriminator hidden dimension
        disc_num_layers: Number of discriminator layers
        disc_in_channels: Number of input channels for discriminator
        disc_adaptive_weight: Whether to use adaptive adversarial weight
    """

    def __init__(
        self,
        dino_loss_weight: float = 1.0,
        recon_loss_weight: float = 1.0,
        vq_loss_weight: float = 1.0,
        # Perceptual loss
        use_perceptual: bool = False,
        perceptual_weight: float = 1.0,
        vgg_ckpt_path: str = "/high_perf_store2/users/zhuzeyu/ckpt/pretrained/vgg16-397923afv2.pth",
        # GAN loss
        use_gan: bool = False,
        disc_start: int = 10000,
        disc_loss: str = "hinge",
        disc_weight: float = 0.5,
        disc_dim: int = 64,
        disc_num_layers: int = 3,
        disc_in_channels: int = 3,
        disc_adaptive_weight: bool = False,
    ):
        super().__init__()

        self.dino_loss_weight = dino_loss_weight
        self.recon_loss_weight = recon_loss_weight
        self.vq_loss_weight = vq_loss_weight

        # Perceptual loss (LPIPS)
        self.use_perceptual = use_perceptual
        self.perceptual_weight = perceptual_weight
        if use_perceptual:
            self.perceptual_loss = LPIPS(vgg_ckpt_path=vgg_ckpt_path).eval()
        else:
            self.perceptual_loss = None

        # GAN loss
        self.use_gan = use_gan
        if use_gan:
            assert disc_loss == "hinge", f"Only 'hinge' loss supported, got {disc_loss}"
            self.discriminator = PatchGANDiscriminator(
                input_nc=disc_in_channels,
                n_layers=disc_num_layers,
                ndf=disc_dim,
            )
            self.disc_loss_fn = hinge_d_loss
            self.gen_adv_loss_fn = hinge_gen_loss
            self.disc_start = disc_start
            self.disc_weight = disc_weight
            self.disc_adaptive_weight = disc_adaptive_weight
        else:
            self.discriminator = None

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer):
        """Calculate adaptive weight for adversarial loss"""
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        return d_weight

    def forward(
        self,
        pred_dino: Tensor,
        target_dino: Tensor,
        pred_img: Tensor,
        target_img: Tensor,
        vq_loss: Tensor,
        update_type: str = "generator",
        global_step: int = 0,
        last_layer: Optional[nn.Parameter] = None,
    ) -> dict:
        """
        Compute combined loss.

        Args:
            pred_dino: Predicted DINO features [B, H, W, D]
            target_dino: Ground truth DINO features [B, H, W, D]
            pred_img: Reconstructed images [B, C, H, W]
            target_img: Ground truth images [B, C, H, W]
            vq_loss: VQ commitment loss from quantizer
            update_type: 'generator' or 'discriminator'
            global_step: Current training step
            last_layer: Last layer of decoder for adaptive weight

        Returns:
            dict with losses
        """

        if update_type == "generator":
            # === Generator update ===

            # 1. DINO semantic loss
            dino_loss = cosine_similarity_loss(pred_dino, target_dino)
            dino_loss_weighted = self.dino_loss_weight * dino_loss

            # 2. Image reconstruction loss
            recon_loss = mse_reconstruction_loss(pred_img, target_img)
            recon_loss_weighted = self.recon_loss_weight * recon_loss

            # 3. VQ loss
            vq_loss_weighted = self.vq_loss_weight * vq_loss

            # 4. Perceptual loss (optional)
            if self.use_perceptual and self.perceptual_loss is not None:
                p_loss = self.perceptual_loss(pred_img, target_img).mean()
                p_loss_weighted = self.perceptual_weight * p_loss
            else:
                p_loss = torch.tensor(0.0, device=pred_img.device)
                p_loss_weighted = torch.tensor(0.0, device=pred_img.device)

            # 5. GAN loss (optional)
            if self.use_gan and self.discriminator is not None:
                logits_fake = self.discriminator(pred_img.contiguous())
                gen_adv_loss_raw = self.gen_adv_loss_fn(logits_fake)

                # Adaptive weight or fixed weight
                if self.disc_adaptive_weight and last_layer is not None:
                    nll_loss = recon_loss_weighted + p_loss_weighted
                    disc_adaptive_weight = self.calculate_adaptive_weight(
                        nll_loss, gen_adv_loss_raw, last_layer=last_layer
                    )
                else:
                    disc_adaptive_weight = 1.0

                # Adopt weight based on training step
                disc_weight = adopt_weight(
                    self.disc_weight,
                    global_step,
                    threshold=self.disc_start,
                )

                gen_adv_loss = disc_adaptive_weight * disc_weight * gen_adv_loss_raw
            else:
                gen_adv_loss_raw = torch.tensor(0.0, device=pred_img.device)
                gen_adv_loss = torch.tensor(0.0, device=pred_img.device)

            # Total loss
            total_loss = (
                dino_loss_weighted
                + recon_loss_weighted
                + vq_loss_weighted
                + p_loss_weighted
                + gen_adv_loss
            )

            return {
                "total_loss": total_loss,
                "dino_loss": dino_loss,  # Unweighted for logging
                "recon_loss": recon_loss,
                "vq_loss": vq_loss,
                "perceptual_loss": p_loss,
                "gen_adv_loss": gen_adv_loss_raw,
                "dino_loss_weighted": dino_loss_weighted,
                "recon_loss_weighted": recon_loss_weighted,
                "vq_loss_weighted": vq_loss_weighted,
                "perceptual_loss_weighted": p_loss_weighted,
                "gen_adv_loss_weighted": gen_adv_loss,
            }

        elif update_type == "discriminator":
            # === Discriminator update ===
            if not self.use_gan or self.discriminator is None:
                # No discriminator, return zero loss
                return {
                    "disc_loss": torch.tensor(0.0, device=pred_img.device),
                    "disc_weight": 0.0,
                    "logits_real": 0.0,
                    "logits_fake": 0.0,
                }

            logits_real = self.discriminator(target_img.contiguous().detach())
            logits_fake = self.discriminator(pred_img.contiguous().detach())

            disc_weight = adopt_weight(
                self.disc_weight,
                global_step,
                threshold=self.disc_start,
            )

            disc_loss = disc_weight * self.disc_loss_fn(logits_real, logits_fake)

            return {
                "disc_loss": disc_loss,
                "disc_weight": disc_weight,
                "logits_real": logits_real.mean().item(),
                "logits_fake": logits_fake.mean().item(),
            }

        else:
            raise ValueError(f"Unknown update_type: {update_type}")


# Simplified version for backward compatibility
class ResidualVQLoss(nn.Module):
    """
    Simple combined loss for Residual VQ-VAE (no perceptual/GAN).
    For backward compatibility.
    """

    def __init__(
        self,
        dino_loss_weight: float = 1.0,
        recon_loss_weight: float = 1.0,
        vq_loss_weight: float = 1.0,
    ):
        super().__init__()

        self.dino_loss_weight = dino_loss_weight
        self.recon_loss_weight = recon_loss_weight
        self.vq_loss_weight = vq_loss_weight

    def forward(
        self,
        pred_dino: Tensor,
        target_dino: Tensor,
        pred_img: Tensor,
        target_img: Tensor,
        vq_loss: Tensor,
    ) -> dict:
        """
        Compute combined loss.

        Args:
            pred_dino: Predicted DINO features [B, H, W, D]
            target_dino: Ground truth DINO features [B, H, W, D]
            pred_img: Reconstructed images [B, C, H, W]
            target_img: Ground truth images [B, C, H, W]
            vq_loss: VQ commitment loss from quantizer

        Returns:
            dict with total_loss and individual losses
        """
        # DINO semantic loss
        dino_loss = cosine_similarity_loss(pred_dino, target_dino)

        # Image reconstruction loss
        recon_loss = mse_reconstruction_loss(pred_img, target_img)

        # Weighted sum
        total_loss = (
            self.dino_loss_weight * dino_loss
            + self.recon_loss_weight * recon_loss
            + self.vq_loss_weight * vq_loss
        )

        return {
            "total_loss": total_loss,
            "dino_loss": dino_loss,
            "recon_loss": recon_loss,
            "vq_loss": vq_loss,
        }
