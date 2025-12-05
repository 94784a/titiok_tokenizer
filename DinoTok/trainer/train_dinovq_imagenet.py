"""
训练DinoVQ Residual VQ-VAE模型
包含图像重建和DINO特征重建的可视化
"""

import os
import json
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path
from accelerate import Accelerator, DistributedDataParallelKwargs, DistributedType
from accelerate.utils import set_seed
from ema_pytorch import EMA
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance

from data.dataset.imagenet import build_imagenet, ImageNet
from model.dinovq_r import DinoVQModel, DinoVQLoss

from trainer.utils import cycle, accum_log, unnormalize, count_params_b

set_seed(42)  # 设置全局可复现种子
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images"""

    def __init__(self, window_size=11, channel=3):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.channel = channel
        self.window = self.create_window(window_size, channel)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor(
            [
                np.exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
                for x in range(window_size)
            ]
        )
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            self.window = window

        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = (
            F.conv2d(img1 * img1, window, padding=self.window_size // 2, groups=channel)
            - mu1_sq
        )
        sigma2_sq = (
            F.conv2d(img2 * img2, window, padding=self.window_size // 2, groups=channel)
            - mu2_sq
        )
        sigma12 = (
            F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=channel)
            - mu1_mu2
        )

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )
        return ssim_map.mean()


class DinoVQEvaluator:
    def __init__(self, device):
        self.device = device
        self.ssim = SSIM().to(device)
        # Sobel filters for edge detection
        self.sobel_x = (
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
            .view(1, 1, 3, 3)
            .to(device)
        )
        self.sobel_y = (
            torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
            .view(1, 1, 3, 3)
            .to(device)
        )

        # FID metric
        # feature=2048 for InceptionV3
        self.fid = FrechetInceptionDistance(
            feature=2048,
            normalize=True,
            feature_extractor_weights_path="/high_perf_store2/users/zhuzeyu/ckpt/pretrained/weights-inception-2015-12-05-6726825d.pth",
        ).to(device)

    def reset_fid(self):
        self.fid.reset()

    def update_fid(self, real, recon):
        """
        Update FID statistics.
        real, recon: [B, C, H, W], in [0, 1] range (normalize=True handles scaling to [0, 255] internally if needed,
        but torchmetrics FID with normalize=True expects [0, 1] float images)
        """
        self.fid.update(real, real=True)
        self.fid.update(recon, real=False)

    def compute_fid(self):
        return self.fid.compute().item()

    def compute_image_metrics(self, real, recon, lpips_fn=None):
        """
        Compute standard image quality metrics.
        real, recon: [B, C, H, W], normalized to [0, 1] usually, but LPIPS expects [-1, 1]
        """
        metrics = {}

        # MSE / MAE / PSNR (assume inputs are [0, 1])
        mse = F.mse_loss(real, recon)
        mae = F.l1_loss(real, recon)
        psnr = 10 * torch.log10(1.0 / (mse + 1e-8))

        metrics["mse"] = mse.item()
        metrics["mae"] = mae.item()
        metrics["psnr"] = psnr.item()

        # SSIM
        metrics["ssim"] = self.ssim(real, recon).item()

        # LPIPS (expects [-1, 1])
        if lpips_fn is not None:
            real_norm = real * 2 - 1
            recon_norm = recon * 2 - 1
            lpips_val = lpips_fn(real_norm, recon_norm).mean()
            metrics["lpips"] = lpips_val.item()

        return metrics

    def compute_freq_metrics(self, real, recon):
        """
        Compute frequency domain and detail preservation metrics.
        """
        metrics = {}

        # 1. FFT Analysis
        # Convert to grayscale for FFT
        real_gray = real.mean(dim=1, keepdim=True)
        recon_gray = recon.mean(dim=1, keepdim=True)

        fft_real = torch.fft.fft2(real_gray)
        fft_recon = torch.fft.fft2(recon_gray)

        # Magnitude spectrum
        mag_real = torch.abs(fft_real)
        mag_recon = torch.abs(fft_recon)

        # Log magnitude for better visualization/metric
        log_mag_real = torch.log(mag_real + 1e-8)
        log_mag_recon = torch.log(mag_recon + 1e-8)

        # L1 difference in frequency domain (focus on magnitude)
        fft_l1 = F.l1_loss(log_mag_real, log_mag_recon)
        metrics["fft_l1"] = fft_l1.item()

        # 2. Edge Analysis (Sobel)
        def get_edges(img):
            # img: [B, 1, H, W]
            edge_x = F.conv2d(img, self.sobel_x, padding=1)
            edge_y = F.conv2d(img, self.sobel_y, padding=1)
            return torch.sqrt(edge_x**2 + edge_y**2 + 1e-8)

        edge_real = get_edges(real_gray)
        edge_recon = get_edges(recon_gray)

        edge_l1 = F.l1_loss(edge_real, edge_recon)
        metrics["edge_l1"] = edge_l1.item()

        return metrics


class DinoVQTrainer(nn.Module):
    def __init__(
        self,
        dinovq_model: DinoVQModel,
        loss_fn: DinoVQLoss,
        dino_vit: nn.Module,
        num_train_steps: int,
        batch_size: int,
        train_dataset: ImageNet,
        val_dataset: ImageNet,
        image_size=(224, 224),
        dataloader_num_workers=16,
        dataloader_prefetch_factor=2,
        lr=3e-4,
        gradient_accumulation_steps=1,
        weight_decay=0.0,
        max_grad_norm=0.5,
        save_results_interval=500,
        save_ckpt_interval=1000,
        output_dir="./results",
        tensorboard_log_dir=None,
        use_ema=False,
        ema_update_after_step=0,
        ema_update_every=1,
        accelerate_kwargs=dict(),
        residual_reg_weight=0.0,
    ):
        super().__init__()
        self.dinovq_model = dinovq_model
        self.loss_fn = loss_fn
        self.dino_vit = dino_vit
        self.image_size = image_size
        self.patch_size = 16
        self.lr = lr
        self.use_ema = use_ema
        self.use_gan = self.loss_fn.use_gan

        # 初始化accelerator
        # If using GAN, we might have unused parameters in generator step (discriminator)
        # but we handle that by separating them.
        # However, if we use the "Swap and Wrap" method, dinovq_model doesn't have discriminator params.
        # So find_unused_parameters=False should be fine for generator.
        # But let's set it to True just in case to avoid issues.
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=self.use_gan)
        self.accelerator = Accelerator(
            **accelerate_kwargs, kwargs_handlers=[ddp_kwargs]
        )

        if self.is_main_process and use_ema:
            self.ema_dinovq_model = EMA(
                self.dinovq_model,
                update_after_step=ema_update_after_step,
                update_every=ema_update_every,
            )
        self.register_buffer("steps", torch.Tensor([0]))

        self.num_train_steps = num_train_steps
        self.batch_size = batch_size
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.save_results_interval = save_results_interval
        self.save_ckpt_interval = save_ckpt_interval

        # Generator optimizer
        self.optim = torch.optim.AdamW(
            self.dinovq_model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
        )

        # Discriminator optimizer (if using GAN)
        if self.use_gan:
            self.discr_optim = torch.optim.AdamW(
                self.loss_fn.discriminator.parameters(),
                lr=lr,  # Use same LR for now
                weight_decay=weight_decay,
                betas=(0.9, 0.95),
            )
        else:
            self.discr_optim = None

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=dataloader_num_workers,
            pin_memory=True,
            prefetch_factor=dataloader_prefetch_factor,
        )
        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=dataloader_num_workers,
        )

        self.dino_vit.eval()
        self.dino_vit.requires_grad_(False)
        self.dino_vit.to(self.device)

        if self.use_gan:
            (
                self.dinovq_model,
                self.loss_fn,
                self.optim,
                self.discr_optim,
                self.train_dataloader,
                self.val_dataloader,
            ) = self.accelerator.prepare(
                self.dinovq_model,
                self.loss_fn,
                self.optim,
                self.discr_optim,
                self.train_dataloader,
                self.val_dataloader,
            )
        else:
            (
                self.dinovq_model,
                self.loss_fn,
                self.optim,
                self.train_dataloader,
                self.val_dataloader,
            ) = self.accelerator.prepare(
                self.dinovq_model,
                self.loss_fn,
                self.optim,
                self.train_dataloader,
                self.val_dataloader,
            )
        if self.is_main_process and use_ema:
            base_model = self.accelerator.unwrap_model(self.dinovq_model)
            self.ema_dinovq_model = EMA(
                base_model,
                update_after_step=ema_update_after_step,
                update_every=ema_update_every,
            )

        self.residual_reg_weight = residual_reg_weight
        self.train_dataloader_iter = cycle(self.train_dataloader)
        self.val_dataloader_iter = cycle(self.val_dataloader)

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.tensorboard_log_dir = Path(
            tensorboard_log_dir
            if tensorboard_log_dir is not None
            else self.output_dir / "tb_logs"
        )
        self.tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.tensorboard_log_dir))
        if self.is_main_process:
            self.evaluator = DinoVQEvaluator(self.device)
        self.print(
            f"[Rank {self.accelerator.process_index}] READY on device {self.device}"
        )

    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_main_process(self):
        return self.accelerator.is_main_process

    @property
    def is_local_main_process(self):
        return self.accelerator.is_local_main_process

    @property
    def is_distributed(self):
        return not (
            self.accelerator.distributed_type == DistributedType.NO
            and self.accelerator.num_processes == 1
        )

    def print(self, msg):
        self.accelerator.print(msg)

    def save_ckpt(self, path):
        if not self.is_main_process:
            return

        pkg = dict(
            dinovq_model=self.accelerator.get_state_dict(self.dinovq_model),
            loss_fn=self.accelerator.get_state_dict(self.loss_fn),
            optim=self.optim.state_dict(),
            steps=self.steps,
        )
        if self.use_gan:
            pkg["discr_optim"] = self.discr_optim.state_dict()

        torch.save(pkg, path)

    def load_ckpt(self, path):
        path = Path(path)
        assert path.exists(), f"Checkpoint load path {str(path)} does not exist."
        pkg = torch.load(path)

        dinovq_model = self.accelerator.unwrap_model(self.dinovq_model)
        dinovq_model.load_state_dict(pkg["dinovq_model"])

        loss_fn = self.accelerator.unwrap_model(self.loss_fn)
        loss_fn.load_state_dict(pkg["loss_fn"])

        self.optim.load_state_dict(pkg["optim"])

        if self.use_gan and "discr_optim" in pkg:
            self.discr_optim.load_state_dict(pkg["discr_optim"])

        self.steps = pkg["steps"]

    def visualize_dino_features_pca(
        self, pred_dino, target_dino, tag, steps, max_samples=4
    ):
        """
        使用PCA降维可视化DINO特征

        Args:
            pred_dino: 预测的DINO特征 [B, H, W, D]
            target_dino: 真值DINO特征 [B, H, W, D]
            tag: tensorboard tag
            steps: 当前步数
            max_samples: 最多可视化的样本数
        """
        if not self.is_main_process:
            return

        B, H, W, D = pred_dino.shape
        n_samples = min(B, max_samples)

        # 取前n_samples个样本
        pred = pred_dino[:n_samples].cpu().detach().float()  # [n, H, W, D]
        target = target_dino[:n_samples].cpu().detach().float()

        # Reshape to [n*H*W, D]
        pred_flat = pred.reshape(-1, D).numpy()
        target_flat = target.reshape(-1, D).numpy()

        # PCA降维到3维用于RGB可视化
        pca = PCA(n_components=3)

        # 合并pred和target一起做PCA，确保在同一空间
        combined = np.concatenate([pred_flat, target_flat], axis=0)
        pca.fit(combined)

        pred_pca = pca.transform(pred_flat)  # [n*H*W, 3]
        target_pca = pca.transform(target_flat)  # [n*H*W, 3]

        # 归一化到[0, 1]用于RGB显示
        def normalize_for_vis(x):
            x_min = x.min(axis=0, keepdims=True)
            x_max = x.max(axis=0, keepdims=True)
            return (x - x_min) / (x_max - x_min + 1e-8)

        pred_pca_norm = normalize_for_vis(pred_pca)
        target_pca_norm = normalize_for_vis(target_pca)

        # Reshape回 [n, H, W, 3]
        pred_rgb = pred_pca_norm.reshape(n_samples, H, W, 3)
        target_rgb = target_pca_norm.reshape(n_samples, H, W, 3)

        # 转换为 [n, 3, H, W] for tensorboard
        pred_rgb = torch.from_numpy(pred_rgb).permute(0, 3, 1, 2).float()
        target_rgb = torch.from_numpy(target_rgb).permute(0, 3, 1, 2).float()

        # 并排显示 target | pred
        comparison = torch.cat([target_rgb, pred_rgb], dim=3)  # [n, 3, H, 2W]

        # 创建grid
        grid = make_grid(comparison, nrow=1, normalize=False, pad_value=1.0)

        # 保存到tensorboard
        self.writer.add_image(tag, grid, steps)

        # 也保存为图片文件
        save_path = self.output_dir / f"{tag.replace('/', '_')}_{steps}.png"
        save_image(grid, save_path)

        self.print(f"Saved DINO PCA visualization to {save_path}")

    def visualize_image_reconstruction(
        self, imgs, recon_imgs, tag, steps, max_samples=8
    ):
        """
        可视化图像重建结果

        Args:
            imgs: 原始图像 [B, 3, H, W] (已归一化)
            recon_imgs: 重建图像 [B, 3, H, W]
            tag: tensorboard tag
            steps: 当前步数
            max_samples: 最多可视化的样本数
        """
        if not self.is_main_process:
            return

        n_samples = min(imgs.size(0), max_samples)

        # 取前n_samples个样本
        imgs = imgs[:n_samples].cpu()
        recon_imgs = recon_imgs[:n_samples].cpu()

        # 反归一化到[0, 1]用于显示
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        imgs_unnorm = imgs * std + mean
        recon_unnorm = recon_imgs * std + mean

        # Clamp到[0, 1]
        imgs_unnorm = torch.clamp(imgs_unnorm, 0, 1)
        recon_unnorm = torch.clamp(recon_unnorm, 0, 1)

        # 计算误差图（逐像素MSE）
        error_map = (
            (imgs_unnorm - recon_unnorm).pow(2).mean(dim=1, keepdim=True)
        )  # [n, 1, H, W]

        # 归一化误差图用于显示
        error_map = (error_map - error_map.min()) / (
            error_map.max() - error_map.min() + 1e-8
        )
        error_map = error_map.repeat(1, 3, 1, 1)  # 转为RGB

        # 并排显示：原图 | 重建 | 误差
        comparison = torch.cat(
            [imgs_unnorm, recon_unnorm, error_map], dim=3
        )  # [n, 3, H, 3W]

        # 创建grid
        grid = make_grid(comparison, nrow=1, normalize=False, pad_value=1.0)

        # 保存到tensorboard
        self.writer.add_image(tag, grid, steps)

        # 也保存为图片文件
        save_path = self.output_dir / f"{tag.replace('/', '_')}_{steps}.png"
        save_image(grid, save_path)

        self.print(f"Saved image reconstruction visualization to {save_path}")

    def visualize_disentanglement(self, val_model, img, tag, steps, max_samples=8):
        """
        Visualize disentanglement of semantics (e) and details (r).
        1. Original
        2. e only (r=0)
        3. r only (e=mean)
        4. Full reconstruction (e+r)
        """
        if not self.is_main_process:
            return

        B = img.shape[0]
        n_samples = min(B, max_samples)
        img = img[:n_samples].to(self.device)

        with torch.inference_mode():
            # 1. Encode
            z = val_model.encode(img)

            # 2. Quantize -> e
            vq_out = val_model.quantizer(z)
            if isinstance(vq_out, dict):
                e = vq_out["embeddings"]
            else:
                e, _, _ = vq_out

            # 3. Residual -> r
            r = val_model.compute_residual(z, e)

            # --- Ablation 1: e only (r=0) ---
            r_zero = torch.zeros_like(r)
            fused_e_only = val_model.fuse_latents(e, r_zero)
            recon_e_only = val_model.decode_image(fused_e_only)

            # --- Ablation 2: r only (e=mean) ---
            e_mean = torch.zeros_like(e)

            fused_r_only = val_model.fuse_latents(e_mean, r)
            recon_r_only = val_model.decode_image(fused_r_only)

            # --- Full Reconstruction ---
            fused_full = val_model.fuse_latents(e, r)
            recon_full = val_model.decode_image(fused_full)

            # --- another random noise as residual ---
            eps = 1e-6
            r_mean = r.mean(dim=(0, 1), keepdim=True)  # [1, 1, D]
            r_std = r.std(dim=(0, 1), unbiased=False, keepdim=True) + eps
            r_rand = torch.randn_like(r) * r_std * 10 + r_mean
            fused_rand = val_model.fuse_latents(e, r_rand)
            recon_rand = val_model.decode_image(fused_rand)

        # Prepare for visualization
        def _prep(x):
            # Assuming unnormalize is defined elsewhere or needs to be added
            # For now, using the same unnormalization logic as visualize_image_reconstruction
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
            return (x * std + mean).clamp(0, 1).cpu().detach()

        img_vis = _prep(img)
        recon_e_vis = _prep(recon_e_only)
        recon_r_vis = _prep(recon_r_only)
        recon_full_vis = _prep(recon_full)
        recon_rand_vis = _prep(recon_rand)

        # Concatenate: Original | e-only | r-only | Full | recon_rand_vis
        comparison = torch.cat(
            [img_vis, recon_e_vis, recon_r_vis, recon_full_vis, recon_rand_vis], dim=3
        )

        # Grid
        grid = make_grid(comparison, nrow=1, normalize=False, pad_value=1.0)

        # Save
        self.writer.add_image(tag, grid, steps)
        save_path = self.output_dir / f"{tag.replace('/', '_')}_{steps}.png"
        save_image(grid, save_path)
        self.print(f"Saved disentanglement visualization to {save_path}")

    def train_step(self):
        device = self.device
        steps = int(self.steps.item())
        logs = {}

        ################ train ###############
        self.dinovq_model.train()
        self.loss_fn.train()

        for _ in range(self.gradient_accumulation_steps):
            img, label = next(self.train_dataloader_iter)
            img = img.to(device)

            # 获取DINO真值特征 [B, L, D]
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                dino_features_gt = self.dino_vit(img, return_all=True)[
                    "x_norm_patchtokens"
                ]

            # === Generator Training ===
            # Freeze discriminator if using GAN
            if self.use_gan:
                # Access underlying module if wrapped by DDP
                loss_module = (
                    self.loss_fn.module
                    if hasattr(self.loss_fn, "module")
                    else self.loss_fn
                )
                for p in loss_module.discriminator.parameters():
                    p.requires_grad = False
                self.loss_fn.eval()  # Optional, but good practice

            # Forward pass (Generator)
            outputs = self.dinovq_model(img, return_all=True)

            # Compute generator losses
            losses = self.loss_fn(
                pred_dino=outputs["dino_features"],
                target_dino=dino_features_gt,
                pred_img=outputs["reconstructed_imgs"],
                target_img=img,
                vq_loss=outputs["vq_loss"],
                update_type="generator",
                global_step=steps,
            )

            residual_reg_loss = (outputs["r"] ** 2).mean() * self.residual_reg_weight

            total_loss = losses["total_loss"] + residual_reg_loss

            gas = self.gradient_accumulation_steps
            self.accelerator.backward(total_loss / gas)

            # Log generator losses
            gen_log = {
                "loss": total_loss.item() / gas,
                "dino_loss": losses["dino_loss"].item() / gas,
                "recon_loss": losses["recon_loss"].item() / gas,
                "vq_loss": losses["vq_loss"].item() / gas,
                "r_regularization_loss": residual_reg_loss.item() / gas,
            }
            if "perceptual_loss" in losses:
                gen_log["perceptual_loss"] = losses["perceptual_loss"].item() / gas
            if "gen_adv_loss" in losses:
                gen_log["gen_adv_loss"] = losses["gen_adv_loss"].item() / gas

            # Log VQ metrics
            if "perplexity" in outputs:
                gen_log["perplexity"] = outputs["perplexity"].item()
            if "avg_usage" in outputs:
                gen_log["avg_usage"] = outputs["avg_usage"].item()

            accum_log(logs, gen_log)

            # === Discriminator Training (Optional) ===
            if self.use_gan:
                # Unfreeze discriminator
                loss_module = (
                    self.loss_fn.module
                    if hasattr(self.loss_fn, "module")
                    else self.loss_fn
                )
                for p in loss_module.discriminator.parameters():
                    p.requires_grad = True
                self.loss_fn.train()

                # Compute discriminator losses
                # Note: We pass detached reconstructions
                disc_losses = self.loss_fn(
                    pred_dino=None,  # Not used for discriminator
                    target_dino=None,
                    pred_img=outputs["reconstructed_imgs"].detach(),
                    target_img=img,
                    vq_loss=None,
                    update_type="discriminator",
                    global_step=steps,
                )

                disc_loss = disc_losses["disc_loss"]
                self.accelerator.backward(disc_loss / gas)

                accum_log(
                    logs,
                    {
                        "disc_loss": disc_loss.item() / gas,
                        "logits_real": disc_losses["logits_real"],
                        "logits_fake": disc_losses["logits_fake"],
                        "disc_weight": disc_losses["disc_weight"],
                    },
                )

        if self.max_grad_norm is not None:
            self.accelerator.clip_grad_norm_(
                self.dinovq_model.parameters(), self.max_grad_norm
            )
            if self.use_gan:
                loss_module = (
                    self.loss_fn.module
                    if hasattr(self.loss_fn, "module")
                    else self.loss_fn
                )
                self.accelerator.clip_grad_norm_(
                    loss_module.discriminator.parameters(), self.max_grad_norm
                )

        self.optim.step()
        self.optim.zero_grad()

        if self.use_gan:
            self.discr_optim.step()
            self.discr_optim.zero_grad()

        if self.is_main_process:
            for k, v in logs.items():
                self.writer.add_scalar(f"train/{k}", v, steps)

        if self.is_main_process and self.use_ema:
            self.ema_dinovq_model.update()

        ################ validate ###############
        if not (steps % self.save_results_interval):
            self.dinovq_model.eval()
            unwrapped_model = self.accelerator.unwrap_model(self.dinovq_model)
            val_model = (
                self.ema_dinovq_model.ema_model
                if (self.use_ema and self.is_main_process)
                else unwrapped_model
            )

            with torch.inference_mode():
                img, label = next(self.val_dataloader_iter)
                img = img.to(device)

                # 获取DINO真值特征
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    dino_features_patches = self.dino_vit(img, return_all=True)[
                        "x_norm_patchtokens"
                    ]

                # DinoVQ前向传播
                outputs = val_model(img, return_all=True)

                # 获取输出
                z = outputs["z"]  # [B, L, D]
                e = outputs["e"]  # quantized
                r = outputs["r"]  # residual
                dino_features_pred = outputs["dino_features"]  # [B, H, W, D]
                recon_imgs = outputs["reconstructed_imgs"]  # [B, 3, H, W]
                vq_loss = outputs["vq_loss"]

                # 重塑DINO features GT到spatial format [B, H, W, D]
                B, L, D = dino_features_patches.shape
                H = W = int(L**0.5)
                dino_features_gt = dino_features_patches.reshape(B, H, W, D)

                # 计算指标
                # 1. DINO特征相似度
                dino_features_pred_flat = dino_features_pred.flatten(
                    1, 2
                )  # [B, H*W, D]
                dino_features_gt_flat = dino_features_gt.flatten(1, 2)

                pred_norm = F.normalize(dino_features_pred_flat, dim=-1)
                gt_norm = F.normalize(dino_features_gt_flat, dim=-1)
                dino_cos_sim = (pred_norm * gt_norm).sum(-1).mean()
                dino_mse = (dino_features_pred - dino_features_gt).pow(2).mean()

                # 3. 残差统计
                residual_norm = r.pow(2).mean()
                residual_ratio = residual_norm / (z.pow(2).mean() + 1e-8)

                val_logs = {
                    "val/dino_cos_sim": dino_cos_sim.item(),
                    "val/dino_mse": dino_mse.item(),
                    "val/vq_loss": vq_loss.item(),
                    "val/residual_norm": residual_norm.item(),
                    "val/residual_ratio": residual_ratio.item(),
                }

                if self.is_main_process:
                    # 可视化：图像重建
                    self.visualize_image_reconstruction(
                        imgs=img,
                        recon_imgs=recon_imgs,
                        tag="val/image_reconstruction",
                        steps=steps,
                        max_samples=8,
                    )

                    # 可视化：DINO特征 PCA
                    self.visualize_dino_features_pca(
                        pred_dino=dino_features_pred,
                        target_dino=dino_features_gt,
                        tag="val/dino_features_pca",
                        steps=steps,
                        max_samples=4,
                    )

                    # 保存指标到tensorboard
                    for k, v in val_logs.items():
                        self.writer.add_scalar(k, v, steps)

                    self.print(f"Step {steps} | Validation metrics: {val_logs}")

            self.accelerator.wait_for_everyone()

        ################ save ckpt ###############
        if not (steps % self.save_ckpt_interval):
            if self.is_main_process:
                model_path = str(self.output_dir / f"dinovq_model.{steps}.pt")
                self.save_ckpt(model_path)

                if self.use_ema:
                    ema_state_dict = self.ema_dinovq_model.state_dict()
                    model_path = str(self.output_dir / f"dinovq_model.{steps}.ema.pt")
                    torch.save(ema_state_dict, model_path)
                self.print(f"{steps}: saving model to {str(self.output_dir)}")
            self.accelerator.wait_for_everyone()

        self.steps += 1
        return logs

    def evaluate(self):
        """
        Run full evaluation on validation set.
        """
        if not self.is_main_process:
            return

        self.dinovq_model.eval()
        unwrapped_model = self.accelerator.unwrap_model(self.dinovq_model)
        val_model = (
            self.ema_dinovq_model.ema_model
            if (self.use_ema and self.is_main_process)
            else unwrapped_model
        )
        val_model.eval()

        # Get LPIPS function
        # lpips_fn = (
        #     self.loss_fn.perceptual_loss
        #     if hasattr(self.loss_fn, "perceptual_loss")
        #     else None
        # )
        if hasattr(self.loss_fn, "module"):
            lpips_fn = (
                self.loss_fn.module.perceptual_loss
                if hasattr(self.loss_fn.module, "perceptual_loss")
                else self.loss_fn.perceptual_loss
            )
        else:
            lpips_fn = self.loss_fn.perceptual_loss

        total_metrics = {}
        num_batches = 0

        self.evaluator.reset_fid()
        self.print("Starting evaluation...")

        with torch.inference_mode():
            for batch_idx, (img, _) in enumerate(
                tqdm(self.val_dataloader, desc="Evaluating")
            ):
                if batch_idx >= 10:
                    break
                img = img.to(self.device)

                # Forward pass
                outputs = val_model(img, return_all=True)
                recon_imgs = outputs["reconstructed_imgs"]
                z = outputs["z"]
                r = outputs["r"]

                # Unnormalize
                img_unnorm = unnormalize(img).clamp(0, 1)
                recon_unnorm = unnormalize(recon_imgs).clamp(0, 1)

                # Update FID
                self.evaluator.update_fid(img_unnorm, recon_unnorm)

                # Compute metrics
                batch_metrics = {}

                # Energy Ratio (|r|^2 / |z|^2)
                r_energy = r.pow(2).mean()
                z_energy = z.pow(2).mean()
                energy_ratio = r_energy / (z_energy + 1e-8)
                batch_metrics["energy_ratio"] = energy_ratio.item()
                batch_metrics["r_energy"] = r_energy.item()

                # Image Quality
                iq_metrics = self.evaluator.compute_image_metrics(
                    img_unnorm, recon_unnorm, lpips_fn
                )
                batch_metrics.update(iq_metrics)

                # Frequency/Detail
                freq_metrics = self.evaluator.compute_freq_metrics(
                    img_unnorm, recon_unnorm
                )
                batch_metrics.update(freq_metrics)

                # Accumulate
                for k, v in batch_metrics.items():
                    total_metrics[k] = total_metrics.get(k, 0.0) + v

                num_batches += 1

                # Visualize first batch
                if batch_idx == 0:
                    self.visualize_image_reconstruction(
                        imgs=img,
                        recon_imgs=recon_imgs,
                        tag="eval/image_reconstruction",
                        steps=self.steps.item(),
                        max_samples=64,
                    )
                    self.visualize_disentanglement(
                        val_model=val_model,
                        img=img,
                        tag="eval/disentanglement",
                        steps=self.steps.item(),
                        max_samples=64,
                    )

        # Average metrics
        avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}

        # Compute FID
        self.print("Computing FID...")
        fid_score = self.evaluator.compute_fid()
        avg_metrics["fid"] = fid_score

        # Print results
        print("\n" + "=" * 30)
        print("Evaluation Results:")
        print("=" * 30)
        for k, v in avg_metrics.items():
            print(f"{k:<15}: {v:.4f}")
        print("=" * 30 + "\n")

        # Log to tensorboard
        for k, v in avg_metrics.items():
            self.writer.add_scalar(f"eval/{k}", v, self.steps.item())

        return avg_metrics

    def train(self, log_fn=None, config=None):
        if self.is_main_process:
            if config is not None:
                # Format config as a markdown table or code block
                config_str = (
                    "```json\n" + json.dumps(config, indent=2, default=str) + "\n```"
                )
                self.writer.add_text("config", config_str, 0)
            else:
                self.writer.add_text(
                    "config",
                    f"lr={self.lr}, batch_size={self.batch_size}, steps={self.num_train_steps}",
                    0,
                )

        progress_bar = None
        if self.is_main_process:
            progress_bar = tqdm(
                total=self.num_train_steps, desc="Training", dynamic_ncols=True
            )

        while self.steps < self.num_train_steps:
            logs = self.train_step()
            steps = int(self.steps.item())

            if self.is_main_process and progress_bar is not None:
                desc = f"Step {steps}/{self.num_train_steps}"
                if "loss" in logs:
                    desc += f" | loss: {logs['loss']:.4f}"
                if "dino_loss" in logs:
                    desc += f" | dino: {logs['dino_loss']:.4f}"
                if "recon_loss" in logs:
                    desc += f" | recon: {logs['recon_loss']:.4f}"
                if "disc_loss" in logs:
                    desc += f" | disc: {logs['disc_loss']:.4f}"
                progress_bar.set_description(desc)
                progress_bar.update(1)

            if log_fn is not None and callable(log_fn):
                log_fn(logs)

        self.print("training complete")
        if self.is_main_process:
            self.writer.close()
            progress_bar.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DinoVQ Residual VQ-VAE on ImageNet"
    )

    # Model args
    parser.add_argument(
        "--dinov3_ckpt",
        type=str,
        default="/high_perf_store2/users/zhuzeyu/ckpt/pretrained/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
        help="path to dinov3 checkpoint",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        nargs=2,
        default=[224, 224],
        help="input image size [H, W]",
    )
    parser.add_argument("--patch_size", type=int, default=16, help="patch size")
    parser.add_argument(
        "--encoder_embed_dim", type=int, default=768, help="encoder embedding dimension"
    )
    parser.add_argument("--encoder_depth", type=int, default=12, help="encoder depth")
    parser.add_argument(
        "--encoder_num_heads", type=int, default=12, help="encoder num heads"
    )
    parser.add_argument("--codebook_size", type=int, default=8192, help="codebook size")
    parser.add_argument(
        "--codebook_dim",
        type=int,
        default=None,
        help="codebook dimension (None=use encoder_dim)",
    )
    parser.add_argument(
        "--dino_dim", type=int, default=768, help="DINO feature dimension"
    )
    parser.add_argument(
        "--dino_decoder_depth", type=int, default=8, help="DINO decoder depth"
    )
    parser.add_argument(
        "--image_decoder_dim", type=int, default=768, help="image decoder dimension"
    )
    parser.add_argument(
        "--image_decoder_depth", type=int, default=8, help="image decoder depth"
    )

    # Loss weights
    parser.add_argument(
        "--dino_loss_weight", type=float, default=1.0, help="DINO loss weight"
    )
    parser.add_argument(
        "--recon_loss_weight",
        type=float,
        default=1.0,
        help="reconstruction loss weight",
    )
    parser.add_argument(
        "--vq_loss_weight", type=float, default=1.0, help="VQ loss weight"
    )

    # Perceptual & GAN Loss args
    parser.add_argument(
        "--use_perceptual", action="store_true", help="use perceptual loss"
    )
    parser.add_argument(
        "--perceptual_weight", type=float, default=1.0, help="perceptual loss weight"
    )
    parser.add_argument(
        "--vgg_ckpt_path",
        type=str,
        default="/high_perf_store2/users/zhuzeyu/ckpt/pretrained/vgg16-397923afv2.pth",
        help="path to vgg checkpoint",
    )

    parser.add_argument("--use_gan", action="store_true", help="use GAN loss")
    parser.add_argument(
        "--residual_reg_weight",
        type=float,
        default=0.0,
        help="residual L2 regularization weight",
    )

    parser.add_argument(
        "--disc_start", type=int, default=10000, help="step to start discriminator"
    )
    parser.add_argument(
        "--disc_weight", type=float, default=0.1, help="discriminator weight"
    )
    parser.add_argument("--disc_dim", type=int, default=64, help="discriminator dim")
    parser.add_argument(
        "--disc_num_layers", type=int, default=3, help="discriminator layers"
    )
    parser.add_argument(
        "--disc_adaptive_weight",
        action="store_true",
        help="use adaptive adversarial weight",
    )

    # Trainer args
    parser.add_argument(
        "--num_train_steps", type=int, default=100000, help="number of train steps"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument(
        "--dataloader_num_workers", type=int, default=8, help="dataloader workers"
    )
    parser.add_argument(
        "--dataloader_prefetch_factor", type=int, default=2, help="prefetch factor"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="gradient accumulation",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay")
    parser.add_argument(
        "--max_grad_norm", type=float, default=1.0, help="max grad norm"
    )
    parser.add_argument(
        "--save_results_interval", type=int, default=500, help="eval interval"
    )
    parser.add_argument(
        "--save_ckpt_interval", type=int, default=5000, help="save checkpoint interval"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./results/dinovq", help="output directory"
    )
    parser.add_argument(
        "--tensorboard_log_dir",
        type=str,
        default=None,
        help="tensorboard log directory",
    )
    parser.add_argument("--use_ema", action="store_true", help="use EMA")
    parser.add_argument(
        "--ema_update_after_step", type=int, default=1000, help="EMA update after step"
    )
    parser.add_argument(
        "--ema_update_every", type=int, default=1, help="EMA update interval"
    )

    # Dataset args
    parser.add_argument(
        "--data_path",
        type=str,
        default="/high_perf_store2/users/xiexuezhen/Imagenet/imagenet",
        help="path to ImageNet dataset",
    )

    parser.add_argument("--resume_from_checkpoint", type=str, help="path/to/checkpoint")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation only")
    args = parser.parse_args()

    if args.tensorboard_log_dir is None:
        env_tb_log_dir = os.getenv("TENSORBOARD_LOG_PATH")
        if env_tb_log_dir is not None:
            args.tensorboard_log_dir = env_tb_log_dir
        else:
            args.tensorboard_log_dir = os.path.join(args.output_dir, "tensorboard_logs")

    print(f"Saving tensorboard logs to {args.tensorboard_log_dir}")
    return args


def main():
    args = parse_args()

    # Save config
    config_save_path = os.path.join(args.output_dir, "train_config.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(config_save_path, "w") as f:
        json.dump(vars(args), f, indent=4)
    print(f"Saved config to {config_save_path}")

    # Load datasets
    train_dataset, val_dataset = build_imagenet(
        data_path=args.data_path,
        final_reso=tuple(args.img_size),
        norm_mean=[0.485, 0.456, 0.406],
        norm_std=[0.229, 0.224, 0.225],
    )

    # Load DINOv3 model
    dinov3_vitl16 = torch.hub.load(
        "/high_perf_store2/users/yaoziyang/public_code/dinov3",
        "dinov3_vitb16",
        source="local",
        weights=args.dinov3_ckpt,
    )

    # Create DinoVQ model
    dinovq_model = DinoVQModel(
        img_size=tuple(args.img_size),
        patch_size=args.patch_size,
        in_chans=3,
        out_chans=3,
        encoder_embed_dim=args.encoder_embed_dim,
        encoder_depth=args.encoder_depth,
        encoder_num_heads=args.encoder_num_heads,
        codebook_size=args.codebook_size,
        codebook_dim=args.codebook_dim,
        dino_dim=args.dino_dim,
        dino_decoder_depth=args.dino_decoder_depth,
        image_decoder_dim=args.image_decoder_dim,
        image_decoder_depth=args.image_decoder_depth,
    )
    dinovq_model.init_weights()
    count_params_b(dinovq_model, trainable_only=True, verbose=True)

    # Create Loss
    loss_fn = DinoVQLoss(
        dino_loss_weight=args.dino_loss_weight,
        recon_loss_weight=args.recon_loss_weight,
        vq_loss_weight=args.vq_loss_weight,
        # Perceptual & GAN
        use_perceptual=args.use_perceptual,
        perceptual_weight=args.perceptual_weight,
        vgg_ckpt_path=args.vgg_ckpt_path,
        use_gan=args.use_gan,
        disc_start=args.disc_start,
        disc_weight=args.disc_weight,
        disc_dim=args.disc_dim,
        disc_num_layers=args.disc_num_layers,
        disc_adaptive_weight=args.disc_adaptive_weight,
    )

    # Create trainer
    trainer = DinoVQTrainer(
        dinovq_model=dinovq_model,
        loss_fn=loss_fn,
        dino_vit=dinov3_vitl16,
        num_train_steps=args.num_train_steps,
        batch_size=args.batch_size,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        image_size=tuple(args.img_size),
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_prefetch_factor=args.dataloader_prefetch_factor,
        lr=args.lr,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        save_results_interval=args.save_results_interval,
        save_ckpt_interval=args.save_ckpt_interval,
        output_dir=args.output_dir,
        tensorboard_log_dir=args.tensorboard_log_dir,
        use_ema=args.use_ema,
        ema_update_after_step=args.ema_update_after_step,
        ema_update_every=args.ema_update_every,
        accelerate_kwargs=dict(),
        residual_reg_weight=args.residual_reg_weight,
    )

    # Load checkpoint if specified
    if args.resume_from_checkpoint:
        trainer.load_ckpt(args.resume_from_checkpoint)
        print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")

    if args.evaluate:
        trainer.evaluate()
        return

    trainer.train(config=vars(args))


if __name__ == "__main__":
    main()
