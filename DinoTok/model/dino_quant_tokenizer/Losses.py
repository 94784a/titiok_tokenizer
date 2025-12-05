import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils.lpips import LPIPS

def _to_4d(t):
    """支持 [B,H,W] / [B,1,H,W] / [B,N,H,W] -> [B*C,1,H,W]"""
    if t.dim() == 3:
        t = t.unsqueeze(1)
    if t.dim() == 5:
        B, N, H, W = t.shape
        t = t.view(B*N, 1, H, W)
    elif t.dim() == 4:
        if t.size(1) != 1:
            # 把通道/视角并到 batch 维，便于统一处理
            B, C, H, W = t.shape
            t = t.view(B*C, 1, H, W)
    else:
        raise ValueError(f"Unsupported tensor shape: {t.shape}")
    return t

def _make_valid_mask(gt, max_depth=None):
    m = (gt > 0).float()
    if max_depth is not None:
        m = m * (gt <= float(max_depth)).float()
    return m

def _berhu(pred, target, mask, c=None, eps=1e-6):
    diff = (pred - target).abs()
    diff = diff * mask
    if mask.sum() < 1:
        return pred.sum()*0.0
    if c is None:
        c = 0.2 * diff.max().clamp_min(eps).detach()
    l1 = diff
    l2 = (diff**2 + c**2) / (2*c + eps)
    out = torch.where(diff <= c, l1, l2)
    return out.sum() / mask.sum().clamp_min(1.0)

def _spatial_grads(x):
    # x: [B,1,H,W]
    dx = x[..., :, 1:] - x[..., :, :-1]   # H x (W-1)
    dy = x[..., 1:, :] - x[..., :-1, :]   # (H-1) x W
    return dx, dy

def _grad_loss(pred, target, mask, log_space=True, levels=4, eps=1e-6):
    if mask.sum() < 1:
        return pred.sum()*0.0

    x = torch.log(pred.clamp_min(eps)) if log_space else pred
    y = torch.log(target.clamp_min(eps)) if log_space else target
    m = mask

    loss = 0.0
    for lvl in range(levels):
        dx_x, dy_x = _spatial_grads(x)
        dx_y, dy_y = _spatial_grads(y)

        mx = m[..., :, 1:] * m[..., :, :-1]
        my = m[..., 1:, :] * m[..., :-1, :]

        gx = (dx_x - dx_y).abs() * mx
        gy = (dy_x - dy_y).abs() * my

        denom_x = mx.sum().clamp_min(1.0)
        denom_y = my.sum().clamp_min(1.0)
        loss = loss + gx.sum()/denom_x + gy.sum()/denom_y

        if lvl < levels - 1:
            x = F.avg_pool2d(x, 2, 2, ceil_mode=True)
            y = F.avg_pool2d(y, 2, 2, ceil_mode=True)
            m = F.avg_pool2d(m, 2, 2, ceil_mode=True)
            m = (m > 0.999).float()

    return loss

def _silog(pred, target, mask, lam=0.85, scale=1.0, eps=1e-6):
    if mask.sum() < 1:
        return pred.sum()*0.0
    p = torch.log(pred.clamp_min(eps))
    t = torch.log(target.clamp_min(eps))
    g = (p - t) * mask
    n = mask.sum().clamp_min(1.0)
    mean_g = g.sum()/n
    mean_g2 = (g*g).sum()/n
    silog = torch.sqrt((mean_g2 - lam * mean_g * mean_g).clamp_min(0.0))
    return scale * silog

class MetricDepthLoss(nn.Module):
    """
    绝对深度综合损失：
      total = wb * BerHu + wg * Grad + ws * SILog
    用法：
      crit = MetricDepthLoss(wb=1.0, wg=0.5, ws=0.1, max_depth=80.)
      out = crit(pred, gt)  # 返回 dict(total, berhu, grad, silog)
    """
    def __init__(self,
                 wb: float = 1.0,
                 wg: float = 0.5,
                 ws: float = 0.0,
                 # BerHu
                 berhu_c: float | None = None,
                 # Grad
                 grad_log_space: bool = True,
                 grad_levels: int = 4,
                 # SILog
                 silog_lambda: float = 0.85,
                 silog_scale: float = 1.0,
                 # common
                 use_mask: bool = True,
                 max_depth: float | None = None,
                 eps: float = 1e-6):
        super().__init__()
        self.wb = wb
        self.wg = wg
        self.ws = ws
        self.berhu_c = berhu_c
        self.grad_log_space = grad_log_space
        self.grad_levels = grad_levels
        self.silog_lambda = silog_lambda
        self.silog_scale = silog_scale
        self.use_mask = use_mask
        self.max_depth = max_depth
        self.eps = eps

    def forward(self, pred_m, gt_m):
        """
        pred_m, gt_m: [B,H,W] 或 [B,1,H,W] 或 [B,N,H,W]（单位：米）
        返回：dict(total, berhu, grad, silog)
        """
        pred = _to_4d(pred_m)
        gt   = _to_4d(gt_m)

        if pred.shape != gt.shape:
            raise ValueError(f"pred {pred.shape} and gt {gt.shape} must match")

        mask = _make_valid_mask(gt, self.max_depth) if self.use_mask else torch.ones_like(gt)

        losses = {}
        # BerHu（主项）
        if self.wb > 0:
            losses['berhu'] = _berhu(pred, gt, mask, c=self.berhu_c, eps=self.eps)
        else:
            losses['berhu'] = pred.sum()*0.0

        # Grad（一致性）
        if self.wg > 0:
            losses['grad'] = _grad_loss(pred, gt, mask,
                                        log_space=self.grad_log_space,
                                        levels=self.grad_levels,
                                        eps=self.eps)
        else:
            losses['grad'] = pred.sum()*0.0

        # SILog（辅助）
        if self.ws > 0:
            losses['silog'] = _silog(pred, gt, mask,
                                     lam=self.silog_lambda,
                                     scale=self.silog_scale,
                                     eps=self.eps)
        else:
            losses['silog'] = pred.sum()*0.0

        total = self.wb*losses['berhu'] + self.wg*losses['grad'] + self.ws*losses['silog']
        losses['total'] = total
        return losses


class RGBLoss(nn.Module):
    def __init__(
        self,
        reconstruction_loss='l1',    
    ):
        super().__init__()

        # 选择重建损失类型
        if reconstruction_loss == "l1":
            self.rec_loss = F.l1_loss
        elif reconstruction_loss == "l2":
            self.rec_loss = F.mse_loss
        else:
            raise ValueError(f"Unknown rec loss type '{reconstruction_loss}'.")
        self.perceptual_loss = LPIPS()

    def forward(self, inputs, reconstructions,
                        perceptual_weight=1.0,
                        rec_weight=1.0,):
        """_summary_

        Args:
            inputs (0-1): [B 3 H W]
            reconstructions (0-1): [B 3 H W]

        Returns:
            dict: loss
        """
        inputs = inputs.clamp(0.0, 1.0)* 2.0 - 1.0
        reconstructions = reconstructions * 2.0 - 1.0
        rec_loss = self.rec_loss(reconstructions, inputs)
        p_loss = inputs.new_tensor(0.0)
        if perceptual_weight > 0:
            p_loss = self.perceptual_loss(reconstructions, inputs).mean()
        # 最终输出
        return {
            "rec_loss": rec_weight * rec_loss,
            "perceptual_loss": perceptual_weight * p_loss,
        }