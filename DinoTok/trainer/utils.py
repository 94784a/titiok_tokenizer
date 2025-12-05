import io
import os
import sys
import torch
import torch.nn.functional as F
import importlib.util
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
import matplotlib.cm as cm
from matplotlib.axes._axes import Axes
from pathlib import Path
from PIL import Image
from torchvision.utils import make_grid, save_image


def load_py_path(name: str, path: str):
    # 1. 构造 Path 对象，并指定一个临时模块名
    module_path = Path(path)
    module_name = name

    # 2. 通过 importlib 构造 spec
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)

    # 3. 把模块注册到 sys.modules，然后执行加载
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    # 4. 从 module 中取出 train_pipeline_cfg
    return module







def cycle(dl):
    while True:
        for data in dl:
            yield data


def cycle_v2(dl):
    while True:
        yield from iter(dl)


def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.0)
        log[key] = old_value + new_value
    return log


def unnormalize(imgs, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    mean = torch.tensor(mean).view(1, -1, 1, 1).to(imgs.device)
    std = torch.tensor(std).view(1, -1, 1, 1).to(imgs.device)
    if imgs.ndim == 5:
        # video
        mean = mean.unsqueeze(-1)
        std = std.unsqueeze(-1)
    return imgs * std + mean


def img_normalize(imgs, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    mean = torch.tensor(mean).view(1, -1, 1, 1).to(imgs.device)
    std = torch.tensor(std).view(1, -1, 1, 1).to(imgs.device)
    if imgs.ndim == 5:
        # video
        mean = mean.unsqueeze(-1)
        std = std.unsqueeze(-1)
    return (imgs * 1.0 - mean) / std


def hist2d_samples(
    samples,
    ax: Optional[Axes] = None,
    bins: int = 200,
    x_scale: float = 5.0,
    y_scale: float = 5.0,
    percentile: int = 99,
    **kwargs,
):
    H, xedges, yedges = np.histogram2d(
        samples[:, 0],
        samples[:, 1],
        bins=bins,
        range=[[-x_scale, x_scale], [-y_scale, y_scale]],
    )

    # Determine color normalization based on the 99th percentile
    cmax = np.percentile(H, percentile)
    cmin = 0.0
    norm = cm.colors.Normalize(vmax=cmax, vmin=cmin)

    # Plot using imshow for more control
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    ax.imshow(H.T, extent=extent, origin="lower", norm=norm, **kwargs)


def plot_img_err_overlay_batch(
    imgs_4d: torch.Tensor,  # (B, 3, H, W)
    errs_3d: torch.Tensor,  # (B, H, W) —— 已上采样到图像分辨率
    writer,  # SummaryWriter
    steps: int,
    tag: str = "val/overlay_batch",
    mean=None,  # 如 ImageNet mean: [0.485, 0.456, 0.406]
    std=None,  # 如 ImageNet std : [0.229, 0.224, 0.225]
    cmap: str = "magma",
    alpha: float = 0.45,
    max_samples: int = None,  # 限制展示样本数（避免过长）
    per_sample_minmax: bool = True,  # True: 每张图单独做min-max；False: 全局min-max
    save_dir: str = None,
):
    """
    生成一个 Figure：按行堆叠多个样本；每行三列：原图 / 误差热图 / 叠加
    并一次性写入 TensorBoard。
    """
    assert imgs_4d.dim() == 4 and imgs_4d.size(1) == 3, "imgs_4d 应为 (B,3,H,W)"
    assert errs_3d.dim() == 3, "errs_3d 应为 (B,H,W)"
    B = imgs_4d.size(0)
    K = min(B, max_samples) if (max_samples is not None) else B

    # 反归一化
    imgs = imgs_4d.detach().float().cpu()
    if (mean is not None) and (std is not None):
        mean_t = torch.as_tensor(mean, dtype=imgs.dtype)[None, :, None, None]
        std_t = torch.as_tensor(std, dtype=imgs.dtype)[None, :, None, None]
        imgs = imgs * std_t + mean_t
    imgs = imgs.clamp(0, 1)

    # 误差归一化
    errs = errs_3d.detach().float().cpu()
    if per_sample_minmax:
        # 每张图单独 [0,1]
        e = errs.clone().view(B, -1)
        e_min = e.min(dim=1, keepdim=True).values
        e_max = e.max(dim=1, keepdim=True).values
        e_norm = ((e - e_min) / (e_max - e_min + 1e-12)).view_as(errs)
        errs = e_norm
    else:
        # 全局 [0,1]
        emin = float(errs.min())
        emax = float(errs.max())
        errs = (errs - emin) / (emax - emin + 1e-12)

    # 绘图：K行×3列
    fig_h = 3 * K
    fig = plt.figure(figsize=(9, fig_h))
    for i in range(K):
        img = imgs[i].permute(1, 2, 0).numpy()  # HWC
        err = errs[i].numpy()

        ax1 = plt.subplot(K, 3, i * 3 + 1)
        ax1.imshow(img)
        ax1.set_title(f"Image [{i}]")
        ax1.axis("off")

        ax2 = plt.subplot(K, 3, i * 3 + 2)
        hm = ax2.imshow(err, cmap=cmap)
        ax2.set_title("Patch MSE")
        ax2.axis("off")
        plt.colorbar(hm, ax=ax2, fraction=0.046, pad=0.04)

        ax3 = plt.subplot(K, 3, i * 3 + 3)
        ax3.imshow(img)
        ax3.imshow(err, cmap=cmap, alpha=alpha)
        ax3.set_title("Overlay")
        ax3.axis("off")

    fig.tight_layout()
    if save_dir is not None:
        plt.savefig(str(save_dir / f"{steps}.png"), dpi=300)
    writer.add_figure(tag, fig, global_step=steps)
    plt.close(fig)


def pca_tokens_to_rgb(tokens_BLC, hw: tuple[int, int]):
    """
    tokens_BLC: (B, L, D_dino)  已是首尺度 L=h0*w0 的 patch 特征（如先做过 adaptive_avg_pool）
    hw: (h0, w0)
    return: (B,3,h0,w0) in [0,1]
    """
    B, L, C = tokens_BLC.shape
    h0, w0 = hw
    assert L == h0 * w0

    # 每图独立中心化 + PCA（用 SVD）
    X = tokens_BLC.float()  # (B,L,C)
    X = X - X.mean(dim=1, keepdim=True)  # 逐图减均值

    # 用低秩 SVD 取前三主分量方向
    # torch.pca_lowrank 更快（1.12+），回退到SVD也行
    try:
        U, S, V = torch.pca_lowrank(X, q=3, center=False)  # V: (C,3)
        comps = X @ V  # (B,L,3)
    except Exception:
        # 逐图 SVD（更稳，但慢一点）
        comps_list = []
        for b in range(B):
            xb = X[b]  # (L,C)
            # SVD on covariance方向：对 xb 的转置做 SVD 也行
            # torch.linalg.svd 返回 U(CxC), S, Vh(CxC)
            # 这里用经济分解：先对 (C,L) 做SVD，取右奇异向量
            U_b, S_b, Vh_b = torch.linalg.svd(xb, full_matrices=False)
            V_b = U_b[:, :3]  # (C,3)
            comps_list.append(xb @ V_b)  # (L,3)
        comps = torch.stack(comps_list, dim=0)  # (B,L,3)

    # 线性拉伸到 [0,1]（逐图逐通道）
    comps = comps.reshape(B, h0, w0, 3).permute(0, 3, 1, 2).contiguous()  # (B,3,h0,w0)
    # 防极端值，可选做个小的百分位裁剪
    B_, C_, H_, W_ = comps.shape
    comps_flat = comps.view(B_, C_, -1)
    mins = comps_flat.min(dim=-1, keepdim=True).values
    maxs = comps_flat.max(dim=-1, keepdim=True).values
    rng = (maxs - mins).clamp_min(1e-6)
    comps_norm = ((comps_flat - mins) / rng).view(B_, C_, H_, W_).clamp(0, 1)
    return comps_norm


@torch.no_grad()
def pca_tokens_to_rgb_batchwise(
    tokens_BLC: torch.Tensor,
    hw: tuple[int, int],
    use_robust=True,
    q_low=0.01,
    q_high=0.99,
):
    """
    Batch-wise PCA → RGB 可视化（跨图统一主成分与缩放）
    Args:
      tokens_BLC: (B, L, C_dino)  首尺度 patch 特征 (L = h0*w0)
      hw: (h0, w0)
      use_robust: 是否用分位数做鲁棒缩放（避免极端值）
      q_low, q_high: 分位数范围
    Returns:
      rgb: (B, 3, h0, w0), in [0,1]
    """
    assert tokens_BLC.dim() == 3
    B, L, C = tokens_BLC.shape
    h0, w0 = hw
    assert L == h0 * w0, f"L={L} 与 h0*w0={h0*w0} 不一致"

    X = tokens_BLC.float()  # (B,L,C)
    X = X.reshape(B * L, C)  # (BL, C)

    # 全局中心化（跨 batch）
    mean = X.mean(dim=0, keepdim=True)  # (1, C)
    Xc = X - mean  # (BL, C)

    # 取前三主成分（优先 pca_lowrank，回退 SVD）
    try:
        # q=3：取前 3 个特征向量，V: (C,3)
        _, _, V = torch.pca_lowrank(Xc, q=3, center=False)
    except Exception:
        # 回退：对协方差做 SVD：cov = Xc^T Xc / (n-1)
        # 直接对 Xc 做 SVD 也可：Xc = U S V^T，取 V 的前三列
        U_, S_, Vh_ = torch.linalg.svd(Xc, full_matrices=False)  # Xc = U S V^T
        V = Vh_.T[:, :3]  # (C,3)

    # 统一方向（避免成分符号随机）：让每个成分在全局上的平均投影为正
    proj_mean = (Xc @ V).mean(dim=0)  # (3,)
    sign = torch.sign(
        torch.where(proj_mean == 0, torch.ones_like(proj_mean), proj_mean)
    )
    V = V * sign.view(1, 3)

    # 投影并还原到 (B, L, 3) → (B, 3, h0, w0)
    comps = (Xc @ V).reshape(B, L, 3)  # (B, L, 3)
    comps = comps.permute(0, 2, 1).reshape(B, 3, h0, w0).contiguous()

    # 跨 batch 的统一缩放：每个通道用全局 min-max 或分位数范围
    C_, H_, W_ = comps.shape[1:]
    flat = comps.view(B, 3, -1)  # (B,3,HW)

    if use_robust:
        flat_ch = flat.permute(1, 0, 2).reshape(3, -1)  # (3, B*HW)
        lo = torch.quantile(flat_ch, q_low, dim=1, keepdim=True).view(1, 3, 1)
        hi = torch.quantile(flat_ch, q_high, dim=1, keepdim=True).view(1, 3, 1)
    else:
        # 全局 min/max（跨 batch & 空间）
        lo = flat.amin(dim=2, keepdim=True).amin(dim=0, keepdim=True)  # (1,3,1)
        hi = flat.amax(dim=2, keepdim=True).amax(dim=0, keepdim=True)  # (1,3,1)

    scale = (hi - lo).clamp_min(1e-6)
    comps_norm = ((flat - lo) / scale).clamp(0, 1).view(B, 3, H_, W_)

    return comps_norm


def apply_black_mask_to_image(
    img_BCHW: torch.Tensor, h: int, w: int, mask_BL: torch.Tensor
) -> torch.Tensor:
    """
    将 img_BCHW ([B,3,H,W] 0..1) 在 (h,w) 网格下、mask=True 的 patch 置黑。
    """
    B, C, H, W = img_BCHW.shape
    assert C == 3, "only support RGB for viz"
    assert H % h == 0 and W % w == 0, "H/W must be divisible by h/w for block masking"

    ph, pw = H // h, W // w
    out = img_BCHW.clone()
    # mask_BL: [B, L=h*w]
    mask_BL = mask_BL.view(B, h, w)
    for bi in range(B):
        for yi in range(h):
            for xi in range(w):
                if mask_BL[bi, yi, xi]:
                    ys, ye = yi * ph, (yi + 1) * ph
                    xs, xe = xi * pw, (xi + 1) * pw
                    out[bi, :, ys:ye, xs:xe] = 0.0
    return out


def analyze_topk_mass(q: torch.Tensor, k_list=(1, 5, 10, 50, 100)):
    """
    q: [M, V] soft label 概率分布
    k_list: 想看的 top-k 范围

    return: dict, 每个 k 对应的累计概率均值
    """
    # [M,V] -> [M,V] 排序后
    q_sorted, _ = torch.sort(q, dim=-1, descending=True)  # 每行概率从大到小
    result = {}
    for k in k_list:
        mass = q_sorted[:, :k].sum(dim=-1)  # 每个样本 top-k 概率和 [M]
        result[f"top{k}_mean"] = mass.mean().item()
        result[f"top{k}_median"] = mass.median().item()
    return result


def soft_assign_probs(
    conti_MC: torch.Tensor,
    E_VC: torch.Tensor,
    tau: float,
    l2_norm: bool = False,
) -> torch.Tensor:
    """
    assign soft probs to each vq code.
    see "SoftVQ-VAE- Efficient 1-Dimensional Continuous Tokenizer"
    reference https://arxiv.org/abs/2412.10958

    Args:
        conti_MC (torch.Tensor): continuous vae encoder feature, [M, C]
        E_VC (torch.Tensor): codebook embedding, [V, C]
        tau (float): temperature

    Returns:
        q (torch.Tensor): softmax(-||x-c||^2 / tau),  [M, V]
    """
    assert tau > 0, "tau must be > 0"

    with torch.cuda.amp.autocast(enabled=False):
        x = conti_MC.float()
        E = E_VC.float()

        if l2_norm:
            x = F.normalize(x, p=2, dim=-1)
            E = F.normalize(E, p=2, dim=-1)
            sim = x @ E.t()  # [M, V], cosine similarity
            logits = (2.0 * sim) / float(tau)
            # cosine similarity is bounded, do not need the following code
            # logits = logits - logits.amax(dim=-1, keepdim=True)w
            q = torch.softmax(logits, dim=-1)
            return q

        x2 = (x**2).sum(dim=-1, keepdim=True)  # [M, 1]
        E2 = (E**2).sum(dim=-1, keepdim=False).unsqueeze(0)  # [1, V]
        # distance ||x - c||^2 = |x|^2 + |c|^2 - 2xc
        dot = x @ E.T  # [M, V]
        dist = x2 + E2 - 2.0 * dot  # [M, V]

        logits = -dist / float(tau)
        logits = logits - logits.amax(dim=-1, keepdim=True)

        q = torch.softmax(logits, dim=-1)
        return q


def render_action_strip_single(
    px: float,
    py: float,
    yaw: float,
    ranges: (
        dict | None
    ),  # {"pose_x": (min,max), "pose_y": (min,max), "yaw": (min,max)} 或 None
    width: int,
    height: int = 28,
) -> torch.Tensor:
    """
    生成一张 (3, height, width) 的 RGB strip。
    输入 px/py/yaw 已经归一化到 [0,1]。
    - 横向灰/彩色底条范围固定为 0..1
    - 若提供 ranges，则在 0 所在的位置（映射到 0..1）画一条深灰参考线（仅当 0 落在 [min,max] 内）
    - 用红色竖线 + 下三角标出当前取值
    颜色：
      pose_x -> 蓝色 #4C78A8
      pose_y -> 青绿 #72B7B2
      yaw    -> 橙色 #F58518
    返回 torch.Tensor，shape=[3, height, width]，值域[0,1]
    """
    # 归一化并夹紧
    vals_norm = [
        float(np.clip(px, 0.0, 1.0)),
        float(np.clip(py, 0.0, 1.0)),
        float(np.clip(yaw, 0.0, 1.0)),
    ]
    names = ["pose_x", "pose_y", "yaw"]
    bar_cols = {
        "pose_x": "#4C78A8",
        "pose_y": "#72B7B2",
        "yaw": "#F58518",
    }

    # Figure 尺寸（按像素反推英寸）
    fig_h = max(1.0, height / 100.0)
    fig_w = max(1.0, width / 100.0)
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(fig_w, fig_h), dpi=100)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for ax, name, vnorm in zip(axes, names, vals_norm):
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0, 1)

        # 彩色底条（0..1）
        ax.barh(0.5, width=1.0, left=0.0, height=0.6, color=bar_cols[name])

        # 0 参考线（若提供 ranges 且 0 在范围内）
        if ranges is not None and name in ranges:
            vmin, vmax = ranges[name]
            if vmax > vmin and (vmin <= 0.0 <= vmax):
                x_zero = (0.0 - vmin) / (vmax - vmin)
                x_zero = float(np.clip(x_zero, 0.0, 1.0))
                ax.axvline(x_zero, color="#444444", linewidth=1.0, alpha=0.85)

        # 红色取值标记：竖线 + 下三角
        ax.axvline(vnorm, color="red", linewidth=2.0)
        ax.plot([vnorm], [0.5], marker=(3, 0, -90), color="red", markersize=6)

        # 左侧标签（同底条主色，增强区分度）
        ax.text(
            0.0,
            0.85,
            name,
            fontsize=7,
            va="center",
            ha="left",
            color=bar_cols[name],
            alpha=0.95,
            fontweight="bold",
        )

        # 去坐标轴
        ax.axis("off")

    plt.subplots_adjust(hspace=0.15, top=0.98, bottom=0.02, left=0.02, right=0.98)

    # 渲染为 PNG 并读取为张量
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    # 严格 resize 到目标像素，防止 bbox 裁剪导致尺寸偏差
    img = img.resize((width, height), Image.BICUBIC)
    arr = np.array(img).astype(np.float32) / 255.0  # [H, W, 3]
    tensor = torch.from_numpy(arr).permute(2, 0, 1)  # -> [3, H, W]
    return tensor


def render_action_strip_batch(
    px_t: torch.Tensor,
    py_t: torch.Tensor,
    yaw_t: torch.Tensor,
    ranges: dict,
    width: int,
    height: int = 28,
) -> torch.Tensor:
    B = px_t.shape[0]
    out = []
    for b in range(B):
        strip = render_action_strip_single(
            float(px_t[b].item()),
            float(py_t[b].item()),
            float(yaw_t[b].item()),
            ranges=ranges,
            width=width,
            height=height,
        )
        out.append(strip.unsqueeze(0))
    return torch.cat(out, dim=0)  # [B, 3, H, W]


def count_params_b(
    model: torch.nn.Module, trainable_only: bool = True, verbose: bool = True
) -> float:
    """
    统计模型参数量（以 Billion 为单位）。
    Args:
        model: nn.Module
        trainable_only: True 仅统计 requires_grad=True 的参数；False 统计全部参数
        verbose: True 时打印可读信息
    Returns:
        params_b: float，参数量（B）
    """
    if trainable_only:
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        n = sum(p.numel() for p in model.parameters())
    params_b = n / 1e9
    if verbose:
        kind = "trainable" if trainable_only else "total"
        print(f"Parameters ({kind}): {n:,}  ({params_b:.3f} B)")
    return params_b


def _resize_CHW_or_BCHW(x: torch.Tensor, size, mode="bicubic", align_corners=False):
    if x.dim() == 3:
        return F.interpolate(
            x.unsqueeze(0),
            size=size,
            mode=mode,
            align_corners=align_corners,
            antialias=True,
        ).squeeze(0)
    elif x.dim() == 4:
        return F.interpolate(
            x, size=size, mode=mode, align_corners=align_corners, antialias=True
        )
    raise ValueError(f"Expected 3D or 4D tensor, got shape {tuple(x.shape)}")


def _to01(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().float()
    # 常见输出在 [-1,1]；也兼容已在 [0,1] 的情况
    if x.min().item() < -0.05 or x.max().item() > 1.05:
        x = (x.clamp(-1, 1) + 1.0) * 0.5
    return x.clamp(0, 1)


@torch.no_grad()
def render_sample_multiscale_timeline(
    sample_idx: int,
    img_BC_T1_HW: torch.Tensor,  # [B,3,T+1,H,W]
    idx_list: list[torch.Tensor],  # len S; 每个 [B,T+1,Ls]
    pred_imgs_per_scale: list[
        list[torch.Tensor]
    ],  # len S; 每个 len T; 元素 [B,3,hs,ws]
    recon_t0_per_scale: list[torch.Tensor],  # len S; 每个 [B,3,hs,ws]
    hws: list[tuple[int, int]],  # [(h0,w0),..., (hS-1,wS-1)]
    pose_x_BT: torch.Tensor,  # [B,T]
    pose_y_BT: torch.Tensor,  # [B,T]
    yaw_BT: torch.Tensor,  # [B,T]
    action_strip_fn,  # -> [B,3,Hstrip,W]
    action_ranges: dict,
    vae,
    out_path: str,
    ar_strip_height: int = 84,
    add_gt_vq_bottom_rows: bool = True,
    bottom_band_h: int = 8,
    bottom_border_w: int = 2,
    sep_w: int = 3,
):
    device = img_BC_T1_HW.device
    B, _, T1, Hraw, Wraw = img_BC_T1_HW.shape
    T = T1 - 1
    S = len(hws)

    # 最高尺度尺寸
    cell_H, cell_W = recon_t0_per_scale[-1][sample_idx].shape[-2:]
    grid_W = (T + 1) * cell_W
    grid_H = S * cell_H

    # ============ 顶部动作条 ============
    strip_b_CHW = torch.zeros(
        3, ar_strip_height, grid_W, dtype=torch.float32, device=device
    )
    for t in range(T):
        seg_BCHW = action_strip_fn(
            pose_x_BT[:, t].detach().cpu(),
            pose_y_BT[:, t].detach().cpu(),
            yaw_BT[:, t].detach().cpu(),
            ranges=action_ranges,
            width=cell_W,
            height=ar_strip_height,
        )
        seg_b_CHW = _to01(seg_BCHW[sample_idx]).to(
            strip_b_CHW.device, strip_b_CHW.dtype
        )
        x0 = (t + 1) * cell_W
        strip_b_CHW[:, :, x0 : x0 + cell_W] = seg_b_CHW

    # 竖向分隔线（仅画在 strip 上）
    for t in range(0, T + 1):
        x = t * cell_W
        x0 = max(0, x - sep_w // 2)
        x1 = min(grid_W, x0 + sep_w)
        strip_b_CHW[:, :, x0:x1] = 1.0  # 白色

    # ============ 中部多尺度网格（黑底；小图不 resize） ============
    grid_img = torch.zeros(3, grid_H, grid_W, dtype=torch.float32, device=device)

    # 第 0 列：t=0 重建
    for s in range(S):
        y0 = s * cell_H
        img0 = _to01(recon_t0_per_scale[s][sample_idx]).to(grid_img.dtype)
        h0, w0 = img0.shape[-2:]
        grid_img[:, y0 : y0 + h0, 0:w0] = img0

    # t=1..T 生成
    for s in range(S):
        y0 = s * cell_H
        for t in range(1, T + 1):
            x0 = t * cell_W
            img_pred = _to01(pred_imgs_per_scale[s][t - 1][sample_idx]).to(
                grid_img.dtype
            )
            hp, wp = img_pred.shape[-2:]
            grid_img[:, y0 : y0 + hp, x0 : x0 + wp] = img_pred

    # ============ 底部两行（按 cell 渲染，避免整行广播导致的发白） ============
    def _make_decorated_tile(img_CHW: torch.Tensor, band_rgb, border_rgb):
        """返回 [3,cell_H,cell_W]：先贴图到 tile，再在 tile 内画顶带和描边。"""
        tile = torch.zeros(3, cell_H, cell_W, dtype=torch.float32, device=device)
        img = _to01(img_CHW).to(tile.dtype)
        if img.shape[-2:] != (cell_H, cell_W):
            img = _resize_CHW_or_BCHW(
                img, size=(cell_H, cell_W), mode="bicubic", align_corners=False
            )
        tile[...] = img
        # 顶部色带
        band = torch.tensor(band_rgb, dtype=tile.dtype, device=tile.device).view(
            3, 1, 1
        )
        tile[:, :bottom_band_h, :] = band
        # 四周描边
        border = torch.tensor(border_rgb, dtype=tile.dtype, device=tile.device).view(
            3, 1, 1
        )
        bw = bottom_border_w
        if bw > 0:
            tile[:, :bw, :] = border
            tile[:, -bw:, :] = border
            tile[:, :, :bw] = border
            tile[:, :, -bw:] = border
        return tile

    if add_gt_vq_bottom_rows:
        finest_h, finest_w = hws[-1]

        # ——— GT 行（按 cell 生成） ———
        gt_row = torch.zeros(3, cell_H, grid_W, dtype=torch.float32, device=device)
        for t in range(T + 1):
            gt_t = _to01(img_BC_T1_HW[sample_idx, :, t])
            tile = _make_decorated_tile(
                gt_t, band_rgb=[0.0, 1.0, 0.0], border_rgb=[0.0, 1.0, 0.0]
            )  # 绿色
            gt_row[:, :, t * cell_W : (t + 1) * cell_W] = tile

        # ——— VQ 行（按 cell 生成） ———
        vq_row = torch.zeros(3, cell_H, grid_W, dtype=torch.float32, device=device)
        idx_finest_BT1L = idx_list[-1]
        for t in range(T + 1):
            if t == 0:
                vq_t = _to01(recon_t0_per_scale[-1][sample_idx])
            else:
                idx_t = idx_finest_BT1L[sample_idx, t]  # [Lfinest]
                vq_t = vae.forward_decode(
                    idx_t[None], [1, vae.Cvae, finest_h, finest_w]
                )[0]
                vq_t = _to01(vq_t)
            tile = _make_decorated_tile(
                vq_t, band_rgb=[1.0, 1.0, 0.0], border_rgb=[1.0, 1.0, 0.0]
            )  # 黄色
            vq_row[:, :, t * cell_W : (t + 1) * cell_W] = tile

        canvas = torch.cat([strip_b_CHW, grid_img, gt_row, vq_row], dim=1)
    else:
        canvas = torch.cat([strip_b_CHW, grid_img], dim=1)

    save_image(canvas.cpu().clamp(0, 1), out_path)


@torch.no_grad()
def render_action_compare_bars_batch(
    gt_px,
    gt_py,
    gt_yaw,  # [B] 连续值
    pr_px,
    pr_py,
    pr_yaw,  # [B] 连续值
    ranges: dict,
    width: int = 384,
    height: int = 84,
):
    """
    返回: Tensor [B, 3, H, W], RGB. 每行对应一个动作维度
    GT用绿色，pred用红色，条形长度按物理范围归一化
    """

    def _to_unit(x, lo, hi):
        # 连续值 -> [0,1] 归一化；输入可为 Tensor
        return ((x - lo) / max(1e-8, (hi - lo))).clamp(0, 1)

    B = gt_px.shape[0]
    H, W = int(height), int(width)
    img = torch.zeros(B, 3, H, W, dtype=torch.float32)  # 黑底

    rows = 3
    row_h = H // rows
    pad = max(2, row_h // 10)  # 行内边距
    bar_h = max(4, (row_h - 2 * pad) // 3)  # GT/Pred 两条 + 行间距
    gap = max(2, (row_h - 2 * pad) - 2 * bar_h)

    dims = [
        ("pose_x", gt_px, pr_px),
        ("pose_y", gt_py, pr_py),
        ("yaw", gt_yaw, pr_yaw),
    ]

    for r, (name, gt_v, pr_v) in enumerate(dims):
        lo, hi = ranges[name]
        span = max(1e-8, (hi - lo))
        # 归一化到 [0,1] -> 水平长度
        gt_u = _to_unit(gt_v, lo, hi)
        pr_u = _to_unit(pr_v, lo, hi)

        y0 = r * row_h + pad
        y1 = y0 + bar_h
        y2 = y1 + gap
        y3 = y2 + bar_h

        zero_inside = lo <= 0.0 <= hi
        if zero_inside:
            u0 = (0.0 - lo) / span
            x0 = int(round(pad + u0 * (W - 2 * pad)))  # 零位像素列
            # 画零位竖线（淡灰）
            x0a, x0b = max(pad, x0 - 1), min(W - pad, x0 + 1)
            img[:, :, (r * row_h) : ((r + 1) * row_h), x0a:x0b] = 0.35
        else:
            x0 = pad  # 退化为左锚

        # 画 GT（绿色）条
        x_end_gt = (gt_u * (W - 2 * pad)).round().long() + pad
        # 画 Pred（红色）条
        x_end_pr = (pr_u * (W - 2 * pad)).round().long() + pad
        # 以 x0 为锚绘制：右扩表示正、左扩表示负
        # 注意：当 zero 不在范围内时，x0==pad，行为与旧版一致
        for b in range(B):
            # GT（绿色）
            xe = int(x_end_gt[b].item())
            if zero_inside:
                xa, xb = (x0, xe) if xe >= x0 else (xe, x0)
            else:
                xa, xb = (pad, xe)
            if xb > xa:
                img[b, 1, y0:y1, xa:xb] = 0.85  # G channel

            # Pred（红色）
            xe = int(x_end_pr[b].item())
            if zero_inside:
                xa, xb = (x0, xe) if xe >= x0 else (xe, x0)
            else:
                xa, xb = (pad, xe)
            if xb > xa:
                img[b, 0, y2:y3, xa:xb] = 0.85  # R channel

        # 画分隔线（中灰）
        y_mid = r * row_h
        img[:, :, y_mid : y_mid + 1, :] = 0.25

    return img.clamp(0, 1)


@torch.no_grad()
def gather_and_grid_first_ranks(
    accelerator,
    panel_local: torch.Tensor,  # [T*B_local, 3, H, W]，本进程
    B_local: int,
    T: int,
    first_ranks: int = 5,  # 取前 K 个 rank（每个 rank 全部 B_local 列）
    ncols: int | None = None,  # 每行列数（默认 = K * B_local）
):
    """
    目标布局：从上到下 = 时间（近->远），从左到右 = 样本（列对齐）。
    步骤：
      1) gather: [T*B_total, 3, H, W]
      2) reshape: [world, T, B_local, 3, H, W]   （按 rank 分块）
      3) 选前 K 个 rank → permute 到 [T, K, B_local, 3, H, W]
      4) 合并列：by_t = [T, K*B_local, 3, H, W]
      5) 展平为 [T*K*B_local, 3, H, W]，用 nrow=K*B_local 生成“每步一行”的 grid
    """
    device = accelerator.device
    panel_local = panel_local.to(device)  # [T*B_local, 3, H, W]
    panel_g = accelerator.gather_for_metrics(panel_local)  # [T*B_total, 3, H, W]

    world = accelerator.num_processes
    C, H, W = int(panel_g.shape[1]), int(panel_g.shape[2]), int(panel_g.shape[3])

    # 安全检查
    B_total = B_local * world
    assert (
        panel_g.shape[0] == T * B_total
    ), f"shape mismatch: got {panel_g.shape[0]} vs T*B_total={T*B_total}"

    # [world, T, B_local, C, H, W] —— 关键：按 rank 分块再展开时间与样本
    panel_wtb = panel_g.view(world, T, B_local, C, H, W)

    # 仅保留前 K 个 rank
    K = min(first_ranks, world)
    panel_wtb = panel_wtb[:K]  # [K, T, B_local, C, H, W]

    # 置换到 [T, K, B_local, C, H, W]，再合并列为 [T, K*B_local, C, H, W]
    panel_tkb = panel_wtb.permute(1, 0, 2, 3, 4, 5).contiguous()
    by_t = panel_tkb.view(T, K * B_local, C, H, W)  # 每个时间步 K*B_local 列

    # 展平为 [T*K*B_local, C, H, W]，并按“每步一整行”排版
    flat = by_t.reshape(T * K * B_local, C, H, W).cpu()
    ncols = (K * B_local) if ncols is None else min(ncols, K * B_local)

    grid = make_grid(flat, nrow=ncols, padding=2)  # 行=时间步，列=样本
    return grid


# ------------------------ 工具：渲染 code indices ------------------------
@torch.no_grad()
def render_action_indices_grid_batch(
    indices: torch.Tensor, H_img: int, W_img: int, pad_px: int = 6
):
    """
    indices: [B, K] (long)
    输出: [B, 3, H_img, W_img]，每个样本一张图片，K 行格子（从上到下），格子里写 code index。
    兼容 Pillow 的 textbbox / getbbox / textsize。
    """
    from PIL import Image, ImageDraw, ImageFont

    assert indices.ndim == 2, "indices must be [B, K]"
    B, K = indices.shape

    # 每格高度 & 顶/底边距
    cell_h = max(12, H_img // max(1, K))
    panel_h = K * cell_h
    top_pad = max(0, (H_img - panel_h) // 2)
    bottom_pad = H_img - panel_h - top_pad

    # 字号
    font_size = max(10, int(cell_h * 0.6))
    try:
        font = ImageFont.truetype("DejaVuSansMono.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    def _text_wh(draw: ImageDraw.ImageDraw, text: str):
        # 1) Pillow >= 8.0 推荐：textbbox
        if hasattr(draw, "textbbox"):
            # bbox = (l, t, r, b)
            l, t, r, b = draw.textbbox((0, 0), text, font=font)
            return r - l, b - t
        # 2) 字体对象的 getbbox
        if hasattr(font, "getbbox"):
            l, t, r, b = font.getbbox(text)
            return r - l, b - t
        # 3) 旧版：textsize
        if hasattr(draw, "textsize"):
            w, h = draw.textsize(text, font=font)
            return w, h
        # 4) 兜底估计
        return int(len(text) * font_size * 0.6), font_size

    out = []
    for b in range(B):
        img = Image.new("RGB", (W_img, H_img), (0, 0, 0))
        draw = ImageDraw.Draw(img)

        for k in range(K):
            y0 = top_pad + k * cell_h
            y1 = y0 + cell_h - 1
            x0 = pad_px
            x1 = W_img - pad_px

            # 交替深灰底
            shade = 38 if (k % 2 == 0) else 55
            draw.rectangle([x0, y0, x1, y1], fill=(shade, shade, shade))

            text = str(int(indices[b, k].item()))
            tw, th = _text_wh(draw, text)

            # 左上对齐＋垂直居中
            tx = x0 + 8
            ty = y0 + max(0, (cell_h - th) // 2)
            draw.text((tx, ty), text, fill=(240, 240, 240), font=font)

        # 外边框
        draw.rectangle([0, 0, W_img - 1, H_img - 1], outline=(100, 100, 100))

        arr = (
            torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        )  # [3,H,W]
        out.append(arr)

    return torch.stack(out, dim=0)  # [B,3,H_img,W_img]
