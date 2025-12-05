import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

class RGBTokenDecoder(nn.Module):
    """
    输入:  x ∈ (B, width, grid, grid)
    输出:  rgb ∈ (B, 3, grid*patch_size, grid*patch_size)
    严格复现：
      Conv2d(width → 3*patch_size^2, k=1) → Rearrange → Conv2d(3→3, k=3, p=1)
    """
    def __init__(self, width: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.ffn = nn.Sequential(
            nn.Conv2d(width, patch_size * patch_size * 3, kernel_size=1, padding=0, bias=True),
            Rearrange('b (p1 p2 c) h w -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size),
        )
        self.conv_out = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ffn(x.contiguous())
        x = self.conv_out(x)
        return x




class ConvGNAct(nn.Module):
    """Conv -> GroupNorm -> GELU"""
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, groups=8):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.gn = nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch)
        self.act = nn.GELU()
    def forward(self, x):
        return self.act(self.gn(self.conv(x)))


class UpBlock(nn.Module):
    """可学习的2×上采样模块"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.refine = ConvGNAct(out_ch, out_ch)
    def forward(self, x):
        return self.refine(self.deconv(x))


class SemanticSegDecoder(nn.Module):
    """
    通用语义分割解码器：
    - 若干次 2× 可学习上采样
    - 最后按 scale_factor 双线性插值
    - 对 logits 插值，再 softmax
    """
    def __init__(self, in_ch, num_classes, mid_ch=256, num_ups=2, scale_factor=4.0):
        """
        scale_factor: 最终上采样倍数（例如 4 表示从1/4恢复到原图）
        """
        super().__init__()
        self.scale_factor = scale_factor

        # 多层 2× 可学习上采样
        blocks = []
        ch = in_ch
        for _ in range(num_ups):
            blocks.append(UpBlock(ch, mid_ch))
            ch = mid_ch
        self.up_blocks = nn.Sequential(*blocks)

        # 输出头
        self.head = nn.Sequential(
            ConvGNAct(mid_ch, mid_ch),
            nn.Conv2d(mid_ch, num_classes, kernel_size=1)
        )

    def forward(self, x):
        # 1 可学习上采样
        x = self.up_blocks(x)
        # 2 输出 logits
        logits_low = self.head(x)
        # 3 对 logits 进行双线性上采样（固定倍数）
        logits = F.interpolate(logits_low, scale_factor=self.scale_factor,
                               mode="bilinear", align_corners=False)
        # # 4 softmax 得到概率
        # probs = F.softmax(logits, dim=1)
        return logits


class DepthDecoder(nn.Module):
    """
    深度估计解码头：
    - 若干次 2× 可学习上采样
    - 最后按 scale_factor 双线性插值
    - 输出单通道连续深度
    """
    def __init__(self, in_ch, mid_ch=256, num_ups=2, scale_factor=4.0, activation="softplus"):
        super().__init__()
        self.scale_factor = scale_factor

        # 可学习上采样层
        blocks = []
        ch = in_ch
        for _ in range(num_ups):
            blocks.append(UpBlock(ch, mid_ch))
            ch = mid_ch
        self.up_blocks = nn.Sequential(*blocks)

        # 输出头
        self.head = nn.Sequential(
            ConvGNAct(mid_ch, mid_ch),
            nn.Conv2d(mid_ch, 1, kernel_size=1)
        )

        # 激活函数（约束深度范围）
        if activation == "sigmoid":
            self.act = nn.Sigmoid()
        elif activation == "relu":
            self.act = nn.ReLU()
        elif activation == "softplus":
            self.act = nn.Softplus(beta=1.0)   # 可替代 ReLU
        else:
            self.act = nn.Identity()

    def forward(self, x):
        # 1 可学习上采样
        x = self.up_blocks(x)
        # 2 得到单通道 logits
        depth_logits = self.head(x)
        # 3 固定倍率双线性插值
        depth_map = F.interpolate(depth_logits, scale_factor=self.scale_factor,
                                  mode="bilinear", align_corners=False)
        # 4 激活约束范围
        depth = self.act(depth_map)
        return depth
