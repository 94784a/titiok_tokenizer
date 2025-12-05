import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from .Dino_transformer import DinoTransformer
from .CNN_decoder import DepthDecoder, SemanticSegDecoder, RGBTokenDecoder
from .Losses import MetricDepthLoss
class GradientLoss(nn.Module):
    """GradientLoss.

    Adapted from https://www.cs.cornell.edu/projects/megadepth/

    Args:
        valid_mask (bool): Whether filter invalid gt (gt > 0). Default: True.
        loss_weight (float): Weight of the loss. Default: 1.0.
        max_depth (int): When filtering invalid gt, set a max threshold. Default: None.
    """

    def __init__(self, valid_mask=True, loss_weight=1.0, max_depth=None, loss_name="loss_grad"):
        super(GradientLoss, self).__init__()
        self.valid_mask = valid_mask
        self.loss_weight = loss_weight
        self.max_depth = max_depth
        self.loss_name = loss_name

        self.eps = 0.001  # avoid grad explode

    def gradientloss(self, input, target):
        input_downscaled = [input] + [input[:: 2 * i, :: 2 * i] for i in range(1, 4)]
        target_downscaled = [target] + [target[:: 2 * i, :: 2 * i] for i in range(1, 4)]

        gradient_loss = 0
        for input, target in zip(input_downscaled, target_downscaled):
            if self.valid_mask:
                mask = target > 0
                if self.max_depth is not None:
                    mask = torch.logical_and(target > 0, target <= self.max_depth).float()
                N = torch.sum(mask)
            else:
                mask = torch.ones_like(target)
                N = input.numel()
            if mask.sum() == 0:
                return input.sum() * 0.0  # 返回一个与计算图相关的零值
            target = target * mask
            input = input * mask
            
            input_log = torch.log(input + self.eps)
            target_log = torch.log(target + self.eps)
            log_d_diff = input_log - target_log

            log_d_diff = torch.mul(log_d_diff, mask)

            v_gradient = torch.abs(log_d_diff[0:-2, :] - log_d_diff[2:, :])
            v_mask = torch.mul(mask[0:-2, :], mask[2:, :])
            v_gradient = torch.mul(v_gradient, v_mask)

            h_gradient = torch.abs(log_d_diff[:, 0:-2] - log_d_diff[:, 2:])
            h_mask = torch.mul(mask[:, 0:-2], mask[:, 2:])
            h_gradient = torch.mul(h_gradient, h_mask)

            gradient_loss += (torch.sum(h_gradient) + torch.sum(v_gradient)) / N

        return gradient_loss

    def forward(self, depth_pred, depth_gt):
        """Forward function."""

        gradient_loss = self.loss_weight * self.gradientloss(depth_pred, depth_gt)
        return gradient_loss


class SigLoss(nn.Module):
    """SigLoss.

        This follows `AdaBins <https://arxiv.org/abs/2011.14141>`_.

    Args:
        valid_mask (bool): Whether filter invalid gt (gt > 0). Default: True.
        loss_weight (float): Weight of the loss. Default: 1.0.
        max_depth (int): When filtering invalid gt, set a max threshold. Default: None.
        warm_up (bool): A simple warm up stage to help convergence. Default: False.
        warm_iter (int): The number of warm up stage. Default: 100.
    """

    def __init__(
        self, valid_mask=True, loss_weight=1.0, max_depth=None, warm_up=False, warm_iter=100, loss_name="sigloss"
    ):
        super(SigLoss, self).__init__()
        self.valid_mask = valid_mask
        self.loss_weight = loss_weight
        self.max_depth = max_depth
        self.loss_name = loss_name

        self.eps = 0.001  # avoid grad explode

        # HACK: a hack implementation for warmup sigloss
        self.warm_up = warm_up
        self.warm_iter = warm_iter
        self.warm_up_counter = 0

    def sigloss(self, input, target):
        if self.valid_mask:
            valid_mask = target > 0
            if self.max_depth is not None:
                valid_mask = torch.logical_and(target > 0, target <= self.max_depth)
            if valid_mask.sum() == 0:
                return input.sum() * 0.0  # 返回一个与计算图相关的零值

            input = input[valid_mask]
            target = target[valid_mask]
            

        if self.warm_up:
            if self.warm_up_counter < self.warm_iter:
                g = torch.log(input + self.eps) - torch.log(target + self.eps)
                g = 0.15 * torch.pow(torch.mean(g), 2)
                self.warm_up_counter += 1
                return torch.sqrt(g)

        g = torch.log(input + self.eps) - torch.log(target + self.eps)
        Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
        return torch.sqrt(Dg)

    def forward(self, depth_pred, depth_gt):
        """Forward function."""

        loss_depth = self.loss_weight * self.sigloss(depth_pred, depth_gt)
        return loss_depth



class L1DepthLoss(nn.Module):
    """L1 Loss for Depth.

    计算预测深度与真值之间的 L1 损失，适用于伪深度图监督，
    过滤无效深度值（gt <= 0）。

    Args:
        valid_mask (bool): 是否过滤无效 gt（gt > 0），默认 True。
        loss_weight (float): 损失权重，默认 1.0。
        max_depth (int, optional): 当过滤无效 gt 时的最大阈值，默认 None。
        loss_name (str): 损失名称，默认 "l1_loss"。
    """
    def __init__(self, valid_mask=True, loss_weight=1.0, max_depth=None, loss_name="l1_loss",mask_max_depth=False):
        super(L1DepthLoss, self).__init__()
        self.valid_mask = valid_mask
        self.loss_weight = loss_weight
        self.max_depth = max_depth
        self.loss_name = loss_name
        self.mask_max_depth=mask_max_depth

    def forward(self, depth_pred, depth_gt):
        if self.valid_mask:
            mask = depth_gt > 0
            if self.max_depth is not None and self.mask_max_depth:
                mask = torch.logical_and(mask, depth_gt <= self.max_depth)
            if mask.sum() == 0:
                return depth_gt.sum() * 0.0
            depth_pred = depth_pred[mask]
            depth_gt = depth_gt[mask]
        loss = torch.mean(torch.abs(depth_pred - depth_gt))
        return self.loss_weight * loss


class MultiStageUpsample(nn.Module):
    def __init__(self, in_channels, stages=(2,2,2), mode='bilinear'):
        super().__init__()
        self.blocks = nn.ModuleList()
        for factor in stages:
            self.blocks.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=factor, mode=mode, align_corners=False),
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                )
            )
    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class DepthSegHead(nn.Module):
    def __init__(self, in_channels, 
                 mid_channels=64,
                 depth_min=0.0, 
                 depth_max=60.0, 
                 calib_dim=16,
                 num_seg_classes=40,
                 stages=(2,2,2,2),
                 upsample_type='MultiStageUpsample',
                 l1_weight=1.0,
                 grad_weight=1.0,
                 sig_weight=1.0,
                 s_gain=0.1,
                 b_gain=0.1,
                 ):
        super().__init__()
        self.depth_min = depth_min            # 最小深度
        self.depth_max = depth_max            # 最大深度
        self.in_channels = in_channels        # 输入特征维度
        self.num_seg_classes = num_seg_classes
        self.mid_channels = mid_channels
        #------------- 卷积特征提取模块 ----------------
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 2 * mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, 2 * mid_channels),          # 32 组，128%32==0
            nn.ReLU(inplace=True),

            nn.Conv2d(2 * mid_channels, 2 * mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, 2 * mid_channels),          # 32 组，128%32==0
            nn.ReLU(inplace=True),

            nn.Conv2d(2 * mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(16, mid_channels), # 若 mid_channels 不是 32 的倍数，把 32 改成能整除的数（如 16/8/4）
            nn.ReLU(inplace=True),
        )


        #------------- 相机参数编码器----------------
        self.calib_encoder = nn.Sequential(
            nn.Linear(calib_dim, 2 * mid_channels),  # 输出 64(scale) + 64(bias)
            nn.ReLU(inplace=True),
            nn.Linear(2 * mid_channels, 2 * mid_channels),
        )
        nn.init.zeros_(self.calib_encoder[-1].weight)
        nn.init.zeros_(self.calib_encoder[-1].bias)
        self.s_gain = s_gain   # 可调 0.05~0.2
        self.b_gain = b_gain
        # -------------上采样模块----------------
        if upsample_type == 'MultiStageUpsample':
            self.depth_upsample = MultiStageUpsample(in_channels=mid_channels,stages=stages)
            self.seg_upsample = MultiStageUpsample(in_channels=mid_channels,stages=stages)
        else:
            raise ValueError(f"Unsupported upsample_type: {upsample_type}")
        # ----------------深度头----------------
        self.mlp_depth = nn.Sequential(
            nn.Linear(mid_channels, mid_channels, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, 1,   bias=True),
            nn.Softplus(beta=1.0)   # 可替代 ReLU
            # nn.ReLU(inplace=True),
        )
        
        # ----------------语义分割头----------------
        self.mlp_seg = nn.Sequential(
            nn.Linear(mid_channels, mid_channels, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, num_seg_classes,   bias=True),
        )
        
        # ---------------深度loss---------------
        self.l1_loss=L1DepthLoss(loss_weight=l1_weight)
        self.sig_loss=SigLoss(loss_weight=sig_weight,warm_up=True, warm_iter=100)
        self.grad_loss=GradientLoss(loss_weight=grad_weight )


    def forward(self, x, calib, gt_depth=None,gt_seg=None,prefix=''):
        
        #------------------patch级特征提取----------------
        # 1)x: [B, N, H, W, C] -> 合并 batch 和多视角维度
        
        B, N, H, W, C = x.shape
        x = x.contiguous().view(B * N, H, W, C).permute(0, 3, 1, 2).contiguous()  # [B*N, C, H, W]

        x_shared = self.conv_layers(x)  # 卷积提取特征 [B*N, 64, H, W]

        # 2)将 calib 映射成 scale 和 bias
        calib_feat = self.calib_encoder(calib.view(B * N, -1))
        s_raw, b_raw = calib_feat.chunk(2, dim=-1)
        scale = 1.0 + self.s_gain * torch.tanh(s_raw)              # [BN, mid]
        bias  =        self.b_gain * torch.tanh(b_raw)              # [BN, mid]
        scale = scale.view(B*N, self.mid_channels, 1, 1)
        bias  = bias .view(B*N, self.mid_channels, 1, 1)
        x = x_shared * scale + bias                       # 乘加调制

        # 3)上采样特征图（空间分辨率放大）
        x = self.depth_upsample(x)                       # [B*N, 64, aH, aW]

        aH, aW = x.shape[2], x.shape[3]

        
        #------------------深度预测----------------
        # 分类输出：映射为 num_bins 的深度分类预测
        reg_output = self.mlp_depth(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)                  # [B*N, 1, aH, aW]
        # return {},{"token_depth_loss":logits.sum()},None
        pred_depth = reg_output.view(B, N, aH, aW)# [B, N, aH, aW]

        pred_dict = {
            prefix+"depth_rec_pred": torch.clamp(pred_depth, max=self.depth_max).detach(),
        }
        loss_dict={}
        # 如果提供了 GT 深度和 mask，则计算 supervised loss
        if gt_depth is not None:
            gt_depth_mask=gt_depth>0
            # [B, N, H, W] -> [B*N, 1, H, W]
            gt_depth_resized = F.interpolate(gt_depth.view(B * N, 1, gt_depth.shape[-2], gt_depth.shape[-1]), size=(aH, aW), mode='nearest').view(B, N, aH, aW)
            gt_depth_mask_resized = F.interpolate(gt_depth_mask.view(B * N, 1,gt_depth.shape[-2], gt_depth.shape[-1]).float(), size=(aH, aW), mode='nearest').view(B, N, aH, aW)

            gt_depth_resized=torch.clamp(gt_depth_resized, max=self.depth_max)

            pred_dict.update({
                prefix+"depth_rec_gt":gt_depth_resized,
                prefix+"depth_rec_mask": gt_depth_mask_resized,
            })
            grad=self.grad_loss(pred_depth,gt_depth_resized)
            sig=self.sig_loss(pred_depth,gt_depth_resized)
            l1=self.l1_loss(pred_depth,gt_depth_resized)
            
            loss = grad + sig + l1
            
            loss_dict.update({prefix+"token_depth_loss":loss,
                              prefix+"depth_grad":grad,
                              prefix+"depth_sig":sig,
                              prefix+"depth_l1":l1,
                              })
            pred_dict.update({prefix+"token_depth_loss":loss,
                              prefix+"depth_grad":grad,
                              prefix+"depth_sig":sig,
                              prefix+"depth_l1":l1,})

        # ---------- 语义分割 ----------
        # 调用共享上采样特征 x：[B*N, 64, aH, aW]
        x = self.seg_upsample(x_shared)    
        seg_logits = self.mlp_seg(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)   # [B*N, num_seg_classes, aH, aW]
        seg_logits = seg_logits.view(B, N, -1, aH, aW)  # [B, N, C_seg, H, W]
        seg_pred = seg_logits.argmax(dim=2)  # [B, N, H, W]
        pred_dict.update({prefix+"seg_rec_pred_label": seg_pred,})
        #有真值则计算loss
        if gt_seg is not None:
            gt_seg_mask=gt_seg<self.num_seg_classes
            # 插值 GT 到预测尺寸
            gt_seg_resized = F.interpolate(
                gt_seg.view(B * N, 1, gt_seg.shape[-2], gt_seg.shape[-1]).float(),
                size=(aH, aW),
                mode='nearest'
            ).long().view(B * N,aH, aW)
            gt_seg_mask_resized = F.interpolate(
                gt_seg_mask.view(B * N, 1,gt_seg.shape[-2], gt_seg.shape[-1]).float(), 
                size=(aH, aW), 
                mode='nearest'
            ).view(B * N,aH, aW)
            
            # Cross entropy loss
            seg_loss = F.cross_entropy(
                seg_logits.view(B * N, -1, aH, aW),
                gt_seg_resized,
                reduction='none'  # 保留每个像素的 loss
            )  # [B*N, aH, aW]

            seg_loss = (seg_loss * gt_seg_mask_resized).sum() / (gt_seg_mask_resized.sum() + 1e-6)
            # 输出预测结果和 loss
            pred_dict.update({
                prefix+"seg_rec_gt_label": gt_seg_resized.view(B, N, aH, aW),
                prefix+"token_seg_loss": seg_loss ,
                prefix+"seg_rec_mask": gt_seg_mask_resized.view(B, N, aH, aW),
            })
            loss_dict.update({
                prefix+"token_seg_loss": seg_loss
            })


        return pred_dict, loss_dict,
    


class VitDepthHead(nn.Module):
    def __init__(self, 
                 in_channels, 
                 mid_channels=256,
                 encoder_depth=6,
                 num_patches=(27, 48),
                 calib_dim=16,
                 
                 depth_min=0.0, 
                 depth_max=60.0, 
                 l1_weight=1.0,
                 grad_weight=1.0,
                 sig_weight=1.0,
                 ):
        super().__init__()
        # ---------------深度loss---------------
        # self.l1_loss=L1DepthLoss(loss_weight=l1_weight)
        # self.sig_loss=SigLoss(loss_weight=sig_weight,warm_up=True, warm_iter=100)
        # self.grad_loss=GradientLoss(loss_weight=grad_weight )
        
        self.metric_loss = MetricDepthLoss(
                                wb=1.0,          # BerHu 主项权重
                                wg=0.5,          # 梯度一致性权重
                                ws=0.1,          # SILog 辅助（绝对深度时可设 0~0.2）
                                grad_log_space=True,
                                grad_levels=4,
                                max_depth=depth_max   # 只计 0<d<=80 的像素
                            )
        # ----------------h param---------------
        self.depth_min = depth_min            # 最小深度
        self.depth_max = depth_max            # 最大深度
        self.in_channels = in_channels        # 输入特征维度
        self.mid_channels = mid_channels
        
        #-----------------calib encoder----------
        self.calib_encoder = nn.Sequential(
            nn.Linear(calib_dim, in_channels),  # 输出 64(scale) + 64(bias)
            nn.GELU(),
            nn.Linear(in_channels, in_channels),
        )
        # 第一层：He/Kaiming 初始化（适配 GELU）
        nn.init.kaiming_normal_(self.calib_encoder[0].weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.calib_encoder[0].bias)

        # 最后一层：Zero Init —— 让初始 token ≈ 0，不扰动主干
        nn.init.zeros_(self.calib_encoder[2].weight)
        nn.init.zeros_(self.calib_encoder[2].bias)
        self.calib_norm = nn.LayerNorm(in_channels)
        
        #-----------------VIT dencoder------------
        self.vit_decoder = DinoTransformer(
                num_patches = num_patches,
                embed_dim = in_channels,
                encoder_depth = encoder_depth,
        )
        self.vit_decoder.init_weights()
        
        #-----------------CNN decoder-------------
        self.depth_decoder = DepthDecoder(
            in_ch=in_channels, 
            mid_ch=mid_channels,
            activation="sigmoid",
        )
    def forward(self,x, calib, gt_depth=None, prefix=''):
        #------------------patch级特征提取----------------
        # 1)x: [B, N, H, W, C] -> 合并 batch 和多视角维度
        
        B, N, H, W, C = x.shape
        x = x.contiguous().view(B * N, H * W, C).contiguous()  # [B*N, H*W, C]
        calib_feat = self.calib_encoder(calib.view(B * N, -1))        
        calib_token = self.calib_norm(calib_feat.unsqueeze(1))# [B*N, H*W, C]
        x = self.vit_decoder(x=x,additional_tokens=calib_token,num_patches = (H,W))# [B*N, H*W, C]
        x=x.view(B * N, H, W, C).permute(0,3,1,2).contiguous()
        pred=self.depth_decoder(x).contiguous().squeeze(1)# [B, aH, aW]
        
        aH, aW = pred.shape[1], pred.shape[2]
        pred_depth = self.depth_min + (self.depth_max - self.depth_min) * pred.view(B, N, aH, aW)# [B, N, aH, aW]

        pred_dict = {
            prefix+"depth_rec_pred": torch.clamp(pred_depth, max=self.depth_max).detach(),
        }
        # 如果提供了 GT 深度和 mask，则计算 supervised loss
        if gt_depth is not None:
            gt_depth_mask=gt_depth>0
            # [B, N, H, W] -> [B*N, 1, H, W]
            gt_depth_resized = F.interpolate(gt_depth.view(B * N, 1, gt_depth.shape[-2], gt_depth.shape[-1]), size=(aH, aW), mode='nearest').view(B, N, aH, aW)
            gt_depth_mask_resized = F.interpolate(gt_depth_mask.view(B * N, 1,gt_depth.shape[-2], gt_depth.shape[-1]).float(), size=(aH, aW), mode='nearest').view(B, N, aH, aW)

            gt_depth_resized=torch.clamp(gt_depth_resized, max=self.depth_max)

            pred_dict.update({
                prefix+"depth_rec_gt":gt_depth_resized,
                prefix+"depth_rec_mask": gt_depth_mask_resized,
            })
            # grad=self.grad_loss(pred_depth,gt_depth_resized)
            # sig=self.sig_loss(pred_depth,gt_depth_resized)
            # l1=self.l1_loss(pred_depth,gt_depth_resized)
            
            # loss = grad + sig + l1
            loss = self.metric_loss(pred_depth,gt_depth_resized)['total']
            loss_dict={
                    # prefix+"token_depth_loss":loss,
                    # prefix+"depth_grad":grad,
                    # prefix+"depth_sig":sig,
                    # prefix+"depth_l1":l1,
                        }
            pred_dict.update({prefix+"token_depth_loss":loss,
                            #   prefix+"depth_grad":grad,
                            #   prefix+"depth_sig":sig,
                            #   prefix+"depth_l1":l1,
                              })

            return pred_dict, loss_dict, loss
        return pred_dict, {}, None
    
    
    
    
class VitSegHead(nn.Module):
    def __init__(self, 
                 in_channels, 
                 num_classes = 40,
                 mid_channels=256,
                 encoder_depth=6,
                 num_patches=(27, 48),
                 ):
        super().__init__()
        # ----------------h param---------------
        self.in_channels = in_channels        # 输入特征维度
        self.mid_channels = mid_channels
        self.num_seg_classes = num_classes
        #-----------------VIT dencoder------------
        self.vit_decoder = DinoTransformer(
                num_patches = num_patches,
                embed_dim = in_channels,
                encoder_depth = encoder_depth,
        )
        self.vit_decoder.init_weights()
        
        #-----------------CNN decoder-------------
        self.seg_decoder = SemanticSegDecoder(
            in_ch=in_channels, 
            mid_ch=mid_channels,
            num_classes=num_classes,
        )
    def forward(self,x , gt_seg=None, prefix=''):
        #------------------patch级特征提取----------------        
        B, N, H, W, C = x.shape
        x = x.contiguous().view(B * N, H * W, C)  # [B*N, H*W, C]
        x = self.vit_decoder(x=x,num_patches = (H,W))# [B*N, H*W, C]
        x=x.view(B * N, H, W, C).permute(0,3,1,2).contiguous()
        seg_logits=self.seg_decoder(x).contiguous()# [B,classes ,aH, aW]
        aH, aW = seg_logits.shape[2], seg_logits.shape[3]
        
        seg_logits = seg_logits.view(B, N, -1, aH, aW)  # [B, N, classes, H, W]
        seg_pred = seg_logits.argmax(dim=2)  # [B, N, H, W]
        
        pred_dict = {prefix+"seg_rec_pred_label": seg_pred,}
        #有真值则计算loss
        if gt_seg is not None:
            gt_seg_mask=gt_seg<self.num_seg_classes
            # 插值 GT 到预测尺寸
            gt_seg_resized = F.interpolate(
                gt_seg.view(B * N, 1, gt_seg.shape[-2], gt_seg.shape[-1]).float(),
                size=(aH, aW),
                mode='nearest'
            ).long().view(B * N,aH, aW)
            gt_seg_mask_resized = F.interpolate(
                gt_seg_mask.view(B * N, 1,gt_seg.shape[-2], gt_seg.shape[-1]).float(), 
                size=(aH, aW), 
                mode='nearest'
            ).view(B * N,aH, aW)
            
            # Cross entropy loss
            seg_loss = F.cross_entropy(
                seg_logits.view(B * N, -1, aH, aW),
                gt_seg_resized,
                reduction='none'  # 保留每个像素的 loss
            )  # [B*N, aH, aW]

            seg_loss = (seg_loss * gt_seg_mask_resized).sum() / (gt_seg_mask_resized.sum() + 1e-6)
            # 输出预测结果和 loss
            pred_dict.update({
                prefix+"seg_rec_gt_label": gt_seg_resized.view(B, N, aH, aW),
                prefix+"token_seg_loss": seg_loss ,
                prefix+"seg_rec_mask": gt_seg_mask_resized.view(B, N, aH, aW),
            })
            loss_dict={
                # prefix+"token_seg_loss": seg_loss
            }
            return pred_dict, loss_dict, seg_loss
        return pred_dict, {}, None
    


    
class VitRGBHead(nn.Module):
    def __init__(self, 
                 in_channels, 
                 encoder_depth=6,
                 num_patches=(27, 48),
                 ):
        super().__init__()
        # ----------------h param---------------
        self.in_channels = in_channels        # 输入特征维度
        #-----------------VIT dencoder------------
        self.vit_decoder = DinoTransformer(
                num_patches = num_patches,
                embed_dim = in_channels,
                encoder_depth = encoder_depth,
        )
        self.vit_decoder.init_weights()
        
        #-----------------CNN decoder-------------
        self.rgb_decoder = RGBTokenDecoder(
             width = in_channels, 
             patch_size = 16,
        )
    def forward(self,x):
        """_summary_

        Args:
            x [B,N,H,W,C]

        Returns:
            RGB_pred # [B, N, 3, H, W]
        """
        #------------------patch级特征提取----------------        
        B, N, H, W, C = x.shape
        x = x.contiguous().view(B * N, H * W, C)  # [B*N, H*W, C]
        x = self.vit_decoder(x=x,num_patches = (H,W))# [B*N, H*W, C]
        x=x.view(B * N, H, W, C).permute(0,3,1,2).contiguous()
        RGB_pred=self.rgb_decoder(x).contiguous()# [B,3,aH, aW]
        aH, aW = RGB_pred.shape[2], RGB_pred.shape[3]
        
        RGB_pred = RGB_pred.view(B, N, -1, aH, aW)  # [B, N, 3, H, W]
        return RGB_pred