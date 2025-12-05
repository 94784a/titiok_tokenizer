import os
from typing import Dict, Tuple, Optional, Sequence, Literal, Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.init import trunc_normal_
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from .Heads import VitSegHead, VitDepthHead, VitRGBHead
from .Losses import RGBLoss
from .Dino_transformer import DinoTransformer, DinoV3BEncoder
import math

# =========================
# =  最简 distributed_compat
# =========================
_PROCESS_SUBGROUP: Optional[dist.ProcessGroup] = None  # 当前进程的子组；None 表示回退到 WORLD/非分布式


def is_distributed_enabled() -> bool:
    """是否已启用分布式（dist.init_process_group 已调用）。"""
    return dist.is_available() and dist.is_initialized()


def get_rank(group: Optional[dist.ProcessGroup] = None) -> int:
    """获取当前进程在指定组内的 rank；未启用分布式时返回 0。"""
    if not is_distributed_enabled():
        return 0
    return dist.get_rank(group=group)


def get_world_size(group: Optional[dist.ProcessGroup] = None) -> int:
    """获取指定组的 world size；未启用分布式时返回 1。"""
    if not is_distributed_enabled():
        return 1
    return dist.get_world_size(group=group)


def set_process_subgroup(group: Optional[dist.ProcessGroup]) -> None:
    """显式设置当前进程使用的子组句柄；传 None 则清除为未设置状态。"""
    global _PROCESS_SUBGROUP
    _PROCESS_SUBGROUP = group


def new_subgroup(ranks: Sequence[int]) -> dist.ProcessGroup:
    """
    创建一个子组并将其设为当前进程的子组（若当前 rank 属于该组）。
    所有进程都应以相同参数调用；不在 ranks 内的进程不会设置为当前子组。
    """
    global _PROCESS_SUBGROUP
    pg = dist.new_group(ranks=list(ranks))
    if get_rank(dist.group.WORLD) in ranks:
        _PROCESS_SUBGROUP = pg
    return pg


def new_subgroups(all_subgroup_ranks: Sequence[Sequence[int]]) -> None:
    """
    一次性创建多个子组；当前进程会挑选包含自己的那个作为“当前子组”。
    例：new_subgroups(((0,1), (2,3)))  # world_size=4 时将 0/1 放一组，2/3 另一组
    """
    global _PROCESS_SUBGROUP
    my_rank = get_rank(dist.group.WORLD) if is_distributed_enabled() else 0
    for ranks in all_subgroup_ranks:
        pg = dist.new_group(ranks=list(ranks))
        if my_rank in ranks:
            _PROCESS_SUBGROUP = pg


def get_process_subgroup() -> Optional[dist.ProcessGroup]:
    """
    返回当前进程应使用的通信组：
      - 若已设置子组，则返回该子组；
      - 否则在已初始化分布式时返回 WORLD；
      - 若未初始化分布式，返回 None（调用者可据此跳过通信）。
    """
    if _PROCESS_SUBGROUP is not None:
        return _PROCESS_SUBGROUP
    if is_distributed_enabled():
        return dist.group.WORLD
    return None


def get_subgroup_size() -> int:
    """
    返回当前通信组的进程数：
      - 子组存在 => 子组大小
      - 未设子组但分布式已启用 => WORLD 大小
      - 未启用分布式 => 1
    """
    return get_world_size(group=get_process_subgroup())


def get_subgroup_rank() -> int:
    """当前进程在子组内的 rank（未启用分布式时为 0）。"""
    return get_rank(group=get_process_subgroup())


def is_subgroup_main_process() -> bool:
    """当前进程是否为子组内的主进程（rank 0）。"""
    return get_subgroup_rank() == 0

def compute_entropy_loss(affinity, loss_type="softmax", temperature=0.01):
    flat_affinity = affinity.reshape(-1, affinity.shape[-1])
    flat_affinity /= temperature
    probs = F.softmax(flat_affinity, dim=-1)
    log_probs = F.log_softmax(flat_affinity, dim=-1)
    if loss_type == "softmax":
        target_probs = probs
    else:
        raise ValueError("Entropy loss {} not supported".format(loss_type))
    avg_probs = torch.mean(target_probs, dim=0)
    avg_entropy = - torch.sum(avg_probs * torch.log(avg_probs + 1e-5))
    sample_entropy = - torch.mean(torch.sum(target_probs * log_probs, dim=-1))
    loss = sample_entropy - avg_entropy
    return loss
    
class DinoSDRTokenizer(nn.Module):
    def __init__(self, 
                 num_patches: tuple = (15, 30),
                 embed_dim: int = 768,
                 mid_channels: int = 256,
                 codebook_size: int = 8912,
                 num_classes=40,
                 vit_depth=6,
                 calib_dim=16,
                 use_teacher = False,
                 ):
        super().__init__()
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.codebook_size = codebook_size
        
        self.dep_head=VitDepthHead(
            in_channels=embed_dim,
            mid_channels=mid_channels,
            encoder_depth=vit_depth,
            num_patches=num_patches,
            calib_dim=calib_dim,
        )
        self.seg_head=VitSegHead(
            in_channels=embed_dim, 
            num_classes = num_classes,
            mid_channels=mid_channels,
            encoder_depth=vit_depth,
            num_patches=num_patches,
        )
        
        self.rec_head=VitRGBHead(
            in_channels = embed_dim, 
            encoder_depth=vit_depth,
            num_patches=num_patches,
        )
        
        self.rgb_loss = RGBLoss()
        
        self.Dino_decoder = DinoTransformer(
                num_patches = num_patches,
                embed_dim = embed_dim,
                encoder_depth = vit_depth,
        )
        self.Dino_decoder.init_weights()
        
        self.dino_embedding = nn.Embedding(num_embeddings = codebook_size, embedding_dim=mid_channels)
        nn.init.uniform_(self.dino_embedding.weight, -1.0/math.sqrt(mid_channels), 1.0/math.sqrt(mid_channels))
        
        self.use_teacher = use_teacher
        if self.use_teacher:
            self.down_projector = nn.Identity()
            self.up_projector = nn.Identity()
        else:
            self.down_projector = nn.Linear(embed_dim, mid_channels, bias=True )
            self.up_projector = nn.Linear(mid_channels, embed_dim, bias=True )
        
        
        #stage flag
        self.perceptual_w=1.0
        self.rec_w=1.0
        self.decode_rgb = True
        self.decode_segmentation = True
        self.decode_depth = True

        
        self.hard_quant = True
        self.align_Dino = False
        
        self.temperature = 0.07
        self.entropy_temperature = 0.01
        
        self.vq_loss_w=1.0
        self.commit_loss_w=0.25
        self.entropy_loss_w=0.0

        self.dino_mse_w = 1.0
        self.dino_cos_w = 1.0
        
        
        

        

        
    def quant(self, z):
        # reshape z -> (batch, height, width, channel) and flatten
        B,L,C=z.shape
        z = z.view(B*L,C)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        z = F.normalize(z, p=2, dim=-1)
        embedding = F.normalize(self.dino_embedding.weight, p=2, dim=-1)
        d = -2 * z @ embedding.t() + 2.0   


        min_encoding_indices = torch.argmin(d, dim=1)
        with torch.no_grad():
            occ  = torch.bincount(min_encoding_indices.view(-1),  minlength=self.codebook_size).float()
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(occ,  op=dist.ReduceOp.SUM)
            code_book_usage  = (occ  > 0).float().mean()

        if self.hard_quant:
            logits = -d / max(self.temperature, 1e-8)
            probs = torch.softmax(logits, dim=1)            # (B*L, N)
            z_q = embedding[min_encoding_indices]        
            # compute loss for embedding
            vq_loss = torch.mean((z_q - z.detach()) ** 2) 
            commit_loss = torch.mean((z_q.detach() - z) ** 2) 
            # preserve gradients
            z_q = z + (z_q - z).detach()# will introduce z_q slight numerical deviation
        else:
            logits = -d / max(self.temperature, 1e-8)
            probs = torch.softmax(logits, dim=1)            # (B*L, N)
            z_q = probs @ embedding                          # (B*L, C)
            vq_loss = torch.mean((z_q - z) ** 2) 
            commit_loss = z_q.new_tensor(0.0)
            
        if self.use_teacher:
            entropy_loss = d.new_tensor(0.0)
        else:
            entropy_loss = compute_entropy_loss(-d,temperature=self.entropy_temperature)
        z_q = z_q.view(B, L, C)
        probs = probs.view(B, L, self.codebook_size)
        return z_q,probs,code_book_usage , vq_loss, commit_loss, entropy_loss
    
    def decode_all(self,x,calib,prefix='',gt_depth=None,gt_seg=None,target_imgs=None):
        B, N, H, W, C = x.shape
        zero_loss=torch.zeros((), device=x.device, dtype=x.dtype)
        loss_dict, pred_dict={}, {}
        if self.decode_depth:
            depth_pred_dict,depth_log_dict, depth_loss = self.dep_head(x, calib=calib, gt_depth=gt_depth, prefix=prefix)
            loss_dict[prefix+"token_depth_loss"] = depth_loss if depth_loss is not None else zero_loss
            pred_dict.update(depth_pred_dict)
        if self.decode_segmentation:
            seg_pred_dict, seg_log_dict, seg_loss = self.seg_head(x, gt_seg=gt_seg, prefix=prefix)
            loss_dict[prefix+"token_seg_loss"] = seg_loss if seg_loss is not None else zero_loss
            pred_dict.update(seg_pred_dict)

        if self.decode_rgb:
            rgb_pred = self.rec_head(
                x=x,
            )# [b n 3 h w]
            if target_imgs is not None:
                rgb_loss_dict = self.rgb_loss(
                    inputs=target_imgs.flatten(0,1),
                    reconstructions=rgb_pred.flatten(0,1),
                    perceptual_weight=self.perceptual_w,
                    rec_weight=self.rec_w,
                    )
                loss_dict.update(rgb_loss_dict)
                pred_dict.update({
                    prefix+'gt_imgs':target_imgs
                })
            pred_dict.update({
                prefix+'pred_imgs':rgb_pred,
            })
        return pred_dict, loss_dict

        

    def forward(self, x: Tensor, calib: Tensor, 
                cams = 1,
                gt_depth: Tensor=None, 
                gt_seg: Tensor=None, 
                target_imgs: Tensor=None,
                ):
        pred_dict, loss_dict = {}, {}
        B, L, C = x.shape
        H, W = self.num_patches
        assert L == self.num_patches[0] * self.num_patches[1]
        # x_norm =  F.normalize(x, p=2, dim=-1)
        Dino_target = x.clone()
        #-------------------------------------------------
        #               quantinization
        #-------------------------------------------------
        z = self.down_projector(x)
        #quant
        z_q,probs, code_book_usage , vq_loss, commit_loss, entropy_loss = self.quant(z)
        loss_dict.update(
            {   
                "top1_prob_mean":probs.max(dim=-1).values.mean(),
                "code_book_usage":code_book_usage,
                "vq_loss":self.vq_loss_w * vq_loss,
                "commit_loss":self.commit_loss_w * commit_loss,
                "entropy_loss":self.entropy_loss_w * entropy_loss,
            }
        )
        x = self.up_projector(z_q)
        #-------------------------------------------------
        #               Dino Reconstruction
        #-------------------------------------------------
        x=self.Dino_decoder(x)
        if self.align_Dino:
            Dino_mse_loss = F.mse_loss(Dino_target, x)
            Dino_cos_loss=1.0 - F.cosine_similarity(
                F.normalize(x, dim=-1), 
                F.normalize(Dino_target, dim=-1), 
                dim=-1
            ).mean()
            loss_dict.update(
                    {
                        "Dino_mse_loss":self.dino_mse_w * Dino_mse_loss,
                        "Dino_cos_loss":self.dino_cos_w * Dino_cos_loss ,
                    }
                )

        task_pred_dict, task_loss_dict = self.decode_all(
            x=x.view(B//cams,cams,H,W,C),
            prefix='',
            calib=calib,
            gt_depth=gt_depth,
            gt_seg=gt_seg,
            target_imgs=target_imgs,
        )
        loss_dict.update(task_loss_dict)
        pred_dict.update(task_pred_dict)
        
        

        return pred_dict, loss_dict
    @torch.no_grad()
    def update_hparams(self, cur_iter, cur_epoch,max_epochs=0,max_iters=0):
        e = int(cur_iter)
        T = max_iters
        p1 = int(0.3 * T)
        p2 = int(0.6 * T)

        if e < p1:
            # Stage 1: soft + DINO
            self.hard_quant = False
            self.align_Dino = True
 

            # 简单线性退火温度：0.5 -> 0.15
            progress = 0.0 if p1 <= 0 else min(max(float(e) / float(p1), 0.0), 1.0)
            self.temperature = 0.5 - (0.5 - 0.2) * (1.0 + math.cos(math.pi * progress)) * 0.5
            self.entropy_temperature = self.temperature + 0.1

            self.vq_loss_w = 1.0
            self.commit_loss_w = 0.0
            self.entropy_loss_w = 0.10

            self.dino_mse_w = 1.0
            self.dino_cos_w = 1.0
            self.decode_segmentation = False
            self.decode_depth = False
            
            self.perceptual_w=.0
            self.rec_w=.0
            self.decode_rgb = False
    
        elif e < p2:
            # Stage 2: hard + DINO
            self.hard_quant = True
            self.align_Dino = True

            self.temperature = 0.2
            self.entropy_temperature = self.temperature + 0.1

            self.vq_loss_w = 1.0
            self.commit_loss_w = 0.25
            self.entropy_loss_w = 0.05

            self.dino_mse_w = 1.0
            self.dino_cos_w = 1.0
            self.decode_segmentation = False
            self.decode_depth = False
            self.perceptual_w=.0
            self.rec_w=.0
            self.decode_rgb = False

        else:
            # Stage 3: hard + DINO + RGB
            self.hard_quant = True
            self.align_Dino = True

            self.temperature = 0.2
            self.entropy_temperature =self.temperature + 0.1

            self.vq_loss_w = 1.0
            self.commit_loss_w = 0.25
            self.entropy_loss_w = 0.02

            self.dino_mse_w = 0.5
            self.dino_cos_w = 0.5
            self.decode_segmentation = True
            self.decode_depth = True
            self.perceptual_w=0.5
            self.rec_w=0.5
            self.decode_rgb = True
    # feedforward only functions
    def quant_encode(self, x: Tensor
                ):
        """_summary_

        Args:
            x (Tensor): [B,L,C]

        Returns:
            z_q, probs: [B,L,C] [B,L,CODE]
        """
        z = self.down_projector(x)
        z_q,probs, _ , _, _, _ = self.quant(z)
        return z_q, probs
    def decode_zq(self, z_q: Tensor, calib: Tensor, 
                cams = 1,
                ):
        x = self.up_projector(z_q)
        x=self.Dino_decoder(x)
        B, L, C = x.shape
        H, W = self.num_patches
        task_pred_dict, task_loss_dict = self.decode_all(
            x=x.view(B//cams,cams,H,W,C),
            calib=calib,
        )
        return task_pred_dict
            
    def save_weights(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        ckpt = {
            "state_dict": self.state_dict(),
            "num_patches": self.num_patches,
            "embed_dim": self.embed_dim,
            "mid_channels": getattr(self, "mid_channels", None),
            "codebook_size": self.codebook_size,
        }
        torch.save(ckpt, path)

    def load_weights(self, path, strict=True):
        ckpt = torch.load(path, map_location='cpu')         # 总是先落到 CPU
        state = ckpt.get('state_dict', ckpt)
        self.load_state_dict(state, strict=strict)



class DinoSDRVQModel(nn.Module):
    def __init__(
        self,
        image_size=(int(400),int(800)),
        codebook_size=4096,
        token_dim = 384,
        head_path = "/high_perf_store2/users/yaoziyang/DinoTok/Micheckpoint_vq_sdr/t-20251031205206-pckwl/epoch_9_head.pth",
        **kwargs,
    ):
        super().__init__()
        REPO_DIR="/high_perf_store2/users/yaoziyang/public_code/dinov3"
        self.img_backbone = torch.hub.load(REPO_DIR, 'dinov3_vitb16', source='local', weights="/high_perf_store2/users/jiangyuncheng/dinov3_ckpt/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth")
        self.patch_size=16
        self.codebook_size = codebook_size
        self.image_size = image_size
        self.dinotok=DinoSDRTokenizer(
            num_patches=(image_size[0] // 16, image_size[1] // 16),
            embed_dim=768,  
            mid_channels = token_dim,
            codebook_size = codebook_size,
        )
        self.dinotok.load_weights(path=head_path)

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
    
    def quant_encode(self, img_inputs=None,):
        '''
        input img_inputs list(Tensor):
                0 imgs      [B,T,N,C,H,W]  
                1+ 其余项     ...
        output z_q, probs: probs[B*N,H*W,C] is soft distribution; z_q [B*N,H*W,CODE] is hard vector
        '''
        img_feats = self._extract_img_feat(img_inputs[0])
        B,T,cams,H,W,C=img_feats.shape
        assert H == self.image_size[0] and W == self.image_size[1]
        last_img_tokens = img_feats[:,-1,...].flatten(2, 3).flatten(0, 1) #input [B*CAMS,H*W,C]
        z_q, probs = self.dinotok.quant_encode(last_img_tokens) #[B,L,C] [B,L,N]
        return z_q, probs, cams
    
    def decode_zq(self,img_inputs,zq,cams=1):
        """_summary_

        Args:
            img_inputs list(Tensor):
                0 imgs      [B,T,N,C,H,W]   (unused)
                1 rots      [B,T,N,3,3]
                2 trans     [B,T,N,3]
                3 intrins   [B,T,N,3,3] 或 [B,N,3,3]
                4+ 其余项     ...
            zq (tensor): [B*cams,H*W,C]
            cams (int, optional): . Defaults to 1.
        """
        calib_T, _ = self._build_calib_tensor(img_inputs)
        pred_dict = self.dinotok.decode_zq(zq,calib=calib_T,cams=cams)
        return pred_dict
    
    def indice2zq(self, indices):
        return self.dinotok.dino_embedding[indices]
    
    def forward(
        self,
        img_inputs=None,
        img_metas=None, # for vis
        **kwargs,
    ):
        predicts = dict()
        img_feats = self._extract_img_feat(img_inputs[0])
        B,T,N,H,W,C=img_feats.shape
        last_img_tokens = img_feats[:,-1,...].flatten(2, 3).flatten(0, 1) #input [B*CAMS,H*W,C]
        
        calib_T, focals = self._build_calib_tensor(img_inputs)
        pred_dict, _ = self.dinotok(
            last_img_tokens.detach(),
            calib=calib_T,
            cams=N,
            )

        predicts.update(pred_dict)
        gt_dicts = {
            "img_metas":img_metas,
        }
        predicts.update(gt_dicts)
        return predicts
    



    def show_results(self, data_dict, pred_dict, work_dir='work_dirs/debug_vis/',prefix=''):
        '''
        负责为批次中的每个相机视图
        Layout (2x4):
        Row 1: Original Image         | Original Image         | Ground Truth Segmentation | Ground Truth Depth
        Row 2: Teacher Reconstruction | Student Reconstruction | Predicted Segmentation    | Predicted Depth
        '''
        # --- 1. Setup and Configuration ---
        seg_color_map_list = [
            [0, 0, 0, 255],    
            [243, 35, 232, 255], 
            [106, 142, 34, 255], 
            [152, 251, 151, 255],
            [187, 153, 152, 255], 
            [220, 220, 0, 255],  
            [249, 170, 31, 255], 
            [219, 112, 146, 255], 
            [200, 200, 200, 255], 
            [220, 19, 59, 255], 
            [255, 0, 0, 255], 
            [119, 12, 32, 255],  
            [0, 0, 255, 255], 
            [128, 192, 0, 255],  
            [0, 0, 142, 255], 
            [0, 0, 70, 255],  
            [0, 60, 100, 255],  
            [0, 80, 100, 255],  
            [70, 70, 70, 255], 
            [138, 0, 138, 255],  
            [70, 130, 180, 255], 
            [238, 18, 136, 255], 
            [255, 246, 143, 255], 
            [139, 69, 18, 255],  
            [255, 127, 80, 255],  
            [47, 79, 78, 255], 
            [0, 128, 0, 255], 
            [193, 0, 65, 255], 
            [0, 250, 153, 255], 
            [173, 255, 48, 255], 
            [0, 63, 192, 255], 
            [127, 0, 191, 255],
            [192, 128, 64, 255],  
            [0, 64, 64, 255], 
            [192, 0, 192, 255],  
            [128, 64, 64, 255],  
            [0, 192, 64, 255],   
            [0, 128, 192, 255],  
            [111, 134, 148, 255],
            [0, 255, 0, 255]] #other为绿色
        SEG_COLOR_MAP = np.array(seg_color_map_list, dtype=np.uint8)[:, :3]
    

        num_cameras = data_dict['img_inputs'][0].shape[2]
        

    
        # --- 2. Main Loop: Iterate through samples and their camera views ---
        for i, result_item in enumerate(pred_dict):
            sample_idx_base = result_item.get('sample_idx', f'batch_{i}')
    
            for cam_idx in range(num_cameras):
                cam_name = str(cam_idx)
                # Create a unique save path for each camera view of each sample.
                save_path = os.path.join(work_dir,str(data_dict['img_metas'].data[0][0]['scene_token']),str(data_dict['img_metas'].data[0][0]['frame_idx']), f"{prefix}_sample_{sample_idx_base}_{cam_name}.png")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                
                # # Create a 2x5 grid for the eight-panel figure.
                fig, axs = plt.subplots(
                    2, 5, figsize=(32, 12), dpi=150,
                    gridspec_kw={'width_ratios': [1, 1, 1, 1, 0.035]}  # 最后一列窄一点给colorbar
                )
                
                # --- 3. Data Preparation for the current view (cam_idx) ---
    
                # 3.1 Original Image (denormalized from ImageNet stats)
                ori_img_tensor = data_dict['img_inputs'][0][i, -1, cam_idx]
                mean = torch.tensor([0.485, 0.456, 0.406], device=ori_img_tensor.device).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225], device=ori_img_tensor.device).view(3, 1, 1)
                ori_img = (ori_img_tensor * std + mean).cpu().clamp(0, 1).permute(1, 2, 0).numpy()
                target_shape_hw = ori_img.shape[:2]
                
                ori_img_160_320=None
                if prefix+'gt_imgs' in result_item:
                    vq_in = result_item[prefix+'gt_imgs'][i, cam_idx].detach().cpu().float()
                    ori_img_160_320 = vq_in.clamp(0, 1).permute(1, 2, 0).numpy()
                        
                # 3.2 Reconstructed Image (denormalized from [-1, 1] range)
                recon_img=None
                if prefix+'pred_imgs' in result_item:
                    recon_img_tensor = result_item[prefix+'pred_imgs'][0, cam_idx].permute(1,2,0)    # (H,W,C), 已经是 [0,1] 浮点
                    recon_img = recon_img_tensor.detach().cpu().clamp(0, 1).numpy()
            
                
                # 3.4 Segmentation Maps
                gt_seg_map = result_item.get(prefix+'seg_rec_gt_label')[i, cam_idx].cpu().numpy().astype(np.uint8) if prefix+'seg_rec_gt_label' in result_item else None
                gt_seg_color = SEG_COLOR_MAP[gt_seg_map % len(SEG_COLOR_MAP)] if gt_seg_map is not None else None
                
                
                pred_seg_map = result_item.get(prefix+'seg_rec_pred_label')[0, cam_idx].cpu().numpy().astype(np.uint8) if prefix+'seg_rec_pred_label' in result_item else None
                pred_seg_color = SEG_COLOR_MAP[pred_seg_map % len(SEG_COLOR_MAP)]if pred_seg_map is not None else None
                
    
                # 3.5 Depth Maps
                gt_depth_map = result_item.get(prefix+'depth_rec_gt')[i, cam_idx].cpu().numpy() if prefix+'depth_rec_gt' in result_item else None
                pred_depth_map = result_item.get(prefix+'depth_rec_pred')[0, cam_idx].cpu().numpy() if prefix+'depth_rec_pred' in result_item else None

                # --- 4. Plotting all prepared data onto the grid ---
    
                # Column 0: Original Image vs. VQVAE_Reconstruction
                # axs[0, 0].imshow(ori_img_160_320)
                # axs[0, 0].set_title("Original VQInput")
                axs[0, 0].axis('off')
                axs[1, 0].axis('off')
                
                # Column 1: Original Image vs. Reconstruction
                axs[0, 1].imshow(ori_img_160_320)
                axs[0, 1].set_title("Original Input")
                if recon_img is not None:
                    axs[1, 1].imshow(recon_img); 
                    axs[1, 1].set_title("Reconstructed Input")
                else:
                    axs[1, 1].axis('off')
    
                # Column 2: Segmentation GT vs. Prediction
                if gt_seg_map is not None:
                    axs[0, 2].imshow(gt_seg_color)
                axs[0, 2].set_title("Ground Truth Segmentation")
                if pred_seg_map is not None:
                    axs[1, 2].imshow(pred_seg_color)
                axs[1, 2].set_title("Predicted Segmentation")
                
                # Column 3: Depth GT vs. Prediction
                if gt_depth_map is not None:
                    im_gt = axs[0, 3].imshow(gt_depth_map, cmap='magma', vmin=0, vmax=60)
                    fig.colorbar(im_gt, cax=axs[0, 4])
                axs[0, 3].set_title("Ground Truth Depth")
                
                if pred_depth_map is not None:
                    im_pred = axs[1, 3].imshow(pred_depth_map, cmap='magma', vmin=0, vmax=60)
                    fig.colorbar(im_pred, cax=axs[1, 4])
                axs[1, 3].set_title("Predicted Depth")
            
                # --- 5. Finalization and Saving ---
                for r in range(2):
                    for c in range(4):   # 只到 3
                        axs[r, c].axis('off')
                    
                # Adjust layout to make room for the suptitle and colorbars.
                plt.tight_layout(rect=[0, 0., 1, 0.96])
                plt.savefig(save_path)
                
                # === 额外生成一张“像素级拼接图”，不经过 Matplotlib，维持分辨率是 160x320 ===
                try:
                    # 用与画布一致的深度范围 & 颜色映射
                    norm = plt.Normalize(vmin=0, vmax=60)
                    cmap = plt.get_cmap('magma')

                    def depth_to_rgb(d):
                        # d: (H,W) float
                        return (cmap(norm(d))[..., :3] * 255).astype(np.uint8)

                    def to_uint8_rgb(x01):
                        # x01: (H,W,3) in [0,1]
                        return (np.clip(x01, 0, 1) * 255).astype(np.uint8)

                    # 统一 tile 尺寸（以 vq 输入为基准）
                    if ori_img_160_320 is not None:
                        H, W = ori_img_160_320.shape[:2]

                    blank = np.zeros((H, W, 3), np.uint8)

                    # row1: [原始VQ输入, 原始VQ输入(第二列), 分割GT, 深度GT]
                    t1 = to_uint8_rgb(ori_img_160_320) if ori_img_160_320 is not None else blank
  

                    t3 = SEG_COLOR_MAP[gt_seg_map % len(SEG_COLOR_MAP)].astype(np.uint8) if gt_seg_map is not None else blank

                    t4 = depth_to_rgb(gt_depth_map) if gt_depth_map is not None else blank

                    # row2: [VQVAE重建, ST重建, 分割Pred, 深度Pred]
   
                    t6 = to_uint8_rgb(recon_img)     if recon_img      is not None else blank

                    t7 = SEG_COLOR_MAP[pred_seg_map % len(SEG_COLOR_MAP)].astype(np.uint8) if pred_seg_map is not None else blank

                    t8 = depth_to_rgb(pred_depth_map) if pred_depth_map is not None else blank

                    row1 = np.concatenate([t1, t3, t4], axis=1)
                    row2 = np.concatenate([t6, t7, t8], axis=1)
                    grid = np.concatenate([row1, row2], axis=0)  # H*2, W*4, 3

                    save_path_px = os.path.splitext(save_path)[0] + "_px.png"
                    Image.fromarray(grid).save(save_path_px)
                except Exception as e:
                    print(f"[show_results] pixel-grid save failed: {e}")
                plt.close(fig) # Close the figure to free up memory.
        
        # A final print statement after the batch is processed might be useful.
        if pred_dict:
             print(f"Finished visualizing batch, saved {len(pred_dict) * num_cameras} images to {work_dir}")

    def _build_calib_tensor(self, img_inputs):
        """
        Args
        ----
        img_inputs : list(Tensor)
            - imgs      [B,T,N,C,H,W]   (用不到)
            - rots      [B,T,N,3,3]
            - trans     [B,T,N,3]
            - intrins   [B,T,N,3,3] 或 [B,N,3,3]
            - 其余项     ...

        Returns
        -------
        calib_T : Tensor  (B,N,16)
            [ fx, fy, cx, cy, R(9), t(3) ]
        focals  : Tensor  (B*N)   — 水平焦距 (fx) 展平成 1-D
        """
        # ----------------------------- 取当前帧 -----------------------------
        rots = img_inputs[1][:, -1]      # (B,N,3,3)
        trans = img_inputs[2][:, -1]      # (B,N,3)
        intrins = img_inputs[3]             # 可能是 (B,T,N,3,3) 也可能是 (B,N,3,3)
        if intrins.ndim == 5:                 # 带时间维
            intrins = intrins[:, -1]          # 取最后一帧 → (B,N,3,3)

        # ----------------------------- 拼 calib_T --------------------------
        fx = intrins[..., 0, 0]
        fy = intrins[..., 1, 1]
        cx = intrins[..., 0, 2]
        cy = intrins[..., 1, 2]

        calib_T = torch.cat([
            fx.unsqueeze(-1), fy.unsqueeze(-1),
            cx.unsqueeze(-1), cy.unsqueeze(-1),
            rots.flatten(start_dim=-2),            # 9
            trans                                  # 3
        ], dim=-1)                                 # → (B,N,16)

        focals = fx.flatten()                      # (B*N)

        return calib_T, focals

