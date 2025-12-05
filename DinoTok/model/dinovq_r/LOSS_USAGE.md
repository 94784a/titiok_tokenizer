# DinoVQ 可选Loss使用说明

## 支持的Loss类型

DinoVQ模型现在支持以下可选loss：

### 1. 基础Loss (必选)
- **DINO Loss**: 语义特征余弦相似度损失
- **Reconstruction Loss**: 图像重建MSE损失  
- **VQ Loss**: Vector Quantization commitment损失

### 2. Perceptual Loss (可选)
使用LPIPS (Learned Perceptual Image Patch Similarity) 增强图像质量

### 3. GAN Loss (可选)
使用PatchGAN discriminator进行对抗训练，提升图像真实感

---

## 使用方法

### 不使用可选loss (默认)

```python
from model.dinovq_r import DinoVQModel

model = DinoVQModel(
    img_size=(224, 224),
    encoder_embed_dim=768,
    codebook_size=8192,
    # 只使用基础loss
    dino_loss_weight=1.0,
    recon_loss_weight=1.0,
    vq_loss_weight=1.0,
)

# 训练时
losses = model.compute_loss(imgs, dino_features_gt)
total_loss = losses['total_loss']
total_loss.backward()
```

### 使用Perceptual Loss

```python
model = DinoVQModel(
    img_size=(224, 224),
    encoder_embed_dim=768,
    codebook_size=8192,
    # 启用perceptual loss
    use_perceptual=True,
    perceptual_weight=1.0,
    vgg_ckpt_path="/path/to/vgg16.pth",
)

# 训练时（同上，自动包含perceptual loss）
losses = model.compute_loss(imgs, dino_features_gt)
```

### 使用GAN Loss (需要分开更新)

```python
model = DinoVQModel(
    img_size=(224, 224),
    encoder_embed_dim=768,
    codebook_size=8192,
    # 启用GAN loss
    use_gan=True,
    disc_start=10000,  # 10000步后开始训练discriminator
    disc_weight=0.5,
    disc_dim=64,
    disc_num_layers=3,
    disc_adaptive_weight=True,  # 自适应权重
)

# 创建两个optimizer
from trainer.optimizer import get_optimizer

# Generator optimizer (包括encoder/decoder/quantizer)
gen_optim = get_optimizer(model.parameters(), lr=1e-4, wd=0.0)

# Discriminator optimizer (单独优化)
disc_optim = get_optimizer(
    model.loss_fn.discriminator.parameters(),
    lr=1e-4,
    wd=0.0
)

# === 训练时需要分开更新 ===
for step in range(num_steps):
    imgs, dino_gt = next(dataloader)
    
    # 1. 更新Generator
    # 冻结discriminator，避免gen_adv_loss梯度传播到disc
    for p in model.loss_fn.discriminator.parameters():
        p.requires_grad = False
    
    gen_losses = model.compute_loss(
        imgs, 
        dino_gt,
        update_type="generator",
        global_step=step,
    )
    
    gen_loss = gen_losses['total_loss']
    gen_loss.backward()
    gen_optim.step()
    gen_optim.zero_grad()
    
    # 2. 更新Discriminator
    # 解冻discriminator
    for p in model.loss_fn.discriminator.parameters():
        p.requires_grad = True
    
    disc_losses = model.compute_loss(
        imgs,
        dino_gt,
        update_type="discriminator",
        global_step=step,
    )
    
    disc_loss = disc_losses['disc_loss']
    disc_loss.backward()
    disc_optim.step()
    disc_optim.zero_grad()
```

---

## 训练脚本示例

参考`trainer/train_dinovq_imagenet.py`，添加GAN支持的完整训练循环：

```python
class DinoVQTrainer:
    def __init__(self, model, ...):
        self.model = model
        
        # 如果使用GAN，创建两个optimizer
        if model.use_gan:
            # Generator
            self.gen_optim = get_optimizer(
                model.parameters(), lr=lr, wd=weight_decay
            )
            # Discriminator  
            self.disc_optim = get_optimizer(
                model.loss_fn.discriminator.parameters(),
                lr=lr, wd=weight_decay
            )
        else:
            # 只有generator
            self.optim = get_optimizer(
                model.parameters(), lr=lr, wd=weight_decay
            )
    
    def train_step(self):
        steps = int(self.steps.item())
        
        if not self.model.use_gan:
            # === 无GAN: 标准训练 ===
            losses = self.model.compute_loss(imgs, dino_gt)
            total_loss = losses['total_loss']
            total_loss.backward()
            self.optim.step()
            self.optim.zero_grad()
        
        else:
            # === 有GAN: 分开更新 ===
            
            # 1. Generator更新
            for p in self.model.loss_fn.discriminator.parameters():
                p.requires_grad = False
            
            gen_losses = self.model.compute_loss(
                imgs, dino_gt,
                update_type="generator",
                global_step=steps,
            )
            
            gen_loss = gen_losses['total_loss']
            gen_loss.backward()
            self.gen_optim.step()
            self.gen_optim.zero_grad()
            
            # 2. Discriminator更新
            for p in self.model.loss_fn.discriminator.parameters():
                p.requires_grad = True
            
            disc_losses = self.model.compute_loss(
                imgs, dino_gt,
                update_type="discriminator",
                global_step=steps,
            )
            
            disc_loss = disc_losses['disc_loss']
            disc_loss.backward()
            self.disc_optim.step()
            self.disc_optim.zero_grad()
```

---

## Loss输出说明

### Generator update返回
```python
{
    'total_loss': 总损失,
    'dino_loss': DINO语义损失 (未加权),
    'recon_loss': 重建损失 (未加权),
    'vq_loss': 量化损失 (未加权),
    'perceptual_loss': 感知损失 (未加权, 如果启用),
    'gen_adv_loss': 生成对抗损失 (未加权, 如果启用),
    # 加权后的版本
    'dino_loss_weighted': ...,
    'recon_loss_weighted': ...,
    'vq_loss_weighted': ...,
    'perceptual_loss_weighted': ...,
    'gen_adv_loss_weighted': ...,
}
```

### Discriminator update返回
```python
{
    'disc_loss': 判别器损失,
    'disc_weight': 当前判别器权重,
    'logits_real': 真实图片logits均值,
    'logits_fake': 生成图片logits均值,
}
```

---

## 注意事项

1. **GAN训练必须分开更新**: Generator和Discriminator的梯度不能互相影响
2. **Discriminator warmup**: 使用`disc_start`参数延迟discriminator训练
3. **自适应权重**: 启用`disc_adaptive_weight`可以自动平衡重建loss和对抗loss
4. **Perceptual loss开销**: LPIPS会增加显存和计算时间
5. **Checkpoint保存**: 使用GAN时需要保存discriminator的state_dict

---

## 推荐配置

### 高质量图像生成
```python
use_perceptual=True,
perceptual_weight=1.0,
use_gan=True,
disc_start=10000,
disc_weight=0.5,
disc_adaptive_weight=True,
```

### 快速训练 (无GAN)
```python
use_perceptual=False,
use_gan=False,
```

### 语义优先
```python
dino_loss_weight=2.0,  # 增加DINO权重
recon_loss_weight=1.0,
vq_loss_weight=0.5,
```
