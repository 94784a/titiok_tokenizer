"""
示例：使用不同的codebook_dim和encoder_dim

演示VectorQuantize的自动投影功能
"""

import torch
from model.dinovq_r import DinoVQModel


def example_same_dim():
    """示例1：codebook_dim = encoder_dim (默认，无投影)"""
    print("=" * 60)
    print("示例1：codebook_dim = encoder_dim (默认)")
    print("=" * 60)
    
    model = DinoVQModel(
        img_size=(224, 224),
        patch_size=16,
        encoder_embed_dim=768,
        codebook_size=8192,
        # codebook_dim=None 默认使用encoder_embed_dim=768
    )
    
    imgs = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        outputs = model(imgs)
    
    print(f"Encoder输出 z: {outputs['z'].shape}")
    print(f"量化后 e:     {outputs['e'].shape}")
    print(f"残差 r:        {outputs['r'].shape}")
    print("✓ 维度一致，无需投影\n")


def example_different_dim():
    """示例2：codebook_dim < encoder_dim (使用投影降维)"""
    print("=" * 60)
    print("示例2：codebook_dim < encoder_dim (降维)")
    print("=" * 60)
    
    model = DinoVQModel(
        img_size=(224, 224),
        patch_size=16,
        encoder_embed_dim=768,  # Encoder输出768维
        codebook_dim=256,       # Codebook只有256维 (节省内存)
        codebook_size=8192,
    )
    
    # 检查quantizer内部的投影层
    has_projection = not isinstance(model.quantizer.project_in, torch.nn.Identity)
    print(f"是否有投影层: {has_projection}")
    if has_projection:
        print(f"  project_in:  {model.quantizer.project_in}")
        print(f"  project_out: {model.quantizer.project_out}")
    
    imgs = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        outputs = model(imgs)
    
    print(f"\nEncoder输出 z: {outputs['z'].shape}  (768维)")
    print(f"量化后 e:     {outputs['e'].shape}  (自动投影回768维)")
    print(f"残差 r:        {outputs['r'].shape}  (768维)")
    
    print("\n内部流程:")
    print("  z [768] → project_in → [256] → 量化 → [256] → project_out → e [768]")
    print("  r = z - e (都是768维)")
    print("✓ 自动投影，维度匹配\n")


def example_larger_codebook_dim():
    """示例3：codebook_dim > encoder_dim (使用投影升维)"""
    print("=" * 60)
    print("示例3：codebook_dim > encoder_dim (升维)")
    print("=" * 60)
    
    model = DinoVQModel(
        img_size=(224, 224),
        patch_size=16,
        encoder_embed_dim=384,  # Encoder输出384维
        codebook_dim=768,       # Codebook使用更大的768维 (更大容量)
        codebook_size=16384,    # 更大的codebook
    )
    
    has_projection = not isinstance(model.quantizer.project_in, torch.nn.Identity)
    print(f"是否有投影层: {has_projection}")
    if has_projection:
        print(f"  project_in:  {model.quantizer.project_in}")
        print(f"  project_out: {model.quantizer.project_out}")
    
    imgs = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        outputs = model(imgs)
    
    print(f"\nEncoder输出 z: {outputs['z'].shape}  (384维)")
    print(f"量化后 e:     {outputs['e'].shape}  (自动投影回384维)")
    print(f"残差 r:        {outputs['r'].shape}  (384维)")
    
    print("\n内部流程:")
    print("  z [384] → project_in → [768] → 量化 → [768] → project_out → e [384]")
    print("✓ 升维后量化，然后降回原维度\n")


def check_parameters():
    """检查不同配置的参数量"""
    print("=" * 60)
    print("参数量对比")
    print("=" * 60)
    
    configs = [
        {"encoder_embed_dim": 768, "codebook_dim": None, "name": "默认(768→768)"},
        {"encoder_embed_dim": 768, "codebook_dim": 256, "name": "降维(768→256)"},
        {"encoder_embed_dim": 384, "codebook_dim": 768, "name": "升维(384→768)"},
    ]
    
    for config in configs:
        model = DinoVQModel(
            img_size=(224, 224),
            patch_size=16,
            encoder_embed_dim=config["encoder_embed_dim"],
            codebook_dim=config["codebook_dim"],
            codebook_size=8192,
            encoder_depth=6,  # 减少depth以快速测试
            dino_decoder_depth=4,
            image_decoder_depth=4,
        )
        
        # 计算总参数量
        total_params = sum(p.numel() for p in model.parameters())
        quantizer_params = sum(p.numel() for p in model.quantizer.parameters())
        
        print(f"\n{config['name']}:")
        print(f"  总参数: {total_params:,}")
        print(f"  量化器参数: {quantizer_params:,}")
        
        # 检查codebook大小
        if hasattr(model.quantizer, '_codebook'):
            codebook = model.quantizer._codebook.embed
            print(f"  Codebook形状: {codebook.shape}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("VectorQuantize 自动投影功能演示")
    print("=" * 60 + "\n")
    
    example_same_dim()
    example_different_dim()
    example_larger_codebook_dim()
    check_parameters()
    
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print("""
VectorQuantize会自动处理维度不匹配：
1. codebook_dim = encoder_dim: 无投影，直接量化
2. codebook_dim < encoder_dim: 降维量化 (节省codebook内存)
3. codebook_dim > encoder_dim: 升维量化 (更大的表示容量)

所有情况下，输出e都会保持和输入z相同的维度，
因此残差r = z - e总是有效的。

优势:
- 小codebook_dim: 节省内存，适合大codebook_size
- 大codebook_dim: 更强表示能力，适合复杂数据
    """)
