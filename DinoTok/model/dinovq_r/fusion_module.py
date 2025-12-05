# Fusion Module for Residual VQ-VAE
# Fuses quantized codes (e) and residual (r) for image decoder

import torch
import torch.nn as nn
from torch import Tensor


class ConcatFusion(nn.Module):
    """
    Fuses quantized codes (e) and residual (r) via concatenation + linear projection.
    
    Args:
        dim: Dimension of both e and r (should be same)
        output_dim: Output dimension (defaults to dim if None)
    """
    
    def __init__(self, dim: int, output_dim: int | None = None):
        super().__init__()
        
        self.dim = dim
        self.output_dim = output_dim or dim
        
        # Linear projection to fuse concatenated e and r
        self.fusion = nn.Linear(dim * 2, self.output_dim)
    
    def forward(self, e: Tensor, r: Tensor) -> Tensor:
        """
        Fuse quantized codes and residual.
        
        Args:
            e: Quantized codes, shape [B, L, dim]
            r: Residual, shape [B, L, dim]
        
        Returns:
            fused: Fused features, shape [B, L, output_dim]
        """
        # Concatenate along feature dimension
        concat = torch.cat([e, r], dim=-1)  # [B, L, dim*2]
        
        # Project to output dimension
        fused = self.fusion(concat)  # [B, L, output_dim]
        
        return fused
