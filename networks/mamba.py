import torch
import torch.nn as nn
from mamba_ssm import Mamba


class BidirectionalMambaBlock(nn.Module):
    """单个双向 Mamba 块（带 LayerNorm 和 Dropout）"""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

        self.mamba_forward = Mamba(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
        )
        self.mamba_backward = Mamba(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
        )
        self.proj = nn.Linear(2 * d_model, d_model)

    def forward(self, x):
        # LayerNorm + residual
        x_norm = self.norm(x)

        # 正向
        y_f = self.mamba_forward(x_norm)
        # 反向
        x_rev = torch.flip(x_norm, dims=[1])
        y_b_rev = self.mamba_backward(x_rev)
        y_b = torch.flip(y_b_rev, dims=[1])

        # 融合
        y = torch.cat([y_f, y_b], dim=-1)
        y = self.proj(y)

        return x + y  # 残差连接


class MambaEncoder(nn.Module):
    def __init__(self, d_model, num_layers=4, expand=2, **mamba_kwargs):
        super().__init__()
        self.layers = nn.ModuleList([
            BidirectionalMambaBlock(
                d_model=d_model,
                expand=expand,
                **mamba_kwargs
            ) for _ in range(num_layers)
        ])

    def forward(self, x):
        """
        x: (B, L, D)
        returns: (B, L, D)
        """
        for layer in self.layers:
            x = layer(x)
        return x