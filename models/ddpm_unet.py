# models/ddpm_unet.py (1/3)
import math
import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    """
    Embedding sinusoidal tipo Transformer para el paso de tiempo t.
    t: (N,) en [0, T-1]
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb_factor = math.log(10000) / (half_dim - 1)
        exponents = torch.arange(half_dim, device=device) * -emb_factor
        t = t.float().unsqueeze(1)  # (N,1)
        sinusoid_inp = t * exponents.exp().unsqueeze(0)  # (N, half_dim)
        emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=1)
        return emb  # (N, dim)


class ResidualBlock(nn.Module):
    """
    Bloque residual con GroupNorm + SiLU y un MLP para inyectar el time embedding.
    """
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.time_mlp = nn.Linear(time_emb_dim, out_ch)

        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # x: (N, C, H, W), t_emb: (N, time_emb_dim)
        h = self.conv1(self.act1(self.norm1(x)))
        time_out = self.time_mlp(t_emb)  # (N, out_ch)
        h = h + time_out[:, :, None, None]
        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.shortcut(x)
