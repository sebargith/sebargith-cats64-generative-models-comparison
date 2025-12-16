# models/ddpm_unet.py
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


class DownBlock(nn.Module):
    """
    Dos bloques residuales + Downsample por Conv(stride=2).
    Devuelve (downsampled, skip) donde skip es la feature antes de hacer downsample.
    """
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int):
        super().__init__()
        self.res1 = ResidualBlock(in_ch, out_ch, time_emb_dim)
        self.res2 = ResidualBlock(out_ch, out_ch, time_emb_dim)
        self.down = nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor):
        x = self.res1(x, t_emb)
        x = self.res2(x, t_emb)
        down = self.down(x)
        return down, x  # down: resolución /2, x: skip connection


class UpBlock(nn.Module):
    """
    Upsample por ConvTranspose2d + concat con skip + dos ResBlocks.
    """
    def __init__(self, in_ch: int, out_ch: int, skip_ch: int, time_emb_dim: int):
        super().__init__()
        # Primero upsample de (H,W) -> (2H,2W)
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
        # Luego concatenamos skip y pasamos por ResBlocks
        self.res1 = ResidualBlock(out_ch + skip_ch, out_ch, time_emb_dim)
        self.res2 = ResidualBlock(out_ch, out_ch, time_emb_dim)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.res1(x, t_emb)
        x = self.res2(x, t_emb)
        return x


class UNetDDPM(nn.Module):
    """
    U-Net ligera para DDPM en imágenes 64x64x3.
    Predice el ruido ε dado x_t y t.
    """
    def __init__(self, img_channels: int = 3, base_channels: int = 64, time_emb_dim: int = 256):
        super().__init__()

        # Embedding de tiempo
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        # Primera conv
        self.init_conv = nn.Conv2d(img_channels, base_channels, 3, padding=1)

        # Down: 64x64 -> 32 -> 16 -> 8
        self.down1 = DownBlock(base_channels, base_channels, time_emb_dim)           # 3 -> 64, 64x64 -> 32x32
        self.down2 = DownBlock(base_channels, base_channels * 2, time_emb_dim)       # 64 -> 128, 32x32 -> 16x16
        self.down3 = DownBlock(base_channels * 2, base_channels * 4, time_emb_dim)   # 128 -> 256, 16x16 -> 8x8

        # Bottleneck
        self.bot1 = ResidualBlock(base_channels * 4, base_channels * 4, time_emb_dim)
        self.bot2 = ResidualBlock(base_channels * 4, base_channels * 4, time_emb_dim)

        # Up: 8x8 -> 16 -> 32 -> 64
        self.up3 = UpBlock(
            in_ch=base_channels * 4,
            out_ch=base_channels * 4,
            skip_ch=base_channels * 4,
            time_emb_dim=time_emb_dim,
        )  # concat con skip3 (256 canales)

        self.up2 = UpBlock(
            in_ch=base_channels * 4,
            out_ch=base_channels * 2,
            skip_ch=base_channels * 2,
            time_emb_dim=time_emb_dim,
        )  # concat con skip2 (128 canales)

        self.up1 = UpBlock(
            in_ch=base_channels * 2,
            out_ch=base_channels,
            skip_ch=base_channels,
            time_emb_dim=time_emb_dim,
        )  # concat con skip1 (64 canales)

        # Salida
        self.final_norm = nn.GroupNorm(8, base_channels)
        self.final_act = nn.SiLU()
        self.final_conv = nn.Conv2d(base_channels, img_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x: (N, 3, 64, 64) en [-1, 1]
        t: (N,) long/int, pasos de 0 a T-1
        """
        t_emb = self.time_mlp(t)  # (N, time_emb_dim)

        x = self.init_conv(x)
        d1, s1 = self.down1(x, t_emb)   # d1: 32x32, s1: 64x64
        d2, s2 = self.down2(d1, t_emb)  # d2: 16x16, s2: 32x32
        d3, s3 = self.down3(d2, t_emb)  # d3: 8x8,  s3: 16x16

        b = self.bot1(d3, t_emb)
        b = self.bot2(b, t_emb)

        u3 = self.up3(b, s3, t_emb)     # 8 -> 16
        u2 = self.up2(u3, s2, t_emb)    # 16 -> 32
        u1 = self.up1(u2, s1, t_emb)    # 32 -> 64

        out = self.final_conv(self.final_act(self.final_norm(u1)))
        # out ≈ ruido ε predicho, misma forma que x
        return out
