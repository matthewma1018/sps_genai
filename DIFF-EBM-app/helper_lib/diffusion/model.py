import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Time embedding (sinusoidal) ----
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: (B,) float or int, will be cast to float
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            torch.arange(half, device=device, dtype=torch.float32)
            * -(math.log(10000.0) / (half - 1))
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # (B, dim)
        return emb

# ---- Basic building blocks ----
def conv3(in_ch, out_ch, s=1):
    return nn.Conv2d(in_ch, out_ch, 3, stride=s, padding=1)

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = conv3(in_ch, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = conv3(out_ch, out_ch)
        # time -> scale/shift (FiLM)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_ch * 2)
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.time_mlp(t_emb).chunk(2, dim=1)   # (B, C)
        h = h + scale.unsqueeze(-1).unsqueeze(-1) * 0 + shift.unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)

class Down(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 4, stride=2, padding=1)  # 2x down

    def forward(self, x):
        return self.op(x)

class Up(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1)  # 2x up

    def forward(self, x):
        return self.op(x)

class UNet(nn.Module):
    def __init__(self, in_ch=3, base=64, time_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Encoder
        self.in_conv = conv3(in_ch, base)
        self.rb1 = ResBlock(base, base, time_dim)
        self.down1 = Down(base)                 # 32 -> 16
        self.rb2 = ResBlock(base, base*2, time_dim)
        self.down2 = Down(base*2)               # 16 -> 8
        self.rb3 = ResBlock(base*2, base*4, time_dim)
        self.down3 = Down(base*4)               # 8 -> 4

        # Bottleneck
        self.rb_mid = ResBlock(base*4, base*4, time_dim)

        # Decoder
        self.up3 = Up(base*4)                   # 4 -> 8
        self.rb_up3 = ResBlock(base*4, base*2, time_dim)
        self.up2 = Up(base*2)                   # 8 -> 16
        self.rb_up2 = ResBlock(base*2, base, time_dim)
        self.up1 = Up(base)                     # 16 -> 32
        self.rb_up1 = ResBlock(base, base, time_dim)

        self.out_norm = nn.GroupNorm(8, base)
        self.out_conv = nn.Conv2d(base, in_ch, 1)

    def forward(self, x, t):
        # x: (B,3,32,32), t: (B,)
        t_emb = self.time_mlp(t)

        x0 = self.in_conv(x)                    # (B, base, 32, 32)
        x1 = self.rb1(x0, t_emb)
        x2 = self.rb2(self.down1(x1), t_emb)
        x3 = self.rb3(self.down2(x2), t_emb)
        mid = self.rb_mid(self.down3(x3), t_emb)

        d3 = self.up3(mid)
        d3 = self.rb_up3(d3, t_emb)
        d2 = self.up2(d3)
        d2 = self.rb_up2(d2, t_emb)
        d1 = self.up1(d2)
        d1 = self.rb_up1(d1, t_emb)

        out = self.out_conv(F.silu(self.out_norm(d1)))
        return out  # predicted noise ε(x_t, t)

def get_model(model_name):
    return UNet()