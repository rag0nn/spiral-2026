import torch
from torch import nn
import torch.nn.functional as F


class FocusStem(nn.Module):
    """
    Space-to-Depth (YOLOv5 Focus) katmani.
    640x640x3 girdiyi 160x160xC ciktiya donusturur (4x downsampling, bilgi kaybi yok).
    - 1. Adim: PixelUnshuffle ile 640x640x3 -> 160x160x48 (uzamsal bilgiyi kanala tasir)
    - 2. Adim: Tek bir Conv ile 48 -> out_channels kanal boyutuna indirir
    Tek bir buyuk Conv'a kiyasla FLOP tasarrufu saglar.
    """
    def __init__(self, in_channels=3, out_channels=32, downscale=4):
        super().__init__()
        # PixelUnshuffle: H,W -> H/d, W/d ve kanallar d^2 kat artar
        self.unshuffle = nn.PixelUnshuffle(downscale_factor=downscale)
        shuffled_channels = in_channels * (downscale ** 2)
        self.conv = Conv(shuffled_channels, out_channels, k=1)

    def forward(self, x):
        return self.conv(self.unshuffle(x))


class DeepStem(nn.Module):
    """
    Ardisik hafif 3x3 evrişimlerle erken downsampling yapan Deep Stem katmani.
    640x640x3 -> 160x160xC (4x downsampling, iki adimda s=2 ile)
    Buyuk kernel tek evrişimden daha iyi gradient akisi ve regularization saglar.
    """
    def __init__(self, in_channels=3, out_channels=32):
        super().__init__()
        mid = out_channels // 2
        self.stem = nn.Sequential(
            # 640x640x3 -> 320x320xmid
            Conv(in_channels, mid, k=3, s=2),
            # 320x320xmid -> 320x320xmid (receptive field genislet)
            Conv(mid, mid, k=3, s=1),
            # 320x320xmid -> 160x160xout
            Conv(mid, out_channels, k=3, s=2),
        )

    def forward(self, x):
        return self.stem(x)


class SeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   padding=kernel_size // 2, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, g=1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, k // 2, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3K2(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, 2 * c_, 1, 1)
        self.cv2 = Conv((2 + n) * c_, c2, 1)
        self.m = nn.ModuleList(Bottleneck(c_, c_, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class DetectBranch(nn.Module):
    def __init__(self, in_c, out_c, n=2):
        super().__init__()
        self.convs = nn.Sequential(*[Conv(in_c, in_c, k=3, s=1) for _ in range(n)])
        self.out = nn.Conv2d(in_c, out_c, 1)

    def forward(self, x):
        return self.out(self.convs(x))


class PosBranch(nn.Module):
    def __init__(self, in_c, n=2):
        super().__init__()
        self.convs = nn.Sequential(*[Conv(in_c, in_c, k=3, s=1) for _ in range(n)])

    def forward(self, x):
        return self.convs(x).mean(dim=[2, 3])


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, pool_size=16):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.pool = nn.AdaptiveAvgPool2d(pool_size)

    def forward(self, q, kv):
        B, C, H, W = q.shape
        qp = self.pool(q)
        kvp = self.pool(kv)
        Ph, Pw = qp.shape[2], qp.shape[3]
        q_flat = qp.flatten(2).transpose(1, 2)
        kv_flat = kvp.flatten(2).transpose(1, 2)
        out, _ = self.attn(q_flat, kv_flat, kv_flat)
        out = out.transpose(1, 2).reshape(B, C, Ph, Pw)
        out = F.interpolate(out, (H, W), mode="bilinear", align_corners=False)
        return out