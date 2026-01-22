import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from .block import ACDA
import math
from timm.models.layers import trunc_normal_tf_
from timm.models.helpers import named_apply
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def _init_weights(module, name, scheme=''):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            # efficientnet like
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)

def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer

class ME(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor=2):
        super(ME, self).__init__()
        mid_channels = in_channels * expansion_factor
        self.use_res = (in_channels == out_channels)

        # 通道扩展
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        # 串联 DW + PW 卷积序列
        self.dw_pw = nn.Sequential(
            # DW 3x3
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # PW 1x1
            nn.Conv2d(mid_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            # DW 5x5
            nn.Conv2d(mid_channels, mid_channels, kernel_size=5, padding=2, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # PW 1x1
            nn.Conv2d(mid_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        shortcut = x
        x = self.expand(x)
        x = self.dw_pw(x)
        out = self.fuse(x)
        if self.use_res:
            out = out + shortcut
        return out



class UP(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu'):
        super(UP, self).__init__()

        # 上采样 → 保留空间结构
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.local_refine = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True) if activation == 'relu' else nn.GELU()
        )

        self.channel_match = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.upsample(x)
        x = self.local_refine(x)
        x = self.channel_match(x)
        return x


class SCFG(nn.Module):
    def __init__(self, F_g, F_l, reduction=16):
        super(SCFG, self).__init__()
        self.align = nn.Conv2d(F_g, F_l, kernel_size=1)
        #DW+PW
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(F_l, F_l, kernel_size=3, padding=1, groups=F_l, bias=False),  # DW卷积
            nn.BatchNorm2d(F_l),
            nn.ReLU(inplace=True),
            nn.Conv2d(F_l, F_l, kernel_size=1, bias=False),  # PW卷积
            nn.BatchNorm2d(F_l),
            nn.ReLU(inplace=True)
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_gate = nn.Sequential(
            nn.Conv2d(F_l, F_l // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(F_l // reduction, F_l, kernel_size=1),
            nn.Sigmoid()
        )

        self.spatial_gate = nn.Sequential(
            nn.Conv2d(F_l, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, g, x):
        g_aligned = self.align(g)
        g = self.conv_fusion(g_aligned)
        x = self.conv_fusion(x)
        fusion = F.relu(g + x)

        # 通道注意力（使用全局语义）
        ch_att = self.channel_gate(self.global_pool(fusion))  # [B, C, 1, 1]

        # 空间注意力（使用局部结构）
        sp_att = self.spatial_gate(fusion)                    # [B, 1, H, W]

        psi = ch_att * sp_att  # broadcast → [B, C, H, W]
        return x * psi




class FLDA(nn.Module):
    def __init__(self, channels=[512,320,128,64]):
        super(FLDA,self).__init__()

        self.me4 = ME(channels[0], channels[0])

        self.up3 = UP(in_channels=channels[0], out_channels=channels[1])
        self.fg3 = SCFG(F_g=channels[1], F_l=channels[1])
        self.me3 = ME(channels[1], channels[1])

        self.up2 = UP(in_channels=channels[1], out_channels=channels[2])
        self.fg2 = SCFG(F_g=channels[2], F_l=channels[2])
        self.me2 = ME(channels[2], channels[2])

        self.up1 = UP(in_channels=channels[2], out_channels=channels[3])
        self.fg1 = SCFG(F_g=channels[3], F_l=channels[3])
        self.me1 = ME(channels[3], channels[3])

        self.da4 = ACDA(channels[0], channels[0])
        self.da3 = ACDA(channels[1], channels[1])
        self.da2 = ACDA(channels[2], channels[2])
        self.da1 = ACDA(channels[3], channels[3])

    def forward(self, x, skips):

        d4 = x
        # d4 = self.da4(x)
        d4 = self.me4(d4)

        # UP3
        d3 = self.up3(d4)

        # SCFG3
        # x3 = skips[0]
        x3 = self.fg3(g =d3, x = skips[0])

        # Additive aggregation 3
        d3 = d3 + x3

        # ME3
        d3 = self.da3(d3)
        d3 = self.me3(d3)

        # UP2
        d2 = self.up2(d3)

        # SCFG2
        # x2 = skips[1]
        x2 = self.fg2(g=d2, x=skips[1])

        # Additive aggregation 2
        d2 = d2 + x2

        # ME2

        d2 = self.da2(d2)
        d2 = self.me2(d2)

        # UP1
        d1 = self.up1(d2)

        # SCFG1
        # x1 = skips[2]
        x1 = self.fg1(g=d1, x=skips[2])

        d1 = d1 + x1

        # ME1

        d1 = self.da1(d1)
        d1 = self.me1(d1)

        return [d4, d3, d2, d1]
