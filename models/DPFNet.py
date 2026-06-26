import torch
import torch.nn as nn
import torch.nn.functional as F
from .torch_wavelets import DWT, IDWT
from .Fourier_T import *
from .Attention import SelfAttention


class HFAM(nn.Module):
    def __init__(self, in_channel) -> None:
        super().__init__()

        self.fscale_d = nn.Parameter(torch.zeros(in_channel), requires_grad=True)
        self.fscale_h = nn.Parameter(torch.zeros(in_channel), requires_grad=True)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x_d = self.gap(x)
        x_h = (x - x_d) * (self.fscale_h[None, :, None, None] + 1.)
        x_d = x_d * self.fscale_d[None, :, None, None]
        return x_d + x_h


class WaveDown(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.dwt = DWT(wave='haar')

        self.to_att = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, 1, 1, 0),
            nn.Sigmoid()
        )
        self.FRDB = APRM(in_channels)

        self.global_ap = HFAM(in_channels * 3)

    def forward(self, x):
        x = self.dwt(x)
        x_ll, x_lh, x_hl, x_hh = x.chunk(4, dim=1)
        # ablation 1(LFGA):
        # lh = x_ll + x_lh
        # hl = x_ll + x_hl
        lh = self.FRDB(x_ll + x_lh)  # ablation APRM
        hl = self.FRDB(x_ll + x_hl)
        att_map = self.to_att(lh + hl)
        # att_map = self.to_att(x_ll)

        hi_bands = torch.cat([x_lh, x_hl, x_hh], dim=1)
        # ablation 2(残差分解机制):
        hi_bands = self.global_ap(hi_bands) + hi_bands  # 微调

        return att_map, hi_bands


class DUIM(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.multi_attention = SelfAttention(dim=in_ch * 2, num_heads=32, bias=True)

    def forward(self, x_pix, x_idwt):
        x = torch.cat([x_pix, x_idwt], dim=1)
        # ablation 5:
        # x_o = x
        x_o = self.multi_attention(x)
        pix_up, o_idwt = x_o.chunk(2, dim=1)
        return pix_up + o_idwt


class WaveUp(nn.Module):
    def __init__(self, pix_ch):
        super().__init__()

        self.idwt = IDWT(wave='haar')
        self.upsapling = nn.Sequential(
            nn.Conv2d(pix_ch, pix_ch * 4, 1, 1, 0),
            nn.PixelShuffle(2)
        )
        self.interact = DUIM(pix_ch)
        self.FFN = MCFN(dim=pix_ch)

    def forward(self, x, hi_bands):
        x_1, x_2 = x.chunk(2, dim=1)
        pix_up = self.upsapling(x_1)
        o_idwt = self.idwt(torch.cat([x_2, hi_bands], dim=1))
        # ablation 6（DUIM模块）:
        o = self.interact(pix_up, o_idwt)
        # o = pix_up + o_idwt

        # ablation 6（MCFN模块）:
        o = self.FFN(o) + o
        return o


class CFEM_DFEM(nn.Module):
    def __init__(self, in_ch=64, hi_ch=96):
        super().__init__()

        # 颜色特征捕获
        self.conv_first_x = nn.Conv2d(in_ch, in_ch * 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.instance_x = nn.InstanceNorm2d(in_ch * 4, affine=True)
        self.conv_out_x = nn.Conv2d(in_ch * 4, in_ch, kernel_size=1, stride=1, padding=0, bias=False)

        # 细节特征捕获
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.refine2 = nn.Conv2d(hi_ch, hi_ch, kernel_size=3, stride=1, padding=1)
        self.conv1010 = nn.Conv2d(hi_ch, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        self.conv1020 = nn.Conv2d(hi_ch, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        self.conv1030 = nn.Conv2d(hi_ch, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        self.refine3 = nn.Conv2d(hi_ch + 3, hi_ch, kernel_size=3, stride=1, padding=1)
        self.upsample = F.upsample_nearest

    def forward(self, x):
        x1, hi_bands = x[0], x[1]

        # ablation 3:(LFGA-CFEM)
        x_out = self.conv_first_x(x1)
        x_out = self.instance_x(x_out)
        out = self.conv_out_x(x_out) + x1
        # out = x1

        # # ablation 4:
        dehaze_hi = self.relu((self.refine2(hi_bands)))
        shape_out = dehaze_hi.data.size()
        shape_out = shape_out[2:4]
        x001 = F.avg_pool2d(dehaze_hi, 64)
        x002 = F.avg_pool2d(dehaze_hi, 32)
        x003 = F.avg_pool2d(dehaze_hi, 16)
        x0010 = self.upsample(self.relu(self.conv1010(x001)), size=shape_out)
        x0020 = self.upsample(self.relu(self.conv1020(x002)), size=shape_out)
        x0030 = self.upsample(self.relu(self.conv1030(x003)), size=shape_out)
        dehaze_hi = torch.cat((x0010, x0020, x0030, dehaze_hi), 1)
        hi_bands = self.refine3(dehaze_hi) + hi_bands
        hi_bands = self.tanh(hi_bands) + hi_bands

        return out, hi_bands


class CFEM(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv_first = nn.Conv2d(in_channels, in_channels * 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.instance = nn.InstanceNorm2d(in_channels * 4, affine=True)
        self.conv_out = nn.Conv2d(in_channels * 4, in_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x_i = self.conv_first(x)
        x_i = self.instance(x_i)
        x_i = self.conv_out(x_i)
        return x_i + x


class DPFNet(nn.Module):
    def __init__(self, input_nc=3, ngf=32):
        super(DPFNet, self).__init__()

        self.conv = nn.Conv2d(input_nc, ngf, kernel_size=1, padding=0)

        self.down2 = nn.Sequential(WaveDown(ngf),
                                   CFEM_DFEM(in_ch=ngf * 2, hi_ch=ngf * 3),
                                   )

        self.down3 = nn.Sequential(WaveDown(ngf * 2),
                                   CFEM_DFEM(in_ch=ngf * 4, hi_ch=ngf * 6),
                                   CFEM_DFEM(in_ch=ngf * 4, hi_ch=ngf * 6),
                                   )
        self.z_bottle = nn.Sequential(
            CFEM(ngf * 4),
            CFEM(ngf * 4)
        )

        self.up1 = WaveUp(pix_ch=ngf * 2)
        self.up2 = WaveUp(pix_ch=ngf)

        self.deconv = nn.Conv2d(ngf, input_nc, 1, 1, bias=True)

        self.tanh = nn.Tanh()
        self.Mix = AdaptiveMixing(m=-0.6)

    def forward(self, input):

        x_1 = self.conv(input)

        x_down2, hi_bands_hr = self.down2(x_1)

        x_down3, hi_bands_lr = self.down3(x_down2)

        # ablation 3:
        # z = x_down3

        z = self.z_bottle(x_down3)

        x_up1 = self.up1(z, hi_bands_lr) # + x_down2

        x_up1 = self.Mix(x_down2, x_up1)

        x_up2 = self.up2(x_up1, hi_bands_hr) + x_1

        out = self.deconv(x_up2)

        out = self.tanh(out)

        return out
