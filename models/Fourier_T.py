import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F


class AdaptiveMixing(nn.Module):
    def __init__(self, m=-0.80):
        super(AdaptiveMixing, self).__init__()
        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w = torch.nn.Parameter(w, requires_grad=True)
        self.w = w
        self.mix_block = nn.Sigmoid()

    def forward(self, fea1, fea2):
        mix_factor = self.mix_block(self.w)
        out = fea1 * mix_factor.expand_as(fea1) + fea2 * (1 - mix_factor.expand_as(fea2))
        return out


class LRB(nn.Module):
    def __init__(self, nChannels, growthRate, kernel_size=1):
        super(LRB, self).__init__()
        # self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
        # bias=False)
        self.conv = nn.Sequential(
            nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                      bias=False), nn.BatchNorm2d(growthRate)
        )
        self.bat = nn.BatchNorm2d(growthRate),
        self.leaky = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        out = self.leaky(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


class SRDB(nn.Module):
    def __init__(self, nChannels, growthRate=64):
        super(SRDB, self).__init__()

        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(nChannels, growthRate, kernel_size=5, padding=2, bias=False)
        self.conv3 = nn.Conv2d(nChannels, growthRate, kernel_size=7, padding=3, bias=False)
        self.leaky1 = nn.LeakyReLU(0.1, inplace=True)
        self.leaky2 = nn.LeakyReLU(0.1, inplace=True)
        self.leaky3 = nn.LeakyReLU(0.1, inplace=True)

        self.conv6 = nn.Conv2d(growthRate * 3, nChannels, kernel_size=1, padding=(1 - 1) // 2,
                               bias=False)

    def forward(self, x):
        x_1 = self.leaky1(self.conv1(x))
        x_2 = self.leaky2(self.conv2(x))
        x_3 = self.leaky3(self.conv3(x))
        x_0 = torch.cat((x_1, x_2, x_3), dim=1)

        x_0 = self.conv6(x_0)

        out = x_0 + x
        return out


class MCFN(nn.Module):
    def __init__(
            self,
            dim,
    ):
        super(MCFN, self).__init__()
        self.dim = dim
        self.dim_sp = dim // 2
        # PW first or DW first?
        self.conv_init = nn.Sequential(  # PW->DW->
            nn.Conv2d(dim, dim * 2, 1),
        )

        self.conv1_1 = nn.Sequential(
            nn.Conv2d(self.dim_sp, self.dim_sp, kernel_size=3, padding=1,
                      groups=self.dim_sp),
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(self.dim_sp, self.dim_sp, kernel_size=5, padding=2,
                      groups=self.dim_sp),
        )
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(self.dim_sp, self.dim_sp, kernel_size=7, padding=3,
                      groups=self.dim_sp),
        )

        self.gelu = nn.GELU()
        self.conv_fina = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
        )

    def forward(self, x):
        x = self.conv_init(x)
        x = list(torch.split(x, self.dim_sp, dim=1))
        x[1] = self.conv1_1(x[1])
        x[2] = self.conv1_2(x[2])
        x[3] = self.conv1_3(x[3])
        x = torch.cat(x, dim=1)
        x = self.gelu(x)
        x = self.conv_fina(x)

        return x


class APRM(nn.Module):
    def __init__(self, nChannels, nDenselayer=3, growthRate=32):
        super(APRM, self).__init__()
        nChannels_1 = nChannels
        nChannels_2 = nChannels
        modules1 = []
        for i in range(nDenselayer):
            modules1.append(LRB(nChannels_1, growthRate))
            nChannels_1 += growthRate
        self.dense_layers1 = nn.Sequential(*modules1)
        modules2 = []
        for i in range(nDenselayer):
            modules2.append(LRB(nChannels_2, growthRate))
            nChannels_2 += growthRate
        self.dense_layers2 = nn.Sequential(*modules2)
        self.conv_1 = nn.Conv2d(nChannels_1, nChannels, kernel_size=1, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(nChannels_2, nChannels, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(x, norm='backward')
        mag = torch.abs(x_freq)
        pha = torch.angle(x_freq)
        mag = self.dense_layers1(mag)
        mag = self.conv_1(mag)
        pha = self.dense_layers2(pha)
        pha = self.conv_2(pha)
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)
        out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')
        out = out + x
        return out
