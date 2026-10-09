import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseEncoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class BaseDecoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError




class TestEncoder(BaseEncoder):
    def __init__(self, in_ch, out_ch, stride=16):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, stride=stride, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class TestDecoder(BaseDecoder):
    def __init__(self, in_ch, out_ch, stride=16):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0)
        self.scale = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return F.interpolate(x, size=None, scale_factor=self.scale, mode='nearest')
