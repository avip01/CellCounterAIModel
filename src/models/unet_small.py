from __future__ import annotations
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
def __init__(self, in_c, out_c):
super().__init__()
self.net = nn.Sequential(
nn.Conv2d(in_c, out_c, 3, padding=1), nn.ReLU(inplace=True),
nn.Conv2d(out_c, out_c, 3, padding=1), nn.ReLU(inplace=True),
)
def forward(self, x):
return self.net(x)


class Down(nn.Module):
def __init__(self, in_c, out_c):
super().__init__()
self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_c, out_c))
def forward(self, x):
return self.net(x)


class Up(nn.Module):
def __init__(self, in_c, out_c):
super().__init__()
self.up = nn.ConvTranspose2d(in_c, in_c//2, 2, stride=2)
self.conv = DoubleConv(in_c, out_c)
def forward(self, x1, x2):
x1 = self.up(x1)
dy = x2.size(2) - x1.size(2)
dx = x2.size(3) - x1.size(3)
x1 = nn.functional.pad(x1, [dx//2, dx-dx//2, dy//2, dy-dy//2])
x = torch.cat([x2, x1], dim=1)
return self.conv(x)


class UNetSmall(nn.Module):
def __init__(self, in_ch=3, out_ch=1):
super().__init__()
self.inc = DoubleConv(in_ch, 32)
self.d1 = Down(32, 64)
self.d2 = Down(64, 128)
self.u1 = Up(128, 64)
self.u2 = Up(64, 32)
self.outc = nn.Conv2d(32, out_ch, 1)
def forward(self, x):
x1 = self.inc(x)
x2 = self.d1(x1)
x3 = self.d2(x2)
x = self.u1(x3, x2)
x = self.u2(x, x1)
return self.outc(x)