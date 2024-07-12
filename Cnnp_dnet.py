import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
import cv2
import numpy as np

class Cnnp(nn.Module):
    def __init__(self):
        super(Cnnp, self).__init__()
        self.p0 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.dw1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.dw2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.p01 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        # decoder
        self.up2 = nn.Sequential(
            CAA(256,1),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.up1 = nn.Sequential(
            CAA(256, 1),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.up0 = nn.Sequential(
            CAA(128+64, 1),
            nn.Conv2d(in_channels=128+64, out_channels=96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
        )
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
        )
        self.relu2 = nn.ReLU(inplace=True)


        self.p3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1),
        )
    def forward(self, image):
        p0 = self.p0(image)
        dw1 = self.dw1(p0)
        dw2 = self.dw2(dw1)
        p1 = self.p01(dw2)
        up2 = self.up2(torch.cat((p1, dw2), dim=1))
        up1 = self.up1(torch.cat((up2, dw1), dim=1))
        up0 = self.up0(torch.cat((up1, p0), dim=1))
        out1 = self.conv1(up0)
        out2 = self.relu1(out1+up0)
        out3 = self.conv2(out2)
        out4 = self.relu2(out3+out2)
        p3 = self.p3(out4)
        return p3

class CAA(nn.Module):
    def __init__(self,inc,rat):
        super(CAA, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.f1 = nn.Conv2d(inc,inc*rat,kernel_size=1)
        self.f2 = nn.Conv2d(inc, inc * rat, kernel_size=1)
        self.g = nn.Conv2d(inc*rat, inc, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.sig = nn.Sigmoid()
    def forward(self,x):
        x0=x
        x1 = self.gap(x)
        f1=self.f1(x1)
        f2=self.f2(x1)
        x=self.relu(f1)*f2
        x=self.sig(self.g(x))
        return x*x0




