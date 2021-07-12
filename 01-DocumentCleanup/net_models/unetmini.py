import torch
import torch.nn as nn
from torch.nn import Module
from torch.nn import Sequential
from torch.nn import Conv2d, Dropout2d, MaxPool2d, ReLU, UpsamplingNearest2d, Sigmoid


class Interpolate(Module):
    def __init__(self, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.mode = mode
        
    def forward(self, x, size):
        x = self.interp(x, size=size, mode=self.mode, align_corners=False)
        return x


class UNetMini(Module):

    def __init__(self, num_classes):
        super(UNetMini, self).__init__()

        self.block1 = Sequential(
            Conv2d(1, 32, kernel_size=3, padding=1),
            ReLU(),
            Dropout2d(0.2),
            Conv2d(32, 32, kernel_size=3, padding=1),
            ReLU(),
        )
        self.pool1 = MaxPool2d((2, 2))

        self.block2 = Sequential(
            Conv2d(32, 64, kernel_size=3, padding=1),
            ReLU(),
            Dropout2d(0.2),
            Conv2d(64, 64, kernel_size=3, padding=1),
            ReLU(),
        )
        self.pool2 = MaxPool2d((2, 2))

        self.block3 = Sequential(
            Conv2d(64, 128, kernel_size=3, padding=1),
            ReLU(),
            Dropout2d(0.2),
            Conv2d(128, 128, kernel_size=3, padding=1),
            ReLU()
        )

        self.up1 = UpsamplingNearest2d(scale_factor=2)
        self.block4 = Sequential(
            Conv2d(192, 64, kernel_size=3, padding=1),
            ReLU(),
            Dropout2d(0.2),
            Conv2d(64, 64, kernel_size=3, padding=1),
            ReLU()
        )

        self.up2 = UpsamplingNearest2d(scale_factor=2)
        self.block5 = Sequential(
            Conv2d(96, 32, kernel_size=3, padding=1),
            ReLU(),
            Dropout2d(0.2),
            Conv2d(32, 32, kernel_size=3, padding=1),
            ReLU()
        )

        self.interpolate = Interpolate(mode='bilinear')
        self.conv2d = Conv2d(32, num_classes, kernel_size=1)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        out1 = self.block1(x)
        out_pool1 = self.pool1(out1)

        out2 = self.block2(out_pool1)
        out_pool2 = self.pool1(out2)

        out3 = self.block3(out_pool2)

        # out_up1 = self.up1(out3)
        out_up1 = self.interpolate(out3, out2.shape[-2:])
        out4 = torch.cat((out_up1, out2), dim=1)
        out4 = self.block4(out4)

        # out_up2 = self.up2(out4)
        out_up2 = self.interpolate(out4, out1.shape[-2:])
        out5 = torch.cat((out_up2, out1), dim=1)
        out5 = self.block5(out5)

        out = self.conv2d(out5)
        out = self.sigmoid(out)

        return out
