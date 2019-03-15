import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class UpConvBlock(nn.Module):
    def __init__(self, in_features, up_scale):
        super(UpConvBlock, self).__init__()
        up_layers = []
        for i in range(up_scale):
            up_layers.append(
                nn.ConvTranspose2d(in_features, in_features, 2, stride=2))
            up_layers.append(nn.ReLU(inplace=True))
        self.convt = nn.Sequential(*up_layers)
        self.conv = nn.Conv2d(in_features, 1, kernel_size=1)

    def forward(self, x):
        x_up = torch.relu(self.convt(x))
        return self.conv(x_up)

class SideConvBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(SideConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_features, out_features, 1, stride=2)

    def forward(self, x):
        return self.conv(x)

class DoubleConvBlock(nn.Module):
    def __init__(self, in_features, out_features, mid_features=None, stride=1):
        super(DoubleConvBlock, self).__init__()
        if mid_features is None:
            mid_features = out_features
        self.conv1 = nn.Conv2d(
            in_features, mid_features, 3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(mid_features)
        self.conv2 = nn.Conv2d(mid_features, out_features, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_features)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class DXtremNet(nn.Module):
    """ Definition of the DXtrem network. """
    def __init__(self):
        super(DXtremNet, self).__init__()
        # Layers.
        self.block_1 = DoubleConvBlock(3, 64, 32, stride=2)
        self.block_2 = DoubleConvBlock(64, 128)
        self.block_3 = DoubleConvBlock(128, 256)
        self.block_4 = DoubleConvBlock(256, 512)
        self.block_5 = DoubleConvBlock(512, 512)
        self.block_6 = DoubleConvBlock(512, 256)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.side_1 = SideConvBlock(64, 128)
        self.side_2 = SideConvBlock(128, 256)
        self.side_3 = SideConvBlock(256, 512)

        self.up_block_1 = UpConvBlock(64, 1)
        self.up_block_2 = UpConvBlock(128, 2)
        self.up_block_3 = UpConvBlock(256, 3)
        self.up_block_4 = UpConvBlock(512, 4)
        self.up_block_5 = UpConvBlock(512, 4)
        self.up_block_6 = UpConvBlock(256, 4)

        self.block_cat = nn.Conv2d(6, 1, kernel_size=1)

    def forward(self, x):
        assert len(x.shape) == 4, x.shape
        image_h, image_w = x.shape[2], x.shape[3]
        # VGG-16 network. Here you put non-linear,
        block_1 = self.block_1(x)

        block_2 = self.block_2(block_1)
        block_2_down = self.maxpool(block_2) + self.side_1(block_1)

        block_3 = self.block_3(block_2_down)
        block_3_down = self.maxpool(block_3) + self.side_2(block_2_down)

        block_4 = self.block_4(block_3_down)
        block_4_down = self.maxpool(block_4) + self.side_3(block_3_down)

        block_5_down = self.block_5(block_4_down)
        block_6_down = self.block_6(block_5_down)

        out_1 = self.up_block_1(block_1)
        out_2 = self.up_block_2(block_2_down)
        out_3 = self.up_block_3(block_3_down)
        out_4 = self.up_block_4(block_4_down)
        out_5 = self.up_block_5(block_5_down)
        out_6 = self.up_block_6(block_6_down)
        results = [out_1, out_2, out_3, out_4, out_5, out_6]

        block_cat = torch.cat(results, dim=1)
        block_cat = self.block_cat(block_cat)

        results.append(block_cat)
        results = [torch.sigmoid(r) for r in results]
        return results


if __name__ == '__main__':
    input = torch.rand(1, 3, 400, 400)
    model = DXtremNet()
    output = model(input)
    for res in output:
        assert res.shape[-2:] == input.shape[-2:], (res.shape, input.shape)
