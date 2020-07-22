import torch
import torch.nn as nn
import torch.nn.functional as F

class _DenseLayer(nn.Sequential):
    def __init__(self, input_features, out_features):
        super(_DenseLayer, self).__init__()
        # self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(input_features, out_features,
                        kernel_size=1, stride=1, bias=True)),
        self.add_module('norm1', nn.BatchNorm2d(out_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(out_features, out_features,
                        kernel_size=3, stride=1, padding=1, bias=True)),
        self.add_module('norm2', nn.BatchNorm2d(out_features))
        # double check the norm1 comment if necessary and put norm after conv2

    def forward(self, x):
        x1, x2 = x
        # maybe I should put here a RELU
        new_features = super(_DenseLayer, self).forward(F.relu(x1)) # F.relu()
        return 0.5 * (new_features + x2), x2

class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, input_features, out_features):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(input_features, out_features)
            self.add_module('denselayer%d' % (i + 1), layer)
            input_features = out_features

class UpConvBlock(nn.Module):
    def __init__(self, in_features, up_scale, mode='deconv'):
        super(UpConvBlock, self).__init__()
        self.up_factor = 2
        self.constant_features = 16

        layers = None
        if mode == 'deconv':
            layers = self.make_deconv_layers(in_features, up_scale)
        elif mode == 'pixel_shuffle':
            layers = self.make_pixel_shuffle_layers(in_features, up_scale)
        assert layers is not None, layers
        self.features = nn.Sequential(*layers)

    def make_deconv_layers(self, in_features, up_scale):
        layers = []
        for i in range(up_scale):
            kernel_size = 2 ** up_scale
            out_features = self.compute_out_features(i, up_scale)
            layers.append(nn.Conv2d(in_features, out_features, 1))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.ConvTranspose2d(
                out_features, out_features, kernel_size, stride=2))
            in_features = out_features
        return layers

    def make_pixel_shuffle_layers(self, in_features, up_scale):
        layers = []
        for i in range(up_scale):
            kernel_size = 2 ** (i + 1)
            out_features = self.compute_out_features(i, up_scale)
            in_features = int(in_features / (self.up_factor ** 2))
            layers.append(nn.PixelShuffle(self.up_factor))
            layers.append(nn.Conv2d(in_features, out_features, 1))
            if i < up_scale:
                layers.append(nn.ReLU(inplace=True))
            in_features = out_features
        return layers

    def compute_out_features(self, idx, up_scale):
        return 1 if idx == up_scale - 1 else self.constant_features

    def forward(self, x):
        return self.features(x)

class SingleConvBlock(nn.Module):
    def __init__(self, in_features, out_features, stride, use_bs=True):
        super(SingleConvBlock, self).__init__()
        self.use_bn=True
        self.conv = nn.Conv2d(in_features, out_features, 1, stride=stride)
        self.bn = nn.BatchNorm2d(out_features)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)

        return x

class DoubleConvBlock(nn.Module):
    def __init__(self, in_features, mid_features,out_features=None, stride=1,
                 use_act=True):
        super(DoubleConvBlock, self).__init__()
        self.use_act=use_act
        if out_features is None:
            out_features = mid_features
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
        if self.use_act:
            x = self.relu(x)
        return x

class DexiNet(nn.Module):
    """ Definition of the DXtrem network. """
    def __init__(self):
        super(DexiNet, self).__init__()
        self.block_1 = DoubleConvBlock(3, 32, 64, stride=2,)
        self.block_2 = DoubleConvBlock(64, 128, use_act=False)
        self.dblock_3 = _DenseBlock(2, 128, 256)
        self.dblock_4 = _DenseBlock(3, 256, 512)
        self.dblock_5 = _DenseBlock(3, 512, 512)
        self.dblock_6 = _DenseBlock(3, 512, 256)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.side_1 = SingleConvBlock(64, 128, 2)
        self.side_2 = SingleConvBlock(128, 256, 2)
        self.side_3 = SingleConvBlock(256, 512, 2)
        self.side_4 = SingleConvBlock(512, 512, 1)
        self.side_5 = SingleConvBlock(512, 256, 1)

        self.pre_dense_2 = SingleConvBlock(128, 256, 2, use_bs=False) # by me, for left skip block4
        self.pre_dense_3 = SingleConvBlock(128, 256, 1)
        self.pre_dense_4 = SingleConvBlock(256, 512, 1)
        self.pre_dense_5_0 = SingleConvBlock(256, 512, 2,use_bs=False)
        self.pre_dense_5 = SingleConvBlock(512, 512, 1)
        self.pre_dense_6 = SingleConvBlock(512, 256, 1)

        self.up_block_1 = UpConvBlock(64, 1)
        self.up_block_2 = UpConvBlock(128, 1)
        self.up_block_3 = UpConvBlock(256, 2)
        self.up_block_4 = UpConvBlock(512, 3)
        self.up_block_5 = UpConvBlock(512, 4)
        self.up_block_6 = UpConvBlock(256, 4)
        self.block_cat = nn.Conv2d(6, 1, kernel_size=1)

    def slice(self, tensor, slice_shape):
        height, width = slice_shape
        return tensor[..., :height, :width]

    def forward(self, x):
        assert len(x.shape) == 4, x.shape
        # Block 1
        block_1 = self.block_1(x)
        block_1_side = self.side_1(block_1)

        # Block 2
        block_2 = self.block_2(block_1)
        block_2_down = self.maxpool(block_2)
        block_2_add = block_2_down + block_1_side
        block_2_side = self.side_2(block_2_add)

        # Block 3
        block_3_pre_dense = self.pre_dense_3(block_2_down)
        block_3, _ = self.dblock_3([block_2_add, block_3_pre_dense])
        block_3_down = self.maxpool(block_3)
        block_3_add = block_3_down + block_2_side
        block_3_side = self.side_3(block_3_add)

        # Block 4
        block_4_pre_dense_256 = self.pre_dense_2(block_2_down)
        block_4_pre_dense = self.pre_dense_4(block_4_pre_dense_256 + block_3_down)
        block_4, _ = self.dblock_4([block_3_add, block_4_pre_dense])
        block_4_down = self.maxpool(block_4)
        block_4_add = block_4_down + block_3_side
        block_4_side = self.side_4(block_4_add)

        # Block 5
        block_5_pre_dense_512 = self.pre_dense_5_0(block_4_pre_dense_256)
        block_5_pre_dense = self.pre_dense_5(block_5_pre_dense_512 + block_4_down )
        block_5, _ = self.dblock_5([block_4_add, block_5_pre_dense])
        block_5_add = block_5 + block_4_side
#        block_5_side = self.side_5(block_5_add)

        # Block 6
        block_6_pre_dense = self.pre_dense_6(block_5)
#        block_5_pre_dense_256 = self.pre_dense_6(block_5_add) # if error uncomment
        block_6, _ = self.dblock_6([block_5_add, block_6_pre_dense])

        # upsampling blocks
        height, width = x.shape[-2:]
        slice_shape = (height, width)
        out_1 = self.slice(self.up_block_1(block_1), slice_shape)
        out_2 = self.slice(self.up_block_2(block_2), slice_shape)
        out_3 = self.slice(self.up_block_3(block_3), slice_shape)
        out_4 = self.slice(self.up_block_4(block_4), slice_shape)
        out_5 = self.slice(self.up_block_5(block_5), slice_shape)
        out_6 = self.slice(self.up_block_6(block_6), slice_shape)
        results = [out_1, out_2, out_3, out_4, out_5, out_6]
        # print(out_1.shape)
        # concatenate multiscale outputs
        block_cat = torch.cat(results, dim=1)  # Bx6xHxW
        block_cat = self.block_cat(block_cat)  # Bx1xHxW

        # return results
        results.append(block_cat)
        return results


if __name__ == '__main__':
    batch_size = 8
    input = torch.rand(batch_size, 3, 400, 400).cuda()
    target = torch.rand(batch_size, 1, 400, 400).cuda()
    model = DexiNet().cuda()
    for i in range(20000):
        print(i)
        output = model(input)
        loss = nn.MSELoss()(output[-1], target)
        loss.backward()
