from tensorflow.keras import callbacks
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from tensorflow import keras
import tensorflow as tf

class _DenseLayer(layers.Layer):
    """_DenseBlock model.

       Arguments:
         out_features: number of output features
    """

    def __init__(self,
                 out_features,
	         **kwargs):
        super(_DenseLayer, self).__init__(**kwargs)
        self.layers = []
        self.layers.append(tf.keras.Sequential(
            [
                layers.Conv2D(filters=out_features, kernel_size=1, strides=1, padding='same', activation=None),
                layers.BatchNormalization(),
                layers.ReLU(),
                layers.Conv2D(filters=out_features, kernel_size=3, strides=1, padding='same', activation=None),
                layers.BatchNormalization(),
            ]))


    def call(self, inputs):
        x1, x2 = tuple(inputs)
        new_features = x1
        for layer in self.layers:
            new_features = layer(new_features)

        return 0.5 * (new_features + x2), x2


class _DenseBlock(layers.Layer):
    """DenseBlock layer.

       Arguments:
         num_layers: number of _DenseLayer's per block
         out_features: number of output features
    """

    def __init__(self,
                 num_layers,
                 out_features,
                 **kwargs):
        super(_DenseBlock, self).__init__(**kwargs)
        self.layers = [_DenseLayer(out_features) for i in range(num_layers)]

    def call(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)

        return inputs


class UpConvBlock(layers.Layer):
    """UpConvDeconvBlock layer.

       Arguments:
         up_scale: int
    """

    def __init__(self,
                 up_scale,
                 **kwargs):
        super(UpConvBlock, self).__init__(**kwargs)
        constant_features = 16

        features = []
        total_up_scale = 2 ** up_scale
        for i in range(up_scale):
            out_features = 1 if i == up_scale-1 else constant_features
            features.append(layers.Conv2D(filters=out_features, kernel_size=1, strides=1, padding='same', activation='relu'))
            features.append(layers.Conv2DTranspose(out_features, total_up_scale, strides=2, padding='same', activation=None))

        self.features = keras.Sequential(features)

    def call(self, inputs):
        return self.features(inputs)


class SingleConvBlock(layers.Layer):
    """SingleConvBlock layer.

       Arguments:
         out_features: number of output features
         stride: stride per convolution
    """

    def __init__(self,
                 out_features,
                 stride,
                 **kwargs):
        super(SingleConvBlock, self).__init__(**kwargs)
        self.conv = layers.Conv2D(filters=out_features, kernel_size=(1, 1), strides=(stride, stride), padding='valid', activation=None)
        self.bn = layers.BatchNormalization()

    def call(self, inputs):
        return self.bn(self.conv(inputs))


class DoubleConvBlock(layers.Layer):
    """DoubleConvBlock layer.

       Arguments:
         mid_features: number of middle features
         out_features: number of output features
         stride: stride per mid-layer convolution
    """

    def __init__(self,
                 mid_features,
                 out_features,
                 stride=1,
                 **kwargs):
        super(DoubleConvBlock, self).__init__(**kwargs)
        self.conv1 = layers.Conv2D(filters=mid_features, kernel_size=(3, 3), strides=stride, padding='same', activation=None)
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(filters=out_features, kernel_size=(3, 3), padding='same', activation=None)
        self.bn2 = layers.BatchNormalization()
        self.relu = layers.ReLU()

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        return self.relu(x)


class DexiNet(tf.keras.Model):
    """DexiNet model."""

    def __init__(self,
                 **kwargs):
        super(DexiNet, self).__init__(**kwargs)
        self.block_1 = DoubleConvBlock(32, 64, stride=2)
        self.block_2 = DoubleConvBlock(128, 128)
        self.dblock_3 = _DenseBlock(2, 256)
        self.dblock_4 = _DenseBlock(3, 512)
        self.dblock_5 = _DenseBlock(3, 512)
        self.dblock_6 = _DenseBlock(3, 256)
        self.maxpool = layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same')

        self.side_1 = SingleConvBlock(128, 2)
        self.side_2 = SingleConvBlock(256, 2)
        self.side_3 = SingleConvBlock(512, 2)
        self.side_4 = SingleConvBlock(512, 1)
        self.side_5 = SingleConvBlock(256, 1)

        self.pre_dense_2 = SingleConvBlock(256, 2)
        self.pre_dense_3 = SingleConvBlock(256, 1)
        self.pre_dense_4 = SingleConvBlock(512, 1)
        self.pre_dense_5_0 = SingleConvBlock(512, 2)
        self.pre_dense_5 = SingleConvBlock(512, 1)
        self.pre_dense_6 = SingleConvBlock(256, 1)

        self.up_block_1 = UpConvBlock(1)
        self.up_block_2 = UpConvBlock(1)
        self.up_block_3 = UpConvBlock(2)
        self.up_block_4 = UpConvBlock(3)
        self.up_block_5 = UpConvBlock(4)
        self.up_block_6 = UpConvBlock(4)

        self.block_cat = layers.Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation=None)


    def slice(self, tensor, slice_shape):
        height, width = slice_shape
        return tensor[..., :height, :width]


    def call(self, x):
        # Block 1
        block_1 = self.block_1(x)
        block_1_side = self.side_1(block_1)

        # Block 2
        block_2 = self.block_2(block_1)
        block_2_down = self.maxpool(block_2)
        block_2_add = block_2_down + block_1_side
        block_2_side = self.side_2(block_2_add)#

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

        # Block 6
        block_6_pre_dense = self.pre_dense_6(block_5)
        block_6, _ = self.dblock_6([block_5_add, block_6_pre_dense])

        # upsampling blocks
        height, width = x.shape[1:3]
        slice_shape = (height, width)
        out_1 = self.slice(self.up_block_1(block_1), slice_shape)
        out_2 = self.slice(self.up_block_2(block_2), slice_shape)
        out_3 = self.slice(self.up_block_3(block_3), slice_shape)
        out_4 = self.slice(self.up_block_4(block_4), slice_shape)
        out_5 = self.slice(self.up_block_5(block_5), slice_shape)
        out_6 = self.slice(self.up_block_6(block_6), slice_shape)
        results = [out_1, out_2, out_3, out_4, out_5, out_6]

        # concatenate multiscale outputs
        block_cat = tf.concat(results, 3)  # BxHxWX6
        block_cat = self.block_cat(block_cat)  # BxHxWX1

        results.append(block_cat)

        return results


def main(epochs):
    batch_size = 8
    input = tf.random.uniform((batch_size, 400, 400, 3))
    target = tf.random.uniform((batch_size, 400, 400, 1))
    model = DexiNet()
    model.compile(optimizer='Adam', loss='mse')
    model.fit(input, target, epochs)


if __name__ == '__main__':
    main(20000)
