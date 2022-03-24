from tensorflow.keras import callbacks
from tensorflow.keras import layers, regularizers
from tensorflow.keras import optimizers, metrics, losses
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from tensorflow import keras
import tensorflow as tf


l2 = regularizers.l2
w_decay=1e-8 #0.0#2e-4#1e-3, 2e-4 # please define weight decay
K.clear_session()
# weight_init = tf.initializers.RandomNormal(mean=0.,stddev=0.01)
# weight_init = tf.initializers.glorot_normal()
weight_init = tf.initializers.glorot_normal()

class _DenseLayer(layers.Layer):
    """_DenseBlock model.

       Arguments:
         out_features: number of output features
    """

    def __init__(self, out_features,**kwargs):
        super(_DenseLayer, self).__init__(**kwargs)
        k_reg = None if w_decay is None else l2(w_decay)
        self.layers = []
        self.layers.append(tf.keras.Sequential(
            [
                layers.ReLU(),
                layers.Conv2D(
                    filters=out_features, kernel_size=(3,3), strides=(1,1), padding='same',
                    use_bias=True, kernel_initializer=weight_init,
                kernel_regularizer=k_reg),
                layers.BatchNormalization(),
                layers.ReLU(),
                layers.Conv2D(
                    filters=out_features, kernel_size=(3,3), strides=(1,1), padding='same',
                    use_bias=True, kernel_initializer=weight_init,
                    kernel_regularizer=k_reg),
                layers.BatchNormalization(),
            ])) # first relu can be not needed


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
                 out_features,**kwargs):
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

    def __init__(self, up_scale,**kwargs):
        super(UpConvBlock, self).__init__(**kwargs)
        constant_features = 16
        k_reg = None if w_decay is None else l2(w_decay)
        features = []
        total_up_scale = 2 ** up_scale
        for i in range(up_scale):
            out_features = 1 if i == up_scale-1 else constant_features
            if i==up_scale-1:
                features.append(layers.Conv2D(
                    filters=out_features, kernel_size=(1,1), strides=(1,1), padding='same',
                    activation='relu', kernel_initializer=tf.initializers.random_normal(mean=0.),
                    kernel_regularizer=k_reg,use_bias=True)) #tf.initializers.TruncatedNormal(mean=0.)
                features.append(layers.Conv2DTranspose(
                    out_features, kernel_size=(total_up_scale,total_up_scale),
                    strides=(2,2), padding='same',
                    kernel_initializer=tf.initializers.random_normal(stddev=0.1),
                    kernel_regularizer=k_reg,use_bias=True)) # stddev=0.1
            else:

                features.append(layers.Conv2D(
                    filters=out_features, kernel_size=(1,1), strides=(1,1), padding='same',
                    activation='relu',kernel_initializer=weight_init,
                kernel_regularizer=k_reg,use_bias=True))
                features.append(layers.Conv2DTranspose(
                    out_features, kernel_size=(total_up_scale,total_up_scale),
                    strides=(2,2), padding='same', use_bias=True,
                    kernel_initializer=weight_init, kernel_regularizer=k_reg))

        self.features = keras.Sequential(features)

    def call(self, inputs):
        return self.features(inputs)


class SingleConvBlock(layers.Layer):
    """SingleConvBlock layer.

       Arguments:
         out_features: number of output features
         stride: stride per convolution
    """

    def __init__(self, out_features, k_size=(1,1),stride=(1,1),
                 use_bs=False, use_act=False,w_init=None,**kwargs): # bias_init=tf.constant_initializer(0.0)
        super(SingleConvBlock, self).__init__(**kwargs)
        self.use_bn = use_bs
        self.use_act = use_act
        k_reg = None if w_decay is None else l2(w_decay)
        self.conv = layers.Conv2D(
            filters=out_features, kernel_size=k_size, strides=stride,
            padding='same',kernel_initializer=w_init,
            kernel_regularizer=k_reg)#, use_bias=True, bias_initializer=bias_init
        if self.use_bn:
            self.bn = layers.BatchNormalization()
        if self.use_act:
            self.relu = layers.ReLU()

    def call(self, inputs):
        x =self.conv(inputs)
        if self.use_bn:
            x = self.bn(x)
        if self.use_act:
            x = self.relu(x)
        return x


class DoubleConvBlock(layers.Layer):
    """DoubleConvBlock layer.

       Arguments:
         mid_features: number of middle features
         out_features: number of output features
         stride: stride per mid-layer convolution
    """

    def __init__(self, mid_features, out_features=None, stride=(1,1),
                 use_bn=True,use_act=True,**kwargs):
        super(DoubleConvBlock, self).__init__(**kwargs)
        self.use_bn =use_bn
        self.use_act =use_act
        out_features = mid_features if out_features is None else out_features
        k_reg = None if w_decay is None else l2(w_decay)

        self.conv1 = layers.Conv2D(
            filters=mid_features, kernel_size=(3, 3), strides=stride, padding='same',
        use_bias=True, kernel_initializer=weight_init,
        kernel_regularizer=k_reg)
        self.bn1 = layers.BatchNormalization()

        self.conv2 = layers.Conv2D(
            filters=out_features, kernel_size=(3, 3), padding='same',strides=(1,1),
        use_bias=True, kernel_initializer=weight_init,
        kernel_regularizer=k_reg)
        self.bn2 = layers.BatchNormalization()
        self.relu = layers.ReLU()

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.use_act:
            x = self.relu(x)
        return x


class DexiNed(tf.keras.Model):
    """DexiNet model."""

    def __init__(self,rgb_mean=None,
                 **kwargs):
        super(DexiNed, self).__init__(**kwargs)

        self.rgbn_mean = rgb_mean
        self.block_1 = DoubleConvBlock(32, 64, stride=(2,2),use_act=False)
        self.block_2 = DoubleConvBlock(128,use_act=False)
        self.dblock_3 = _DenseBlock(2, 256)
        self.dblock_4 = _DenseBlock(3, 512)
        self.dblock_5 = _DenseBlock(3, 512)
        self.dblock_6 = _DenseBlock(3, 256)
        self.maxpool = layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same')

        # left skip connections, figure in Journal
        self.side_1 = SingleConvBlock(128,k_size=(1,1),stride=(2,2),use_bs=True,
                                      w_init=weight_init)
        self.side_2 = SingleConvBlock(256,k_size=(1,1),stride=(2,2),use_bs=True,
                                      w_init=weight_init)
        self.side_3 = SingleConvBlock(512,k_size=(1,1),stride=(2,2),use_bs=True,
                                      w_init=weight_init)
        self.side_4 = SingleConvBlock(512,k_size=(1,1),stride=(1,1),use_bs=True,
                                      w_init=weight_init)
        # self.side_5 = SingleConvBlock(256,k_size=(1,1),stride=(1,1),use_bs=True,
        #                               w_init=weight_init)

        # right skip connections, figure in Journal paper
        self.pre_dense_2 = SingleConvBlock(256,k_size=(1,1),stride=(2,2),
                                      w_init=weight_init) # use_bn=True
        self.pre_dense_3 = SingleConvBlock(256,k_size=(1,1),stride=(1,1),use_bs=True,
                                      w_init=weight_init)
        self.pre_dense_4 = SingleConvBlock(512,k_size=(1,1),stride=(1,1),use_bs=True,
                                      w_init=weight_init)
        # self.pre_dense_5_0 = SingleConvBlock(512, k_size=(1,1),stride=(2,2),
        #                               w_init=weight_init) # use_bn=True
        self.pre_dense_5 = SingleConvBlock(512,k_size=(1,1),stride=(1,1),use_bs=True,
                                      w_init=weight_init)
        self.pre_dense_6 = SingleConvBlock(256,k_size=(1,1),stride=(1,1),use_bs=True,
                                      w_init=weight_init)
        # USNet
        self.up_block_1 = UpConvBlock(1)
        self.up_block_2 = UpConvBlock(1)
        self.up_block_3 = UpConvBlock(2)
        self.up_block_4 = UpConvBlock(3)
        self.up_block_5 = UpConvBlock(4)
        self.up_block_6 = UpConvBlock(4)

        self.block_cat = SingleConvBlock(
            1,k_size=(1,1),stride=(1,1),
            w_init=tf.constant_initializer(1/5))


    def slice(self, tensor, slice_shape):
        height, width = slice_shape
        return tensor[..., :height, :width]


    def call(self, x):
        # Block 1
        x = x-self.rgbn_mean[:-1]
        block_1 = self.block_1(x)
        block_1_side = self.side_1(block_1)

        # Block 2
        block_2 = self.block_2(block_1)
        block_2_down = self.maxpool(block_2) # the key for the second skip connec...
        block_2_add = block_2_down + block_1_side
        block_2_side = self.side_2(block_2_add) #

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
        # block_5_pre_dense_512 = self.pre_dense_5_0(block_4_pre_dense_256)
        block_5_pre_dense = self.pre_dense_5(block_4_down )
        block_5, _ = self.dblock_5([block_4_add, block_5_pre_dense])
        block_5_add = block_5 + block_4_side

        # Block 6
        block_6_pre_dense = self.pre_dense_6(block_5)
        block_6, _ = self.dblock_6([block_5_add, block_6_pre_dense])

        # upsampling blocks
        height, width = x.shape[1:3]
        slice_shape = (height, width)
        out_1 = self.up_block_1(block_1) # self.slice(, slice_shape)
        out_2 = self.up_block_2(block_2)
        out_3 = self.up_block_3(block_3)
        out_4 = self.up_block_4(block_4)
        out_5 = self.up_block_5(block_5)
        out_6 = self.up_block_6(block_6)
        results = [out_1, out_2, out_3, out_4, out_5, out_6]

        # concatenate multiscale outputs
        block_cat = tf.concat(results, 3)  # BxHxWX6
        block_cat = self.block_cat(block_cat)  # BxHxWX1

        results.append(block_cat)

        return results

def weighted_cross_entropy_loss(input, label):
    y = tf.cast(label,dtype=tf.float32)
    negatives = tf.math.reduce_sum(1.-y)
    positives = tf.math.reduce_sum(y)

    beta = negatives/(negatives + positives)
    pos_w = beta/(1-beta)
    cost = tf.nn.weighted_cross_entropy_with_logits(
    labels=label, logits=input, pos_weight=pos_w, name=None)
    cost = tf.reduce_sum(cost*(1-beta))
    return tf.where(tf.equal(positives, 0.0), 0.0, cost)


def pre_process_binary_cross_entropy(bc_loss,input, label,arg, use_tf_loss=False):
    # preprocess data
    y = label
    loss = 0
    w_loss=1.0
    preds = []
    for tmp_p in input:
        # tmp_p = input[i]

        # loss processing
        tmp_y = tf.cast(y, dtype=tf.float32)
        mask = tf.dtypes.cast(tmp_y > 0., tf.float32)
        b,h,w,c=mask.get_shape()
        positives = tf.math.reduce_sum(mask, axis=[1, 2, 3], keepdims=True)
        negatives = h*w*c-positives

        beta2 = positives / (negatives + positives) # negatives in hed
        beta = negatives/ (positives + negatives) # positives in hed
        pos_w = tf.where(tf.equal(y, 0.0), beta, beta2)
        logits = tf.sigmoid(tmp_p)

        l_cost = bc_loss(y_true=tmp_y, y_pred=logits,
                         sample_weight=pos_w)

        preds.append(logits)
        loss += (l_cost*w_loss)

    return preds, loss