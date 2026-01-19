import keras
import tensorflow as tf
from keras import Model, layers, mixed_precision, utils

from config import cfg


# simple u-net
def conv_block(x, filters, kernel_size=3, stride=1):
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding="same")(x)
    x = layers.BatchNormalization()(x)
    # should be better for cnns? [link](https://arxiv.org/abs/1502.01852?spm=a2ty_o01.29997173.0.0.2f78c921ujHBZJ&file=1502.01852)
    return layers.PReLU(shared_axes=[1, 2])(x)


def upsample_block(x, filters):
    x = layers.Conv2DTranspose(filters, 3, strides=2, padding="same")(x)
    return layers.PReLU(shared_axes=[1, 2])(x)


def build_unet_upscaler(lr_shape=(cfg.LR_PATCH, cfg.LR_PATCH, 3), upscale=cfg.UPSCALE):
    inp = layers.Input(shape=lr_shape)

    # encoder
    c1 = conv_block(inp, 64)
    c2 = conv_block(c1, 128)
    p1 = layers.MaxPool2D(2)(c2)

    c3 = conv_block(p1, 256)
    p2 = layers.MaxPool2D(2)(c3)

    # bottleneck
    b = conv_block(p2, 512)

    # decoder
    u1 = upsample_block(b, 256)
    u1 = layers.Concatenate()([u1, c3])
    u1 = conv_block(u1, 256)

    u2 = upsample_block(u1, 128)
    u2 = layers.Concatenate()([u2, c2])
    u2 = conv_block(u2, 128)

    # spatial size = lr x lr
    out = layers.Conv2D(64, 3, padding="same")(u2)
    out = layers.PReLU(shared_axes=[1, 2])(out)
    out = layers.Conv2D(3, 3, padding="same")(out)
    out = layers.Activation("sigmoid", dtype="float32")(out)    # explicit activation with casting

    # final resize to target
    if upscale > 1:
        out = layers.UpSampling2D(size=(upscale, upscale), interpolation="bilinear")(out)  # 4x -> 32->128

    return Model(inp, out, name="first_unet")
