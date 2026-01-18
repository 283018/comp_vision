import keras
import tensorflow as tf
from keras import Model, layers
from keras.applications import VGG19


# TODO!: current PixelShuffle cannot be serialized, so must be preserved on runtime
class PixelShuffle(layers.Layer):
    def __init__(self, scale, **kwargs):
        super().__init__(**kwargs)
        self.scale = int(scale)

    def call(self, x):
        return tf.nn.depth_to_space(x, self.scale)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"scale": self.scale})
        return cfg

def residual_block(x_in, filters, kernel_size=3):
    x = layers.Conv2D(filters, kernel_size, padding="same")(x_in)
    x = layers.BatchNormalization()(x)
    x = layers.PReLU(shared_axes=[1, 2])(x)
    x = layers.Conv2D(filters, kernel_size, padding="same")(x)
    x = layers.BatchNormalization()(x)
    return layers.Add()([x_in, x])


def upsample_pixelshuffle(x_in, filters, scale=2):
    x = layers.Conv2D(filters * (scale**2), 3, padding="same")(x_in)
    x = PixelShuffle(scale=scale)(x)
    return layers.PReLU(shared_axes=[1, 2])(x)


def build_generator(lr_shape=(32, 32, 3), num_res_blocks=12, upscale=4):
    inp = layers.Input(shape=lr_shape)
    x = layers.Conv2D(64, 9, padding="same")(inp)
    x = layers.PReLU(shared_axes=[1, 2])(x)
    skip = x

    for _ in range(num_res_blocks):
        x = residual_block(x, 64)

    x = layers.Conv2D(64, 3, padding="same")(x)
    
    # TODO?: try tanh?
    x = layers.BatchNormalization()(x)  # in theory can decrease quality, but help training stability (more important?)
    x = layers.Add()([x, skip])

    # upsampling
    n_up = {2: 1, 4: 2, 8: 3}.get(upscale, 2)
    for _ in range(n_up):
        x = upsample_pixelshuffle(x, 64, scale=2)

    x = layers.Conv2D(3, 9, padding="same")(x)
    out = layers.Activation("sigmoid", dtype="float32")(x)
    return Model(inp, out, name="generator_resnet")


# discriminator blocks
def disc_block(x, filters, kernel_size=3, strides=1, *, batchnorm=True):
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding="same")(x)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    return layers.LeakyReLU(0.2)(x)


# full image discriminator, instead of per-patch, may reduce local texture quality
def build_discriminator(hr_shape=(128, 128, 3)):
    inp = layers.Input(shape=hr_shape)
    x = layers.Conv2D(64, 3, strides=1, padding="same")(inp)
    x = layers.LeakyReLU(0.2)(x)    # smoother training + used in most implementations

    x = disc_block(x, 64, strides=2, batchnorm=False)   # should stabilize training
    x = disc_block(x, 128, strides=1)
    x = disc_block(x, 128, strides=2)
    x = disc_block(x, 256, strides=1)
    x = disc_block(x, 256, strides=2)
    x = disc_block(x, 512, strides=1)
    x = disc_block(x, 512, strides=2)

    x = layers.Flatten()(x)
    x = layers.Dense(1024)(x)
    x = layers.LeakyReLU(0.2)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return Model(inp, out, name="discriminator")


# VGG fe for perceptual loss, focusing on general image, instead of local textures
def build_vgg_feature_extractor(layer_name="block5_conv4", hr_shape=(128, 128, 3)):
    vgg = VGG19(include_top=False, weights="imagenet", input_shape=hr_shape)
    vgg.trainable = False
    outputs = vgg.get_layer(layer_name).output
    return Model(vgg.input, outputs)
