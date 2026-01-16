# data/augmentation.py
import numpy as np
import tensorflow as tf

from config import cfg


def do_decide(prob: float = 0.5) -> tf.Tensor:
    """
    Decides whether to return `True` or `False`, depending on whether random number is less than provided.
    
    :param prob: prob threshold.
    :return: tensor with `True` or `False`.
    """
    return tf.random.uniform([], 0.0, 1.0) < prob
    

def _gaussian_kernel(kernel_size: int, sigma: tf.Tensor):
    ax = tf.range(-(kernel_size // 2), (kernel_size // 2) + 1, dtype=tf.float32)
    xx, yy = tf.meshgrid(ax, ax)
    kernel = tf.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    kernel = kernel / tf.reduce_sum(kernel)
    return tf.cast(kernel, tf.float32)


def _maybe_blur(lr, prob, blur_intensity):
    def do_blur():
        # random kernel + random sigma
        k = tf.cond(tf.random.uniform([]) < 0.5, lambda: 3, lambda: 5)
        sigma = tf.random.uniform([], 0.1, blur_intensity)
        kernel2d = _gaussian_kernel(k, sigma)
        c = tf.shape(lr)[-1]
        kernel4d = tf.reshape(kernel2d, [k, k, 1, 1])
        kernel4d = tf.tile(kernel4d, [1, 1, c, 1])
        lr_b = tf.expand_dims(lr, 0)
        blurred = tf.nn.depthwise_conv2d(lr_b, kernel4d, strides=[1, 1, 1, 1], padding="SAME")
        return tf.squeeze(blurred, 0)

    return tf.cond(do_decide(prob), do_blur, lambda: lr)


def _maybe_noise(lr, prob, std):
    def do_noise():
        noise = tf.random.normal(tf.shape(lr), mean=0.0, stddev=std, dtype=tf.float32)
        return tf.clip_by_value(lr + noise, 0.0, 1.0)

    return tf.cond(do_decide(prob), do_noise, lambda: lr)


def _maybe_jpeg(lr, prob, q_min, q_max):
    def do_jpeg():
        lr_u8 = tf.image.convert_image_dtype(lr, tf.uint8, saturate=True)
        quality = tf.random.uniform([], q_min, q_max + 1, dtype=tf.int32)
        enc = tf.io.encode_jpeg(lr_u8, format="rgb", quality=int(quality))
        dec = tf.io.decode_jpeg(enc, channels=3)
        return tf.image.convert_image_dtype(dec, tf.float32)

    return tf.cond(do_decide(prob), do_jpeg, lambda: lr)


def augment(  # noqa: PLR0913
    lr,
    hr,
    *,
    lr_flip_prob: float = 0.5,
    ud_flip_prob: float = 0.5,
    rot90_prob:float = 0.75,
    brightness:float = 0.1,
    contrast_range:tuple[float, float] = (0.7, 1.3),
    saturation_range:tuple[float, float] = (0.7, 1.3),
    blur_prob:float = 0.4,
    blur_intensity:float = 0.4,
    noise_prob:float = 0.4,
    noise_std:float=0.03,
    jpeg_prob:float=0.0,
    jpeg_qrange:tuple[int, int] = (30, 95),  # noqa: ARG001
) -> tuple[tf.Tensor, tf.Tensor]:
    """
    :param lr: low-res image.
    :param hr: high-res image.
    :param lr_flip_prob: prob of flipping pair of images vertically.
    :param ud_flip_prob: prob of flipping pair of images horizontally.
    :param rot90_prob: prob of applying rotations by multiples of 90 degrees to pair.
    :param brightness: half-length of range for brightness adjustment for both images. If positive, will randomly adjust brightness within this range.
    :param contrast_range: range of values for random contrast adjustment.
    :param saturation_range: range od values for random saturation adjustment.
    :param blur_prob: prob of applying gaussian **blur** to low-res image.
    :param blur_intensity: maxval for random blur intensity, applied to sigma of gaussian kernel.
    :param noise_prob: prob of adding Gaussian **noise** to low-res image.
    :param noise_std: std of **noise**.
    :param jpeg_prob: prob of applying jpeg compression to pair. Disabled by default.
    :param jpeg_qrange: range of values for jpeg compression quality. Disabled by default.
    
    :return: tuple containing augmented low-res and high-res images as tensors.
    """
    # ---=== flip + rotation ===---
    do_lr_flip = do_decide(lr_flip_prob)
    lr, hr = tf.cond(
        do_lr_flip,
        lambda: (tf.image.flip_left_right(lr), tf.image.flip_left_right(hr)),
        lambda: (lr, hr),
    )

    do_ud_flip = do_decide(ud_flip_prob)
    lr, hr = tf.cond(
        do_ud_flip,
        lambda: (tf.image.flip_up_down(lr), tf.image.flip_up_down(hr)),
        lambda: (lr, hr),
    )

    do_rot90 = do_decide(rot90_prob)
    k = tf.random.uniform([], 1, 3, dtype=tf.int32)
    lr = tf.cond(do_rot90, lambda: tf.image.rot90(lr, k), lambda: lr)
    hr = tf.cond(do_rot90, lambda: tf.image.rot90(hr, k), lambda: hr)

    # # ---=== jitter - motion simulation ===---
    if brightness and brightness > 0:
        b = tf.random.uniform([], -brightness, brightness)
        lr = tf.image.adjust_brightness(lr, b)
        hr = tf.image.adjust_brightness(hr, b)
    c = tf.random.uniform([], contrast_range[0], contrast_range[1])
    lr = tf.image.adjust_contrast(lr, c)
    hr = tf.image.adjust_contrast(hr, c)
    s = tf.random.uniform([], saturation_range[0], saturation_range[1])
    lr = tf.image.adjust_saturation(lr, s)
    hr = tf.image.adjust_saturation(hr, s)

    lr = tf.clip_by_value(lr, 0.0, 1.0)
    hr = tf.clip_by_value(hr, 0.0, 1.0)

    # lr degradation
    lr = _maybe_blur(lr, prob=blur_prob, blur_intensity=blur_intensity)
    lr = _maybe_noise(lr, prob=noise_prob, std=noise_std)
    #* disabled by default
    lr = _maybe_jpeg(lr, prob=jpeg_prob, q_min=jpeg_qrange[0], q_max=jpeg_qrange[1])

    # final clamp
    lr = tf.clip_by_value(lr, 0.0, 1.0)
    hr = tf.clip_by_value(hr, 0.0, 1.0)

    return lr, hr


def make_lr_hr_pair(image):
    # check if bigger, then resize
    shape = tf.shape(image)[:2]
    h = shape[0]
    w = shape[1]

    min_size = tf.minimum(h, w)

    def resize_up():
        scale = tf.cast(cfg.HR_PATCH, tf.float32) / tf.cast(min_size, tf.float32)
        new_h = tf.cast(tf.round(tf.cast(h, tf.float32) * scale), tf.int32)
        new_w = tf.cast(tf.round(tf.cast(w, tf.float32) * scale), tf.int32)
        return tf.image.resize(image, [new_h, new_w], method="bicubic")

    image = tf.cond(min_size < cfg.HR_PATCH, resize_up, lambda: image)

    # random crop
    hr = tf.image.random_crop(image, size=[cfg.HR_PATCH, cfg.HR_PATCH, 3])
    # bicubic downsampling
    lr = tf.image.resize(hr, [cfg.LR_PATCH, cfg.LR_PATCH], method="bicubic")
    return lr, hr


def repeat_random_patches(ds_images, patches_per_image=4):
    return ds_images.flat_map(
        lambda img: tf.data.Dataset.from_tensors(img)
        .repeat(patches_per_image)
        .map(lambda i: make_lr_hr_pair(i), num_parallel_calls=cfg.AUTOTUNE),
    )
