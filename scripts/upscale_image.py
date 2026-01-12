import tensorflow as tf

from config import cfg
from data import load_image


def upscale_image(model, image_path):
    img = load_image(image_path)  # 0..1 float32
    h, w = tf.shape(img)[0], tf.shape(img)[1]

    # resize to multiple HR_PATCH, then split to tiles
    lr = tf.image.resize(img, [h // cfg.UPSCALE, w // cfg.UPSCALE], method="bicubic")
    # pad/resize lr to input size
    lr_input = tf.image.resize(lr, [cfg.LR_PATCH, cfg.LR_PATCH], method="bicubic")
    lr_input = tf.expand_dims(lr_input, 0)
    pred_hr_patch = model.predict(lr_input)[0]
    # naive resize predicted patch to original hr shape
    return tf.image.resize(pred_hr_patch, [h, w], method="bicubic")
