import io
import json
import os
from datetime import datetime
from pathlib import Path

import keras
import pandas as pd
import pytz
import tensorflow as tf
from dotenv import load_dotenv
from keras import Model, layers, utils

load_dotenv(Path("./home.env"))
USER_HOME_DIR = os.getenv("USER_HOME_DIR") or Path.home()

# config
UPSCALE = 4
HR_PATCH = 128
LR_PATCH = HR_PATCH // UPSCALE
BATCH_SIZE = 16
AUTOTUNE = tf.data.AUTOTUNE
DATA_DIR = Path(USER_HOME_DIR) / Path("image_datum")
EPOCHS = 120
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

LOG_ROOT = Path("training_logs")
LOG_ROOT.mkdir(exist_ok=True)


# some scary training logger
# TODO: add sample eval per epoch
class TrainingMonitor(tf.keras.callbacks.Callback):
    def __init__(self, log_root=LOG_ROOT, save_freq_epochs=5, max_samples=3, sample_ds=None):
        super().__init__()
        self.sample_ds = sample_ds
        self.log_root = Path(log_root)
        self.save_freq_epochs = save_freq_epochs
        self.max_samples = max_samples
        self.metrics_csv = self.log_root / "metrics.csv"
        self.samples_dir = self.log_root / "samples"
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        if not self.metrics_csv.exists():
            with Path.open(self.metrics_csv, "w") as f:
                f.write("epoch,datetime,loss,val_loss,psnr_metric,val_psnr_metric,ssim_metric,val_ssim_metric\n")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        row = {
            "epoch": epoch + 1,
            "datetime": datetime.now(pytz.timezone("Poland")).isoformat(),
            "loss": logs.get("loss", ""),
            "val_loss": logs.get("val_loss", ""),
            "psnr_metric": logs.get("psnr_metric", ""),
            "val_psnr_metric": logs.get("val_psnr_metric", ""),
            "ssim_metric": logs.get("ssim_metric", ""),
            "val_ssim_metric": logs.get("val_ssim_metric", ""),
        }
        with Path.open(self.metrics_csv, "a") as f:
            f.write(
                ",".join(
                    str(row[k])
                    for k in [
                        "epoch",
                        "datetime",
                        "loss",
                        "val_loss",
                        "psnr_metric",
                        "val_psnr_metric",
                        "ssim_metric",
                        "val_ssim_metric",
                    ]
                )
                + "\n",
            )

        snapshot = {
            "epoch": epoch + 1,
            "logs": {k: float(v) if isinstance(v, (int, float)) else v for k, v in logs.items()},
        }
        with Path.open(self.log_root / "last_epoch_snapshot.json", "w") as f:
            json.dump(snapshot, f, indent=2)


def load_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    return tf.image.convert_image_dtype(image, tf.float32)


# random crop + downsampling
def make_lr_hr_pair(image):
    # check if bigger, then resize
    shape = tf.shape(image)[:2]
    h = shape[0]
    w = shape[1]

    min_size = tf.minimum(h, w)

    def resize_up():
        scale = tf.cast(HR_PATCH, tf.float32) / tf.cast(min_size, tf.float32)
        new_h = tf.cast(tf.round(tf.cast(h, tf.float32) * scale), tf.int32)
        new_w = tf.cast(tf.round(tf.cast(w, tf.float32) * scale), tf.int32)
        return tf.image.resize(image, [new_h, new_w], method="bicubic")

    image = tf.cond(min_size < HR_PATCH, resize_up, lambda: image)

    # random crop
    hr = tf.image.random_crop(image, size=[HR_PATCH, HR_PATCH, 3])
    # bicubic downsampling
    lr = tf.image.resize(hr, [LR_PATCH, LR_PATCH], method="bicubic")
    return lr, hr


# =======================
# augmentation functions
# =======================


def _gaussian_kernel(kernel_size: int, sigma: float | tf.Tensor):
    ax = tf.range(-(kernel_size // 2), (kernel_size // 2) + 1, dtype=tf.float32)
    xx, yy = tf.meshgrid(ax, ax)
    kernel = tf.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    kernel = kernel / tf.reduce_sum(kernel)
    return tf.cast(kernel, tf.float32)


def _maybe_blur(lr, prob=0.25):
    def do_blur():
        # random kernel + random sigma
        k = tf.cond(tf.random.uniform([]) < 0.5, lambda: 3, lambda: 5)
        sigma = tf.random.uniform([], 0.5, 1.5)
        kernel2d = _gaussian_kernel(k, sigma)
        c = tf.shape(lr)[-1]
        kernel4d = tf.reshape(kernel2d, [k, k, 1, 1])
        kernel4d = tf.tile(kernel4d, [1, 1, c, 1])
        lr_b = tf.expand_dims(lr, 0)
        blurred = tf.nn.depthwise_conv2d(lr_b, kernel4d, strides=[1, 1, 1, 1], padding="SAME")
        return tf.squeeze(blurred, 0)

    return tf.cond(tf.random.uniform([]) < prob, do_blur, lambda: lr)


def _maybe_noise(lr, prob=0.5, std=0.01):
    def do_noise():
        noise = tf.random.normal(tf.shape(lr), mean=0.0, stddev=std, dtype=tf.float32)
        return tf.clip_by_value(lr + noise, 0.0, 1.0)

    return tf.cond(tf.random.uniform([]) < prob, do_noise, lambda: lr)


def _maybe_jpeg(lr, prob=0.5, q_min=30, q_max=95):
    def do_jpeg():
        lr_u8 = tf.image.convert_image_dtype(lr, tf.uint8, saturate=True)
        quality = tf.random.uniform([], q_min, q_max + 1, dtype=tf.int32)
        enc = tf.io.encode_jpeg(lr_u8, format="rgb", quality=quality)
        dec = tf.io.decode_jpeg(enc, channels=3)
        return tf.image.convert_image_dtype(dec, tf.float32)

    return tf.cond(tf.random.uniform([]) < prob, do_jpeg, lambda: lr)


def augment(  # noqa: PLR0913
    lr,
    hr,
    brightness=0.06,
    contrast_range=(0.9, 1.1),
    saturation_range=(0.9, 1.1),
    blur_prob=0.25,
    noise_prob=0.5,
    noise_std=0.01,
    jpeg_prob=0.5,  # noqa: ARG001
    jpeg_qrange=(30, 95),  # noqa: ARG001
):
    # ---=== flip + rotation ===---
    lr = tf.image.random_flip_left_right(lr)
    hr = tf.image.random_flip_left_right(hr)
    lr = tf.image.random_flip_up_down(lr)
    hr = tf.image.random_flip_up_down(hr)
    k = tf.random.uniform([], 0, 4, dtype=tf.int32)
    lr = tf.image.rot90(lr, k)
    hr = tf.image.rot90(hr, k)

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
    lr = _maybe_blur(lr, prob=blur_prob)
    lr = _maybe_noise(lr, prob=noise_prob, std=noise_std)
    # TODO: maybe later
    # lr = _maybe_jpeg(lr, prob=jpeg_prob, q_min=jpeg_qrange[0], q_max=jpeg_qrange[1])

    # final clamp
    lr = tf.clip_by_value(lr, 0.0, 1.0)
    hr = tf.clip_by_value(hr, 0.0, 1.0)

    return lr, hr


def build_dataset(data_dir: Path, batch_size=BATCH_SIZE, *, training=True):
    data_dir = Path(data_dir)
    
    def get_from_dir(d:Path):
        return [
            str(p)
            for p in d.rglob("*")
            if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in ALLOWED_EXTS
        ]   # fmt: skip
    
    has_splits = False
    for child in data_dir.iterdir():
        if not child.is_dir():
            continue
        
        names = {p.name.lower() for p in child.iterdir() if p.is_dir()}
        if "train" in names or "test" in names:
            has_splits = True
            break
    
    img_files = []
    
    if has_splits:
        split_name = "train" if training else "test"
        for subset in data_dir.iterdir():
            if not subset.is_dir():
                continue
            split_dir = subset / split_name
            if not split_dir.exists():
                continue
            # gather files from each subset's train/test folder
            img_files.extend(
                str(p) for p in split_dir.rglob("*.*") if p.is_file() and not p.name.startswith(".")
            )
    else:
        # fallback
        img_files = [
            str(p)
            for p in data_dir.rglob("*.*")
                if p.is_file() and
                not p.name.startswith(".")
        ]  # fmt: skip

    if len(img_files) == 0:
        msg = f"No images in {data_dir!r}"
        raise ValueError(msg)

    ds = tf.data.Dataset.from_tensor_slices(img_files)
    if training:
        ds = ds.shuffle(len(img_files))
    ds = ds.map(lambda p: load_image(p), num_parallel_calls=AUTOTUNE)
    ds = ds.map(lambda img: make_lr_hr_pair(img), num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.map(lambda lr, hr: augment(lr, hr), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    return ds.prefetch(AUTOTUNE)


# simple u-net
def conv_block(x, filters, kernel_size=3, stride=1):
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding="same")(x)
    x = layers.BatchNormalization()(x)
    # should be better for cnns? [link](https://arxiv.org/abs/1502.01852?spm=a2ty_o01.29997173.0.0.2f78c921ujHBZJ&file=1502.01852)
    return layers.PReLU(shared_axes=[1, 2])(x)


def upsample_block(x, filters):
    # TODO: change to something else?
    x = layers.Conv2DTranspose(filters, 3, strides=2, padding="same")(x)
    return layers.PReLU(shared_axes=[1, 2])(x)


def build_unet_upscaler(lr_shape=(LR_PATCH, LR_PATCH, 3), upscale=UPSCALE):
    inp = layers.Input(shape=lr_shape)

    # encoder
    c1 = conv_block(inp, 64)
    c2 = conv_block(c1, 128)
    # TODO: research which pooling to use
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
    out = layers.Conv2D(3, 3, padding="same", activation="sigmoid")(out)

    # final resize to target
    # TODO: check if needed
    if upscale > 1:
        out = layers.UpSampling2D(size=(upscale, upscale), interpolation="bilinear")(out)  # 4x -> 32->128

    return Model(inp, out, name="first_unet")


def compile_and_train(train_ds, log_root=LOG_ROOT, val_ds=None):
    model = build_unet_upscaler()

    # metrics wrappers
    def psnr_metric(y_true, y_pred):
        return tf.image.psnr(y_true, y_pred, max_val=1.0)

    def ssim_metric(y_true, y_pred):
        return tf.image.ssim(y_true, y_pred, max_val=1.0)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=tf.keras.losses.MeanAbsoluteError(),   # TODO: change metrics for better psnr?
        metrics=[psnr_metric, ssim_metric],
    )
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            "unet_best.keras",
            save_best_only=True,
            monitor="val_loss" if val_ds else "loss",
        ),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss" if val_ds else "loss", factor=0.5, patience=6),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss" if val_ds else "loss",
            patience=12,
            restore_best_weights=True,
        ),
    ]

    csv_logger = tf.keras.callbacks.CSVLogger(str(log_root / "metrics.csv"), append=True)
    callbacks.append(csv_logger)

    # TensorBoard
    tb_dir = log_root / "tensorboard"
    callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=str(tb_dir), histogram_freq=1))

    monitor = TrainingMonitor(log_root=log_root, save_freq_epochs=5, max_samples=3)
    callbacks.append(monitor)

    history = model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds, callbacks=callbacks)

    (log_root / "history.json").write_text(json.dumps(history.history, indent=2))

    df = pd.DataFrame(history.history)
    df.index = df.index + 1
    df.index.name = "epoch"
    df.to_csv(log_root / "history.csv")

    model.save(str(log_root / "final_unet.keras"))

    return model, history


def upscale_image(model, image_path):
    img = load_image(image_path)  # 0..1 float32
    h, w = tf.shape(img)[0], tf.shape(img)[1]

    # resize to multiple HR_PATCH, then split to tiles
    lr = tf.image.resize(img, [h // UPSCALE, w // UPSCALE], method="bicubic")
    # pad/resize lr to input size
    lr_input = tf.image.resize(lr, [LR_PATCH, LR_PATCH], method="bicubic")
    lr_input = tf.expand_dims(lr_input, 0)
    pred_hr_patch = model.predict(lr_input)[0]
    # naive resize predicted patch to original hr shape
    return tf.image.resize(pred_hr_patch, [h, w], method="bicubic")


if __name__ == "__main__":
    train_ds = build_dataset(DATA_DIR, batch_size=BATCH_SIZE, training=True)
    model, history = compile_and_train(train_ds)
    model.save("first_unet_test.keras")

    # pass
