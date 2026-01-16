from pathlib import Path

import tensorflow as tf

from config import cfg

from .augmentation import augment, make_lr_hr_pair, repeat_random_patches

# TODO: create actual arguments.


def build_dataset(
    data_dir: Path = cfg.DATA_DIR,
    batch_size=cfg.BATCH_SIZE,
    patches_per_image=1,
    *,
    training=True,
) -> tuple[tf.data.Dataset[tuple[tf.Tensor, tf.Tensor]], int]:
    data_dir = Path(data_dir)

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
            # gather files subdirs for train.test
            img_files.extend(str(p) for p in split_dir.rglob("*.*") if p.is_file() and not p.name.startswith("."))
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

    num_images = len(img_files)

    ds = tf.data.Dataset.from_tensor_slices(img_files)
    if training:
        ds = ds.shuffle(len(img_files))
    ds = ds.map(lambda p: load_image(p), num_parallel_calls=cfg.AUTOTUNE)

    ds = ds.ignore_errors(log_warning=True)

    if training:
        ds = repeat_random_patches(ds, patches_per_image=patches_per_image)
        ds = ds.map(lambda lr, hr: augment(lr, hr), num_parallel_calls=cfg.AUTOTUNE)
        total_samples = num_images * patches_per_image
    else:
        ds = ds.map(lambda img: make_lr_hr_pair(img), num_parallel_calls=cfg.AUTOTUNE)
        total_samples = num_images

    if training:
        ds = ds.repeat()

    ds = ds.batch(batch_size)
    ds = ds.prefetch(cfg.AUTOTUNE)
    return ds, total_samples


def load_image(img_path: Path):
    image = tf.io.read_file(str(img_path))
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    return tf.image.convert_image_dtype(image, tf.float32)
