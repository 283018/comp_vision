import json
import random
from collections.abc import Iterator
from datetime import datetime
from enum import StrEnum
from pathlib import Path

import pytz
import tensorflow as tf

from config import cfg

from .augmentation import augment, make_lr_hr_pair, repeat_random_patches


class RunMode(StrEnum):
    TRAIN = "train"
    TEST = "test"
    EVAL = "eval"
    UNUSED = "unused"


def _rgb_to_lumi(img):
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    return 0.2989 * r + 0.5870 * g + 0.1140 * b


def _mgm(gray):
    x = tf.expand_dims(tf.expand_dims(gray, 0), -1)
    sobel = tf.image.sobel_edges(x)
    dx = sobel[..., 0]
    dy = sobel[..., 1]
    mag = tf.sqrt(tf.square(dx) + tf.square(dy))
    return tf.reduce_mean(mag)


def is_mono_patch(
    hr_patch,
    var_thresh=5e-4,
    grad_thresh=0.02,
):
    gray = _rgb_to_lumi(hr_patch)
    var = tf.math.reduce_variance(gray)
    grad_mean = _mgm(gray)

    is_low_var = var < var_thresh
    is_low_grad = grad_mean < grad_thresh
    return tf.logical_and(is_low_var, is_low_grad)


def _count_files(index_files, dataset_type: RunMode) -> int:
    return sum(1 for s in index_files.values() if s == f"{dataset_type}")


def _ensure_index(  # noqa: PLR0913, PLR0915
    index_path: Path,
    data_dir: Path,
    *,
    seed: int | None,
    run_mode: RunMode,
    split_ratios: tuple[float, float, float] | tuple[float, float],
    regenerate: bool,
) -> Path:
    # always load index if available
    index = {
        "files": {},
        "metadata": {},
    }
    if index_path.exists() and not regenerate:
        with Path.open(index_path) as f:
            index = json.load(f)

    # get current files from dir
    current_files = set()
    for p in data_dir.rglob("*"):
        if p.is_file() and not p.name.startswith("."):
            if p.suffix.lower() not in cfg.ALLOWED_EXTS:
                continue
            try:
                rel_path = str(p.relative_to(data_dir))
                current_files.add(rel_path)
            except ValueError:
                continue

    # all unused at start
    for file in current_files:
        if file not in index["files"]:
            index["files"][file] = str(RunMode.UNUSED)

    # remove files that are no longer exits
    for file in list(index["files"].keys()):
        if file not in current_files:
            del index["files"][file]

    # train mode
    if run_mode == RunMode.TRAIN or regenerate:
        unused_files = [
            f for f, state in index["files"].items()
            if state == str(RunMode.UNUSED)
            ]  # fmt: skip

        if unused_files:
            random.shuffle(unused_files)
            total_unused = len(unused_files)

            split_count = []
            assigned = 0
            for i, ratio in enumerate(split_ratios):
                if i == len(split_ratios) - 1:
                    count = total_unused - assigned
                else:
                    count = round(total_unused * ratio)
                    assigned += count
                split_count.append(count)

            # assign files to splits
            start_idx = 0
            # me, from the past: WHY?
            split_modes = [
                RunMode.TRAIN,
                RunMode.TEST,
                RunMode.EVAL if len(split_ratios) == 3 else None,
            ][: len(split_ratios)]

            for count, mode in zip(split_count, split_modes, strict=True):
                if count <= 0 or mode is None:
                    continue
                end_idx = start_idx + count
                for file in unused_files[start_idx:end_idx]:
                    index["files"][file] = str(mode)
                start_idx = end_idx
    elif run_mode == RunMode.TEST:
        unused_files = [
            f for f, state in index["files"].items()
            if state == str(RunMode.UNUSED)
        ]  # fmt: skip
        if unused_files:
            for file in unused_files:
                index["files"][file] = str(RunMode.TEST)

    elif run_mode == RunMode.EVAL:
        unused_files = [
            f for f, state in index["files"].items()
            if state == str(RunMode.UNUSED)
        ]  # fmt: skip
        if unused_files:
            for file in unused_files:
                index["files"][file] = str(RunMode.EVAL)
    else:
        msg = f"Unknown run_mode after guards: {run_mode}"
        raise RuntimeError(msg)

    # metadata update
    index["metadata"] = {
        "split_ratios": split_ratios,
        "seed": seed,
        "generated_at": datetime.now(pytz.timezone("Poland")).isoformat(),
        "data_dir": str(data_dir.resolve()),
        "total_files": len(index["files"]),
        "train_count": _count_files(index["files"], RunMode.TRAIN),
        "test_count": _count_files(index["files"], RunMode.TEST),
        "eval_count": _count_files(index["files"], RunMode.EVAL),
        "unused_count": _count_files(index["files"], RunMode.UNUSED),
    }

    # save index
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with Path.open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    return index_path


def image_iterator(  # noqa: PLR0913
    run_mode: RunMode,
    split_ratios: tuple[float, float, float] | tuple[float, float] = (0.8, 0.2),
    *,
    data_dir: Path,
    index_path: Path,
    seed: int | None = None,
    regenerate_index: bool = False,
) -> Iterator[str]:
    random.seed(seed)

    msgs = []
    if not (2 <= len(split_ratios) <= 3):
        msgs.append("split_ratios must contain 2 or 3 values")
    if not (0.99 <= sum(split_ratios) <= 1.01):  # floating point error margin
        msgs.append(f"Split ratios must sum to 1.0, got {sum(split_ratios)}")
    if run_mode == RunMode.UNUSED:
        msgs.append(f"{RunMode.UNUSED} cannot be used as option for dataset collection.")
    if msgs:
        msg = "\n" + "\n".join(msgs)
        raise ValueError(msg)

    if not isinstance(run_mode, RunMode):
        valid_modes = ", ".join(mode.value for mode in RunMode if mode != RunMode.UNUSED)
        msg = f"Unknown run mode: {run_mode}, must be one of: {valid_modes}"
        raise TypeError(msg)

    index_path = _ensure_index(
        index_path=index_path,
        data_dir=data_dir,
        seed=seed,
        run_mode=run_mode,
        split_ratios=split_ratios,
        regenerate=regenerate_index,
    )

    with Path.open(index_path) as f:
        index = json.load(f)

    target_state = str(run_mode)
    for rel_path, state in index["files"].items():
        if state == target_state:
            abs_path: Path = data_dir / rel_path
            if abs_path.exists():
                yield str(abs_path.resolve())
            else:
                msg = f"Skipping missing file: {abs_path}"
                tf.get_logger().warning(msg)


def build_dataset(  # noqa: PLR0913
    run_mode: RunMode,
    data_dir: Path = cfg.DATA_DIR,
    batch_size=cfg.BATCH_SIZE,
    patches_per_image=1,
    *,
    index_path: Path = cfg.FILEINDEX_PATH if cfg.FILEINDEX_PATH else cfg.DATA_DIR / "index.json",
    seed: int | None,
    split_ratios: tuple[float, float] | tuple[float, float, float] = (0.8, 0.2),
    regenerate_index: bool = False,
) -> tuple[tf.data.Dataset[tuple[tf.Tensor, tf.Tensor]], int]:
    _ensure_index(
        index_path=index_path,
        data_dir=data_dir,
        seed=seed,
        run_mode=run_mode,
        split_ratios=split_ratios,
        regenerate=regenerate_index,
    )

    def gen():
        yield from image_iterator(
            data_dir=data_dir,
            index_path=index_path,
            seed=seed,
            run_mode=run_mode,
            split_ratios=split_ratios,
            regenerate_index=regenerate_index,
        )

    with index_path.open() as f:
        index = json.load(f)
    total_count = _count_files(index["files"], run_mode)

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=tf.TensorSpec((), tf.string),
    )

    # dataset processing
    if run_mode == RunMode.TRAIN:
        ds = ds.shuffle(total_count)
    ds = ds.map(lambda p: load_image(p), num_parallel_calls=cfg.AUTOTUNE)

    ds = ds.ignore_errors(log_warning=True)

    if run_mode == RunMode.TRAIN:
        ds = repeat_random_patches(ds, patches_per_image=patches_per_image)

        # filtering to monotonic patches
        var_thresh = 5e-4
        grad_thresh = 0.02
        keep_mono_prob = 0.1

        def _keep_fn(_lr, hr):
            mono = is_mono_patch(hr, var_thresh=var_thresh, grad_thresh=grad_thresh)
            return tf.logical_or(
                tf.logical_not(mono),
                tf.random.uniform([], 0.0, 1.0) < keep_mono_prob,
            )

        ds = ds.filter(_keep_fn) # type: ignore

        ds = ds.map(lambda lr, hr: augment(lr, hr), num_parallel_calls=cfg.AUTOTUNE)
        total_samples = total_count * patches_per_image
    else:
        ds = ds.map(lambda img: make_lr_hr_pair(img), num_parallel_calls=cfg.AUTOTUNE)
        total_samples = total_count

    if run_mode == RunMode.TRAIN:
        ds = ds.repeat()

    ds = ds.batch(batch_size)
    ds = ds.prefetch(cfg.AUTOTUNE)
    return ds, total_samples


def load_image(img_path: Path):
    image = tf.io.read_file(img_path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    return tf.image.convert_image_dtype(image, tf.float32)
