
# WARNING:
#   Hic sunt dracones...


import json
import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import tensorflow as tf
from dotenv import load_dotenv
from PIL import Image
from tensorflow import keras

UPSCALE = 4
HR_PATCH = 128
LR_PATCH = HR_PATCH // UPSCALE
PATCHES_PER_IMAGE = 5
SAVE_IMAGES_TOTAL = 250
SAVE_IMAGE_PROB = 0.005
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

load_dotenv(Path("./home.env"))
USER_HOME_DIR = os.getenv("USER_HOME_DIR") or Path.home()
DATA_DIR = Path(USER_HOME_DIR) / "image_data"
LOG_ROOT = Path("eval_logs")
LOG_ROOT.mkdir(exist_ok=True, parents=True)

CHUNK_SIZE = 32
PRED_BATCH_SIZE = 8


def guard_unknown_dim[TDim: (int, float)](dim: TDim | None) -> TDim:
    if dim is None:
        msg = "Tensor dim is unknown (None) - not convertible to int"
        raise ValueError(msg)
    return dim


def psnr_metric(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=1.0)


def ssim_metric(y_true, y_pred):
    return tf.image.ssim(y_true, y_pred, max_val=1.0)


def load_image(path: Path) -> tf.Tensor:
    with Image.open(path) as im:
        img = im.convert("RGB")
        arr = np.asarray(img, dtype=np.uint8)
    arr = arr.astype(np.float32) / 255.0
    return tf.convert_to_tensor(arr, dtype=tf.float32)


def pad_or_crop_to(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)
    # ensure 3 channels
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    h, w = arr.shape[:2]
    # if same shape, return copy (float32)
    if h == target_h and w == target_w:
        return arr.astype(np.float32)

    # pad if smaller
    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)
    if pad_h > 0 or pad_w > 0:
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_width = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
        arr = np.pad(arr, pad_width, mode="edge")
        h, w = arr.shape[:2]

    # crop center if larger
    if arr.shape[0] > target_h or arr.shape[1] > target_w:
        start_y = max(0, (arr.shape[0] - target_h) // 2)
        start_x = max(0, (arr.shape[1] - target_w) // 2)
        arr = arr[start_y : start_y + target_h, start_x : start_x + target_w, :]

    return arr.astype(np.float32)


def save_image_np(img: np.ndarray, path: Path) -> None:
    img_u8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(img_u8).save(str(path))


def collect_test_files(data_dir: Path, fraction: float = 1.0) -> list[Path]:
    files: list[Path] = []
    per_dir = {}

    for child in data_dir.iterdir():
        if not child.is_dir():
            continue
        test_dir = child / "test"
        if test_dir.exists():
            lst = [
                p
                for p in test_dir.rglob("*.*")
                if p.is_file() and p.suffix.lower() in ALLOWED_EXTS and not p.name.startswith(".")
            ]
            if lst:
                per_dir[child] = sorted(lst)

    if per_dir:
        for lst in per_dir.values():
            if fraction >= 1.0:
                chosen = lst
            elif fraction <= 0.0:
                chosen = []
            else:
                k = max(1, int(len(lst) * fraction)) if len(lst) > 0 else 0
                chosen = random.sample(lst, k) if k < len(lst) else lst
            files.extend(chosen)
        return files

    all_files = [
        p
        for p in data_dir.rglob("*.*")
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTS and not p.name.startswith(".")
    ]
    if not all_files:
        return []

    if fraction >= 1.0:
        return sorted(all_files)

    if fraction <= 0.0:
        return []

    k = max(1, int(len(all_files) * fraction))
    sampled = random.sample(all_files, k) if k < len(all_files) else all_files
    return sorted(sampled)


def bicubic_upscale_from_hr(hr: tf.Tensor, upscale=UPSCALE) -> tf.Tensor:
    h = int(guard_unknown_dim(hr.shape[0]))
    w = int(guard_unknown_dim(hr.shape[1]))
    lr_h = math.ceil(h / upscale)
    lr_w = math.ceil(w / upscale)
    lr = tf.image.resize(hr, [lr_h, lr_w], method="bicubic")
    return tf.image.resize(lr, [h, w], method="bicubic")


# left as fallback
def predict_full_image_tiled(  # noqa: PLR0913
    model: keras.Model,
    hr_img: tf.Tensor,
    *,
    upscale=UPSCALE,
    lr_patch=LR_PATCH,
    hr_patch=HR_PATCH,
    batch_size=16,
) -> np.ndarray:
    hr_h = int(guard_unknown_dim(hr_img.shape[0]))
    hr_w = int(guard_unknown_dim(hr_img.shape[1]))

    lr_h = math.ceil(hr_h / upscale)
    lr_w = math.ceil(hr_w / upscale)

    lr_img = tf.image.resize(hr_img, [lr_h, lr_w], method="bicubic")
    lr_img = tf.clip_by_value(lr_img, 0.0, 1.0)

    pad_h = (math.ceil(lr_h / lr_patch) * lr_patch) - lr_h
    pad_w = (math.ceil(lr_w / lr_patch) * lr_patch) - lr_w

    lr_padded = (
        tf.pad(
            lr_img,
            [[0, pad_h], [0, pad_w], [0, 0]],
            mode="REFLECT",
        )
        if pad_h > 0 or pad_w > 0
        else lr_img
    )

    hp = int(guard_unknown_dim(lr_padded.shape[0]))
    wp = int(guard_unknown_dim(lr_padded.shape[1]))

    tiles = []
    coords = []
    for y in range(0, hp, lr_patch):
        for x in range(0, wp, lr_patch):
            patch = lr_padded[y : y + lr_patch, x : x + lr_patch, :]
            tiles.append(patch.numpy())
            coords.append((y, x))

    tiles = np.stack(tiles, axis=0).astype(np.float32)

    # predict in batches
    preds = []
    for i in range(0, tiles.shape[0], batch_size):
        batch = tiles[i : i + batch_size]
        pred_batch = model.predict(batch, verbose=1)
        preds.append(pred_batch)
    preds = np.concatenate(preds, axis=0)

    out_h = math.ceil(lr_h / lr_patch) * hr_patch
    out_w = math.ceil(lr_w / lr_patch) * hr_patch
    canvas = np.zeros((out_h, out_w, 3), dtype=np.float32)

    for idx, (y, x) in enumerate(coords):
        y_hr = (y // lr_patch) * hr_patch
        x_hr = (x // lr_patch) * hr_patch
        canvas[y_hr : y_hr + hr_patch, x_hr : x_hr + hr_patch, :] = preds[idx]

    # crop to exact hr_h hr_w
    canvas = canvas[:hr_h, :hr_w, :]

    return np.clip(canvas, 0.0, 1.0)


# --- metrics ---
def compute_metrics_pair(hr: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    hr_tf = tf.convert_to_tensor(hr, dtype=tf.float32)
    pred_tf = tf.convert_to_tensor(pred, dtype=tf.float32)
    psnr_v = float(tf.image.psnr(hr_tf, pred_tf, max_val=1.0).numpy())
    ssim_v = float(tf.image.ssim(hr_tf, pred_tf, max_val=1.0).numpy())
    mae_v = float(tf.reduce_mean(tf.abs(hr_tf - pred_tf)).numpy())
    return psnr_v, ssim_v, mae_v


def eval_on_testset(  # noqa: PLR0913, PLR0915
    model_path: Path,
    data_dir: Path = DATA_DIR,
    log_root: Path = LOG_ROOT,
    save_examples=16,  # noqa: ARG001
    chunk_size=CHUNK_SIZE,
    pred_batch_size=PRED_BATCH_SIZE,
    fraction: float = 1.0,
):
    fraction = float(max(0.0, min(1.0, fraction)))

    model = keras.models.load_model(
        str(model_path),
        custom_objects={"psnr_metric": psnr_metric, "ssim_metric": ssim_metric},
    )

    test_files = collect_test_files(data_dir, fraction=fraction)
    if not test_files:
        msg = f"No test images found in {data_dir!r}"
        raise ValueError(msg)

    examples_dir = Path(log_root) / "samples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    example_idx = 0

    saved_images_count = 0

    def _process_chunk(chunk_files, global_img_idx_offset):  # noqa: ARG001, PLR0915
        nonlocal example_idx, saved_images_count
        tiles = []
        tiles_meta = []  # (local_img_idx, y, x)
        infos = []       # per chunk
        sample_patches = {}  # local_idx : list [bic_patch, pred_patch, hr_patch, diff_model, diff_bic]

        # tiles and metadata
        for local_idx, p in enumerate(chunk_files):
            try:
                hr_tf = load_image(p)
                hr_np = hr_tf.numpy()
                hr_h = int(hr_np.shape[0])
                hr_w = int(hr_np.shape[1])

                lr_h = max(1, math.ceil(hr_h / UPSCALE))
                lr_w = max(1, math.ceil(hr_w / UPSCALE))

                lr_img = tf.image.resize(hr_tf, [lr_h, lr_w], method="bicubic")
                lr_img = tf.clip_by_value(lr_img, 0.0, 1.0)

                if lr_h < LR_PATCH or lr_w < LR_PATCH:
                    new_h = max(LR_PATCH, lr_h)
                    new_w = max(LR_PATCH, lr_w)
                    lr_img = tf.image.resize(lr_img, [new_h, new_w], method="bicubic")
                    lr_h, lr_w = new_h, new_w

                pad_h = (math.ceil(lr_h / LR_PATCH) * LR_PATCH) - lr_h
                pad_w = (math.ceil(lr_w / LR_PATCH) * LR_PATCH) - lr_w
                lr_padded = (
                    tf.pad(
                        lr_img,
                        [[0, pad_h], [0, pad_w], [0, 0]],
                        mode="REFLECT",
                    )
                    if pad_h > 0 or pad_w > 0
                    else lr_img
                )

                lr_p_h = int(guard_unknown_dim(lr_padded.shape[0]))
                lr_p_w = int(guard_unknown_dim(lr_padded.shape[1]))

                to_save = bool(
                    (SAVE_IMAGES_TOTAL is None or saved_images_count < SAVE_IMAGES_TOTAL)
                    and (random.random() < SAVE_IMAGE_PROB),
                )

                sample_patches[local_idx] = [] if to_save else None

                for y in range(0, lr_p_h, LR_PATCH):
                    for x in range(0, lr_p_w, LR_PATCH):
                        y_hr = (y // LR_PATCH) * HR_PATCH
                        x_hr = (x // LR_PATCH) * HR_PATCH

                        # skip on border
                        if (y_hr + HR_PATCH > hr_h) or (x_hr + HR_PATCH > hr_w):
                            # TODO: print
                            # print(
                            #     f"[skip-border] skipping border tile for file={p} at LR (y={y},x={x}) -> "
                            #     f"HR (y_hr={y_hr},x_hr={x_hr}) would exceed HR size ({hr_h},{hr_w})",
                            # )
                            continue

                        patch = lr_padded[y : y + LR_PATCH, x : x + LR_PATCH, :].numpy().astype(np.float32)
                        if patch.shape != (LR_PATCH, LR_PATCH, 3):
                            # TODO: print
                            # print(f"[skip] skipping tile for file={p} at (y={y},x={x}) with shape={patch.shape}")
                            continue

                        tiles.append(patch)
                        tiles_meta.append((local_idx, y, x))

                bic = tf.image.resize(
                    tf.image.resize(hr_tf, [lr_h, lr_w], method="bicubic"),
                    [hr_h, hr_w],
                    method="bicubic",
                ).numpy()

                infos.append(
                    {
                        "path": str(p),
                        "hr_np": hr_np,
                        "hr_h": hr_h,
                        "hr_w": hr_w,
                        "lr_h": lr_h,
                        "lr_w": lr_w,
                        "lr_p_h": lr_p_h,
                        "lr_p_w": lr_p_w,
                        "pad_h": pad_h,
                        "pad_w": pad_w,
                        "bic_np": bic,
                    },
                )
            except Exception as ex:  # noqa: BLE001
                print(f"failed to prepare tiles for {p}: {ex}")
                infos.append({"path": str(p), "failed": True})
                continue

        if len(tiles) == 0:
            return []

        # normalize before stacking
        target_tile_h = LR_PATCH
        target_tile_w = LR_PATCH
        normalized_tiles = []
        bad_idx = None
        for idx, t in enumerate(tiles):
            try:
                t_np = t if isinstance(t, np.ndarray) else np.asarray(t)
                if t_np.ndim == 2:
                    t_np = np.stack([t_np] * 3, axis=-1)
                if t_np.shape[:2] != (target_tile_h, target_tile_w) or t_np.shape[2] != 3:
                    print(f"[debug] normalizing tile {idx}: {t_np.shape} -> {(target_tile_h, target_tile_w, 3)}")
                t_norm = pad_or_crop_to(t_np, target_tile_h, target_tile_w)
                if t_norm.shape != (target_tile_h, target_tile_w, 3):
                    bad_idx = idx
                    print(f"[error] from shape in tile normalization for {idx}: {t_norm.shape}")
                normalized_tiles.append(t_norm.astype(np.float32))
            except Exception as ex:  # noqa: BLE001
                print(f"[error] failed normalization tile failed for {idx}: {ex}")
                bad_idx = idx  # noqa: F841
                normalized_tiles.append(np.zeros((target_tile_h, target_tile_w, 3), dtype=np.float32))


        expected_tile_shape = (LR_PATCH, LR_PATCH, 3)
        bad_tiles = []
        for j, t in enumerate(normalized_tiles):
            if not isinstance(t, np.ndarray):
                t = np.asarray(t)  # noqa: PLW2901
                normalized_tiles[j] = t
            if t.shape != expected_tile_shape:
                bad_tiles.append((j, t.shape))

        if bad_tiles:
            print(
                f"[ERROR] found {len(bad_tiles)} tiles with unexpected shape e.g.: {bad_tiles[:5]}",
            )
            keep_idx = [i for i in range(len(normalized_tiles)) if (i, normalized_tiles[i].shape) not in bad_tiles]
            if len(keep_idx) == 0:
                print("[ERROR] all tiles invalid in that chunk (skipping whole)")
                return []
            normalized_tiles = [normalized_tiles[i] for i in keep_idx]
            tiles_meta = [tiles_meta[i] for i in keep_idx]

        valid_tiles = []
        valid_meta = []
        for idx, t in enumerate(normalized_tiles):
            if isinstance(t, np.ndarray) and t.shape == (target_tile_h, target_tile_w, 3):
                valid_tiles.append(t)
                valid_meta.append(tiles_meta[idx])
            else:
                print(f"[warning] dropping tile {idx}, wrong shape: {getattr(t, 'shape', None)}")

        if not valid_tiles:
            print("[ERROR] no valid tiles (skipping chunk)")
            return []

        tiles_arr = np.stack(valid_tiles, axis=0).astype(np.float32)
        tiles_meta = valid_meta

        # debug shapes
        unique_shapes = {}
        for s in [t.shape for t in normalized_tiles]:
            unique_shapes[s] = unique_shapes.get(s, 0) + 1
        # TODO: print
        # print(f"[debug] prepared tiles: count={tiles_arr.shape[0]}, tile_shape={tiles_arr.shape[1:]}, unique_shapes={unique_shapes}")


        # run prediction
        try:
            preds = model.predict(tiles_arr, batch_size=pred_batch_size, verbose=1)
        except Exception as ex:
            # surface diagnostics and re-raise so fallback triggers if desired
            print(f"[error] model.predict failed: {ex}; tiles_arr.shape={tiles_arr.shape}")
            raise

        chunk_rows = []
        nonlocal example_idx

        for i_pred, pred in enumerate(preds):
            local_idx, y, x = tiles_meta[i_pred]
            info = infos[local_idx]
            if info.get("failed"):
                continue

            y_hr = (y // LR_PATCH) * HR_PATCH
            x_hr = (x // LR_PATCH) * HR_PATCH

            hr_np = info["hr_np"]
            hr_h = info["hr_h"]
            hr_w = info["hr_w"]

            y_end = min(hr_h, y_hr + HR_PATCH)
            x_end = min(hr_w, x_hr + HR_PATCH)
            hr_patch = hr_np[y_hr:y_end, x_hr:x_end, :]

            bic_np = info["bic_np"]
            bic_patch = bic_np[y_hr:y_end, x_hr:x_end, :]

            pred_patch = pred

            if hr_patch.shape[0] != HR_PATCH or hr_patch.shape[1] != HR_PATCH:
                target_h, target_w = hr_patch.shape[0], hr_patch.shape[1]
                if target_h == 0 or target_w == 0:
                    # skip degenerate patch
                    continue
                pred_patch = tf.image.resize(pred_patch, [target_h, target_w], method="bicubic").numpy()
                bic_patch = tf.image.resize(bic_patch, [target_h, target_w], method="bicubic").numpy()

            psnr_b, ssim_b, mae_b = compute_metrics_pair(hr_patch, bic_patch)
            psnr_m, ssim_m, mae_m = compute_metrics_pair(hr_patch, pred_patch)

            # TODO: print
            # if math.isinf(psnr_b):
            #     print(f"[warning] bicubic PSNR is +inf for file={info['path']} at (y_hr={y_hr},x_hr={x_hr})")
            # if math.isinf(psnr_m):
            #     print(f"[warning] model PSNR is +inf for file={info['path']} at (y_hr={y_hr},x_hr={x_hr})")

            chunk_rows.append(
                {
                    "file": info["path"],
                    "y_hr": int(y_hr),
                    "x_hr": int(x_hr),
                    "psnr_bicubic": psnr_b,
                    "ssim_bicubic": ssim_b,
                    "mae_bicubic": mae_b,
                    "psnr_model": psnr_m,
                    "ssim_model": ssim_m,
                    "mae_model": mae_m,
                    "psnr_delta": psnr_m - psnr_b,
                    "ssim_delta": ssim_m - ssim_b,
                    "mae_delta": mae_m - mae_b,
                },
            )

            sp_list = sample_patches.get(local_idx)
            if sp_list is not None and len(sp_list) < PATCHES_PER_IMAGE:
                h_hr, w_hr = hr_patch.shape[0], hr_patch.shape[1]
                min_side = min(HR_PATCH // 8, 8)
                if h_hr < min_side or w_hr < min_side:
                    continue

                hr_copy = hr_patch if isinstance(hr_patch, np.ndarray) else hr_patch.numpy()
                pred_copy = pred_patch if isinstance(pred_patch, np.ndarray) else pred_patch.numpy()
                bic_copy = bic_patch if isinstance(bic_patch, np.ndarray) else np.array(bic_patch)
                diff_model = np.clip(np.abs(hr_copy - pred_copy) * 4.0, 0.0, 1.0)
                diff_bic = np.clip(np.abs(hr_copy - bic_copy) * 4.0, 0.0, 1.0)
                sp_list.append((bic_copy, pred_copy, hr_copy, diff_model, diff_bic))

        for local_idx, info in enumerate(infos):
            if info.get("failed"):
                continue
            sp_list = sample_patches.get(local_idx, [])
            if not sp_list:
                continue

            heights = [int(gt.shape[0]) for (_, _, gt, _, _) in sp_list]
            widths = [int(gt.shape[1]) for (_, _, gt, _, _) in sp_list]
            target_h = max(heights)
            target_w = max(widths)

            rows_vis = []
            rows_diff = []
            for bic_vis, model_vis, gt_vis, diff_model, diff_bic in sp_list:
                # convert to numpy
                bic_arr = bic_vis if isinstance(bic_vis, np.ndarray) else np.asarray(bic_vis)
                model_arr = model_vis if isinstance(model_vis, np.ndarray) else np.asarray(model_vis)
                gt_arr = gt_vis if isinstance(gt_vis, np.ndarray) else np.asarray(gt_vis)
                diff_model_arr = diff_model if isinstance(diff_model, np.ndarray) else np.asarray(diff_model)
                diff_bic_arr = diff_bic if isinstance(diff_bic, np.ndarray) else np.asarray(diff_bic)

                # normalize each to (target_h, target_w) size
                bic_r = pad_or_crop_to(bic_arr, target_h, target_w)
                model_r = pad_or_crop_to(model_arr, target_h, target_w)
                gt_r = pad_or_crop_to(gt_arr, target_h, target_w)
                diff_model_r = pad_or_crop_to(diff_model_arr, target_h, target_w)
                diff_bic_r = pad_or_crop_to(diff_bic_arr, target_h, target_w)

                concat_row = np.concatenate([bic_r, model_r, gt_r], axis=1)
                diff_row = np.concatenate([diff_model_r, diff_bic_r], axis=1)

                rows_vis.append(concat_row)
                rows_diff.append(diff_row)

            # stack rows vertically
            vis_concat = np.concatenate(rows_vis, axis=0)
            diff_concat = np.concatenate(rows_diff, axis=0)

            base_stem = Path(info["path"]).stem
            out_name = f"patches_{base_stem}.png"

            try:
                if vis_concat.shape[0] != diff_concat.shape[0]:
                    msg = f"vis and diff heights differ: {vis_concat.shape[0]} vs {diff_concat.shape[0]}"
                    raise ValueError(msg)  # noqa: TRY301

                # separator width: small, dynamic but bounded
                separator_w = max(8, min(32, vis_concat.shape[1] // 50))
                # white separator (values in [0,1]); change to 0.0 for black if you prefer
                sep = np.ones((vis_concat.shape[0], separator_w, 3), dtype=np.float32)

                # horizontal layout: [visuals | separator | diffs]
                combined = np.concatenate([vis_concat, sep, diff_concat], axis=1)

                # save single combined image
                save_image_np(combined, examples_dir / out_name)
            except Exception as ex:  # noqa: BLE001
                print(
                    f"[warning] unable to create combined image for {info['path']}: {ex}",
                )
                save_image_np(vis_concat, examples_dir / out_name)
                save_image_np(diff_concat, examples_dir / f"{Path(out_name).stem}_diffs.png")

            saved_images_count += 1

        return chunk_rows

    all_rows = []
    for i in range(0, len(test_files), chunk_size):
        chunk_files = test_files[i : i + chunk_size]
        try:
            chunk_rows = _process_chunk(chunk_files, global_img_idx_offset=i)
            all_rows.extend(chunk_rows)
        except Exception as e:  # noqa: BLE001
            print(f"error processing chunk starting at index {i}: {e}")
            # fallback to per-image eval for that chunk (per-patch using full-image tiled predictor)
            for p in chunk_files:
                try:
                    hr_tf = load_image(p)
                    hr_np = hr_tf.numpy()
                    lr_h = max(1, math.ceil(hr_np.shape[0] / UPSCALE))  # noqa: F841
                    lr_w = max(1, math.ceil(hr_np.shape[1] / UPSCALE))  # noqa: F841
                    bic_np = bicubic_upscale_from_hr(hr_tf, upscale=UPSCALE).numpy()

                    # get per-patch preds using tiled predictor
                    preds = predict_full_image_tiled(
                        model,
                        hr_tf,
                        upscale=UPSCALE,
                        lr_patch=LR_PATCH,
                        hr_patch=HR_PATCH,
                    )
                    # preds here is full image; extract patches similarly to above
                    # (to keep fallback simple we'll just compute image-level metrics)
                    psnr_b, ssim_b, mae_b = compute_metrics_pair(hr_np, bic_np)
                    psnr_m, ssim_m, mae_m = compute_metrics_pair(hr_np, preds)
                    all_rows.append(
                        {
                            "file": str(p),
                            "psnr_bicubic": psnr_b,
                            "ssim_bicubic": ssim_b,
                            "mae_bicubic": mae_b,
                            "psnr_model": psnr_m,
                            "ssim_model": ssim_m,
                            "mae_model": mae_m,
                            "psnr_delta": psnr_m - psnr_b,
                            "ssim_delta": ssim_m - ssim_b,
                            "mae_delta": mae_m - mae_b,
                        },
                    )
                except Exception as e2:  # noqa: BLE001
                    print(f"failed to process {p} in fallback: {e2}")
                    continue

    if len(all_rows) == 0:
        msg = "No results collected from evaluation."
        raise ValueError(msg)

    df = pd.DataFrame(all_rows)
    df.to_csv(log_root / "eval_results.csv", index=False)
    print("Saved results to", log_root / "eval_results.csv")

    df_for_stats = df.replace([np.inf, -np.inf], np.nan)

    summary = df_for_stats[
        [
            "psnr_bicubic",
            "psnr_model",
            "ssim_bicubic",
            "ssim_model",
            "mae_bicubic",
            "mae_model",
        ]
    ].agg(
        ["mean", "std", "median"],
    )
    (log_root / "eval_summary.json").write_text(summary.to_json())
    (log_root / "eval_summary.csv").write_text(summary.to_csv())

    fig1 = px.scatter(
        df,
        x="psnr_bicubic",
        y="psnr_model",
        hover_data=["file", "y_hr", "x_hr"],
        labels={"psnr_bicubic": "PSNR (bicubic)", "psnr_model": "PSNR (model)"},
        title="PSNR: model vs bicubic (per-patch)",
    )
    fig1.add_trace(
        go.Scatter(
            x=[df.psnr_bicubic.min(), df.psnr_bicubic.max()],
            y=[df.psnr_bicubic.min(), df.psnr_bicubic.max()],
            mode="lines",
            line=dict(color="gray", dash="dash"),
            showlegend=False,
        ),
    )
    fig2 = px.histogram(df, x="psnr_delta", nbins=40, title="Histogram of PSNR (model - bicubic) (per-patch)")
    long = df.melt(
        id_vars=["file", "y_hr", "x_hr"],
        value_vars=["psnr_bicubic", "psnr_model"],
        var_name="method",
        value_name="psnr",
    )
    fig3 = px.box(long, x="method", y="psnr", title="PSNR distribution by method (per-patch)")

    html_out = log_root / "eval_plots.html"
    with Path.open(html_out, "w") as f:
        f.write("<html><head><meta charset='utf-8'></head><body>\n")
        f.write("<h1>Evaluation plots (per-patch)</h1>\n")
        f.write("<h2>PSNR: model vs bicubic</h2>\n")
        f.write(fig1.to_html(full_html=True))
        f.write("<h2>PSNR delta histogram</h2>\n")
        f.write(fig2.to_html(full_html=True))
        f.write("<h2>PSNR distributions</h2>\n")
        f.write(fig3.to_html(full_html=True))
        f.write("</body></html>\n")

    print("Saved plots to ", html_out)
    print("Saved example patches to", examples_dir)
    return df, summary


# kept as fallback
def _eval_per_image(
    model_path: Path,
    data_dir: Path = DATA_DIR,
    log_root: Path = LOG_ROOT,
    save_examples: int = 6,
):
    # load model with custom metrics
    model = keras.models.load_model(
        str(model_path),
        custom_objects={"psnr_metric": psnr_metric, "ssim_metric": ssim_metric},
    )

    test_files = collect_test_files(data_dir)
    if not test_files:
        msg = f"No test images found under {data_dir!r}"
        raise ValueError(msg)

    rows = []
    example_idx = 0
    examples_dir = log_root / "eval_examples"
    examples_dir.mkdir(parents=True, exist_ok=True)

    for p in test_files:
        try:
            hr_tf = load_image(p)  # [H, W, 3]
            hr_np = hr_tf.numpy()
            bic_np = bicubic_upscale_from_hr(hr_tf, upscale=UPSCALE).numpy()
            model_np = predict_full_image_tiled(model, hr_tf, upscale=UPSCALE, lr_patch=LR_PATCH, hr_patch=HR_PATCH)

            psnr_b, ssim_b, mae_b = compute_metrics_pair(hr_np, bic_np)
            psnr_m, ssim_m, mae_m = compute_metrics_pair(hr_np, model_np)

            rows.append(
                {
                    "file": str(p),
                    "psnr_bicubic": psnr_b,
                    "ssim_bicubic": ssim_b,
                    "mae_bicubic": mae_b,
                    "psnr_model": psnr_m,
                    "ssim_model": ssim_m,
                    "mae_model": mae_m,
                    "psnr_delta": psnr_m - psnr_b,
                    "ssim_delta": ssim_m - ssim_b,
                    "mae_delta": mae_m - mae_b,
                },
            )

            if example_idx < save_examples:
                # SAMPLES: bicubic, model_pred, GT
                concat = np.concatenate([bic_np, model_np, hr_np], axis=1)
                save_image_np(concat, examples_dir / f"example_{example_idx:02d}_{p.name}")
                example_idx += 1
        except Exception as e:  # noqa: BLE001
            print(f"failed to process {p}: {e}")
            continue

    df = pd.DataFrame(rows)
    df.to_csv(log_root / "eval_results.csv", index=False)
    print("Saved results to", log_root / "eval_results.csv")

    # summary stats
    summary = df[["psnr_bicubic", "psnr_model", "ssim_bicubic", "ssim_model", "mae_bicubic", "mae_model"]].agg(
        ["mean", "std", "median"],
    )
    (log_root / "eval_summary.json").write_text(summary.to_json())
    (log_root / "eval_summary.csv").write_text(summary.to_csv())

    fig1 = px.scatter(
        df,
        x="psnr_bicubic",
        y="psnr_model",
        hover_data=["file"],
        labels={"psnr_bicubic": "PSNR (bicubic)", "psnr_model": "PSNR (model)"},
        title="PSNR: model | bicubic",
    )
    fig1.add_trace(
        go.Line(
            x=[df.psnr_bicubic.min(), df.psnr_bicubic.max()],
            y=[df.psnr_bicubic.min(), df.psnr_bicubic.max()],
            line=dict(color="gray", dash="dash"),
            showlegend=False,
        ),
    )
    fig2 = px.histogram(df, x="psnr_delta", nbins=40, title="hist of psnr: model | bicubic")
    long = df.melt(id_vars=["file"], value_vars=["psnr_bicubic", "psnr_model"], var_name="method", value_name="psnr")
    fig3 = px.box(long, x="method", y="psnr", title="psnr dist")

    html_out = log_root / "eval_plots.html"
    with Path.open(html_out, "w") as f:
        f.write("<html><head><meta charset='utf-8'></head><body>\n")
        f.write("<h1>eval plots</h1>\n")
        f.write("<h2>psnr: model | bicubic</h2>\n")
        f.write(fig1.to_html(full_html=True))
        f.write("<h2>psnr delta hist</h2>\n")
        f.write(fig2.to_html(full_html=True))
        f.write("<h2>psnr dist</h2>\n")
        f.write(fig3.to_html(full_html=True))
        f.write("</body></html>\n")

    print("Saved interactive plots to", html_out)
    print("Saved example comparison images to", examples_dir)
    return df, summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="first_unet_test.keras",
        help="model in .keras format",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DATA_DIR),
        help="main dataset dir",
    )
    parser.add_argument(
        "--log-root",
        type=str,
        default=str(LOG_ROOT),
        help="Dir to save results and logs",
    )
    parser.add_argument(
        "-f",
        "--fraction",
        type=float,
        default=1.0,
        help="Fraction of dataset to evaluate (0.0-1.0)",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    data_dir = Path(args.data_dir)
    log_root = Path(args.log_root)
    log_root.mkdir(parents=True, exist_ok=True)

    df, summary = eval_on_testset(
        model_path,
        data_dir=data_dir,
        log_root=log_root,
        fraction=args.fraction,
    )
    print(
        "Done. \nMean PSNR model:",
        float(df["psnr_model"].mean()),
        "Mean PSNR bicubic:",
        float(df["psnr_bicubic"].mean()),
    )
