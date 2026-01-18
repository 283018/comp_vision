import csv
from pathlib import Path

import numpy as np
import tensorflow as tf
from keras.applications.vgg19 import preprocess_input as vgg_preprocess
from PIL import Image
from plotly import graph_objects as go

from config import cfg
from data.collector import RunMode, image_iterator, load_image
from models.sr_gan.subs import PixelShuffle, build_generator, build_vgg_feature_extractor


def crop_center_divisible_by_upscale(img: tf.Tensor, upscale: int) -> tuple[tf.Tensor, tuple[int, int]]:
    shape = tf.shape(img)
    h = int(shape[0].numpy())
    w = int(shape[1].numpy())

    h_div = (h // upscale) * upscale
    w_div = (w // upscale) * upscale
    if h_div == 0 or w_div == 0:
        h_div = max(upscale, h)
        w_div = max(upscale, w)

    off_y = (h - h_div) // 2
    off_x = (w - w_div) // 2
    hr_crop = tf.image.crop_to_bounding_box(img, off_y, off_x, h_div, w_div)
    return hr_crop, (int(off_y), int(off_x))


def make_lr_hr_pair_eval_fixed(image: tf.Tensor):
    shape = tf.shape(image)
    h = shape[0]
    w = shape[1]

    min_size = tf.minimum(h, w)

    def resize_up():
        scale = tf.cast(cfg.HR_PATCH, tf.float32) / tf.cast(min_size, tf.float32)
        new_h = tf.cast(tf.round(tf.cast(h, tf.float32) * scale), tf.int32)
        new_w = tf.cast(tf.round(tf.cast(w, tf.float32) * scale), tf.int32)
        return tf.image.resize(image, [new_h, new_w], method="bicubic")

    image = tf.cond(min_size < cfg.HR_PATCH, resize_up, lambda: image)

    shape = tf.shape(image)
    h = shape[0]
    w = shape[1]

    off_y = (h - cfg.HR_PATCH) // 2
    off_x = (w - cfg.HR_PATCH) // 2

    hr = tf.image.crop_to_bounding_box(
        image,
        off_y,
        off_x,
        cfg.HR_PATCH,
        cfg.HR_PATCH,
    )

    lr = tf.image.resize(hr, [cfg.LR_PATCH, cfg.LR_PATCH], method="bicubic")
    return lr, hr


# will try later with fully convolutional generator
def make_lr_from_hr(hr: tf.Tensor, upscale: int) -> tf.Tensor:
    h = tf.shape(hr)[0]
    w = tf.shape(hr)[1]
    lr_h = h // upscale
    lr_w = w // upscale
    return tf.image.resize(hr, [lr_h, lr_w], method="bicubic", antialias=True)


def upsample_bicubic(lr: tf.Tensor, upscale: int) -> tf.Tensor:
    h = tf.shape(lr)[0] * upscale
    w = tf.shape(lr)[1] * upscale
    return tf.image.resize(lr, [h, w], method="bicubic", antialias=True)


def to_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)


def save_grid_bicubic_sr_hr_diff(lr_up_np: np.ndarray, sr_np: np.ndarray, hr_np: np.ndarray, out_path: Path) -> None:
    diff = np.clip(np.abs(sr_np.astype(np.int16) - hr_np.astype(np.int16)), 0, 255).astype(np.uint8)
    imgs = [lr_up_np, sr_np, hr_np, diff]
    h, w, c = imgs[0].shape
    canvas = Image.new("RGB", (w * 4, h))
    for i, im in enumerate(imgs):
        pil = Image.fromarray(im)
        canvas.paste(pil, (i * w, 0))
    canvas.save(str(out_path), format="PNG")


def evaluate_on_iterator(  # noqa: PLR0915
    generator: tf.keras.Model,
    img_paths_iter,
    num_images: int,
    out_dir: Path,
    upscale: int,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "metrics.csv"

    rows = []
    cnt = 0

    vgg = build_vgg_feature_extractor(
        layer_name="block5_conv4",
        hr_shape=(cfg.HR_PATCH, cfg.HR_PATCH, 3),
    )

    # for plotting
    psnrs = []
    ssims = []
    vgg_sr_dists = []
    vgg_bic_dists = []

    def compute_vgg_feats(img_tensor):
        x = tf.cast(img_tensor * 255.0, tf.float32)
        x = vgg_preprocess(x)
        x = tf.expand_dims(x, axis=0)  # type: ignore
        return vgg(x)

    for p in img_paths_iter:
        if num_images and cnt >= num_images:
            break
        pth = Path(p)
        try:
            hr = load_image(p)
        except Exception as e:  # noqa: BLE001
            print(f"Skipping {pth}: load_image failed: {e}")
            continue

        # ensure float32 0..1
        hr = tf.image.convert_image_dtype(hr, tf.float32)

        # TODO: START full size update
        # crop center to divisible dims
        # hr_crop, (off_y, off_x) = crop_center_divisible_by_upscale(hr, upscale)

        # if crop is smaller than upscale (rare), pad instead
        # sh = tf.shape(hr_crop)
        # if sh[0] < upscale or sh[1] < upscale:
        #     # pad to upscale using reflect
        #     pad_h = int(upscale - int(sh[0].numpy())) if int(sh[0].numpy()) < upscale else 0
        #     pad_w = int(upscale - int(sh[1].numpy())) if int(sh[1].numpy()) < upscale else 0
        #     pad_top = pad_h // 2
        #     pad_bottom = pad_h - pad_top
        #     pad_left = pad_w // 2
        #     pad_right = pad_w - pad_left
        #     hr_crop = tf.pad(hr_crop, [[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode="REFLECT")

        # lr = make_lr_from_hr(hr_crop, upscale)

        lr, hr_crop = make_lr_hr_pair_eval_fixed(hr)

        lr_b = tf.expand_dims(lr, axis=0)
        sr_b = generator(lr_b, training=False)
        sr = tf.squeeze(sr_b, axis=0)

        # bicubic baseline (for visualization)
        lr_up = tf.image.resize(lr, [cfg.HR_PATCH, cfg.HR_PATCH], method="bicubic")
        # TODO: END fixed sized

        # model expects batch dimension
        lr_b = tf.expand_dims(lr, axis=0)
        sr_b = generator(lr_b, training=False)
        sr = tf.squeeze(sr_b, axis=0)

        # bicubic baseline
        lr_up = upsample_bicubic(lr, upscale)

        # ensure shapes match hr_crop
        sr = tf.image.resize(sr, tf.shape(hr_crop)[:2], method="bicubic")
        lr_up = tf.image.resize(lr_up, tf.shape(hr_crop)[:2], method="bicubic")

        # convert to numpy uint8
        hr_np = to_uint8(hr_crop.numpy())
        sr_np = to_uint8(tf.clip_by_value(sr, 0.0, 1.0).numpy())
        lr_up_np = to_uint8(tf.clip_by_value(lr_up, 0.0, 1.0).numpy())

        # metrics
        psnr_val = float(tf.image.psnr(tf.clip_by_value(sr, 0.0, 1.0), hr_crop, max_val=1.0).numpy())
        ssim_val = float(tf.image.ssim(tf.clip_by_value(sr, 0.0, 1.0), hr_crop, max_val=1.0).numpy())

        # VGG distance
        f_real = compute_vgg_feats(hr_crop)
        f_sr = compute_vgg_feats(sr)
        f_bic = compute_vgg_feats(lr_up)

        # L2 mean
        vgg_sr = float(tf.reduce_mean(tf.square(f_real - f_sr)).numpy())
        vgg_bic = float(tf.reduce_mean(tf.square(f_real - f_bic)).numpy())

        # save visualization
        out_name = f"{cnt + 1:04d}_{pth.stem}.png"
        save_grid_bicubic_sr_hr_diff(lr_up_np, sr_np, hr_np, out_dir / out_name)

        rows.append(
            {
                "sample": pth.name,
                "psnr": psnr_val,
                "ssim": ssim_val,
                "vgg_sr": vgg_sr,
                "vgg_bic": vgg_bic,
                "file": out_name,
            },
        )

        # for plotting
        psnrs.append(psnr_val)
        ssims.append(ssim_val)
        vgg_sr_dists.append(vgg_sr)
        vgg_bic_dists.append(vgg_bic)

        if (cnt + 1) % 10 == 0:
            print(f"Processed {cnt + 1} images")

        cnt += 1

    with Path.open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample", "psnr", "ssim", "vgg_sr", "vgg_bic", "file"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # summary
    psnrs = [r["psnr"] for r in rows]
    ssims = [r["ssim"] for r in rows]
    avg_psnr = float(np.mean(psnrs)) if psnrs else float("nan")
    avg_ssim = float(np.mean(ssims)) if ssims else float("nan")
    avg_vgg_sr = float(np.mean(vgg_sr_dists)) if vgg_sr_dists else float("nan")
    avg_vgg_bic = float(np.mean(vgg_bic_dists)) if vgg_bic_dists else float("nan")
    with Path.open(out_dir / "summary.txt", "w") as f:
        f.write(f"samples: {len(rows)}\n")
        f.write(f"avg_psnr: {avg_psnr:.4f}\n")
        f.write(f"avg_ssim: {avg_ssim:.6f}\n")
        f.write(f"avg_vgg_sr: {avg_vgg_sr:.6e}\n")
        f.write(f"avg_vgg_bic: {avg_vgg_bic:.6e}\n")

    print(f"Done. {len(rows)} samples \navg: PSNR={avg_psnr:.4f}, SSIM={avg_ssim:.6f}")
    print(f"avg VGG L2 (SR): {avg_vgg_sr:.6e}, (Bic): {avg_vgg_bic:.6e}")

    # PSNR + SSIM
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            y=psnrs,
            mode="lines+markers",
            name="PSNR",
            marker=dict(symbol="circle"),
        ),
    )

    fig.add_trace(
        go.Scatter(
            y=ssims,
            mode="lines+markers",
            name="SSIM",
            marker=dict(symbol="x"),
            yaxis="y2",  # optional: separate scale
        ),
    )

    fig.update_layout(
        title="PSNR + SSIM per sample",
        xaxis_title="sample",
        yaxis=dict(
            title="PSNR",
            showgrid=True,
            gridcolor="rgba(0,0,0,0.1)",
        ),
        yaxis2=dict(
            title="SSIM",
            overlaying="y",
            side="right",
            showgrid=False,
            range=[0, 1],  # SSIM range
        ),
        legend=dict(x=0.01, y=0.99),
        width=900,
        height=400,
    )

    # fig.write_image(out_dir / "psnr_ssim_per_sample.png")
    fig.write_html(out_dir / "psnr_ssim_per_sample.html")


    # VGG L2
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            y=vgg_bic_dists,
            mode="lines+markers",
            name="VGG L2 (bicubic)",
            marker=dict(symbol="circle"),
        ),
    )

    fig.add_trace(
        go.Scatter(
            y=vgg_sr_dists,
            mode="lines+markers",
            name="VGG L2 (SR)",
            marker=dict(symbol="x"),
        ),
    )

    fig.update_layout(
        title="VGG L2 distance",
        xaxis_title="sample",
        yaxis_title="VGG L2 distance",
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(0,0,0,0.1)",
        ),
        legend=dict(x=0.01, y=0.99),
        width=900,
        height=400,
    )

    # fig.write_image(out_dir / "vgg_l2_per_sample.png")
    fig.write_html(out_dir / "vgg_l2_per_sample.html")


if __name__ == "__main__":
    model = tf.keras.models.load_model(
        "logs/model_epoch_snapshots/model_epoch_20_generator.keras",
        custom_objects={"PixelShuffle": PixelShuffle},
        compile=False,
    )
    img_iter = image_iterator(
        RunMode.TEST,
        index_path=cfg.FILEINDEX_PATH,  # type: ignore
        split_ratios=(0.8, 0.2),
        data_dir=cfg.DATA_DIR,
        regenerate_index=False,
    )
    evaluate_on_iterator(
        model,
        img_iter,
        num_images=200,
        out_dir=Path("logs/eval"),
        upscale=cfg.UPSCALE,
    )
