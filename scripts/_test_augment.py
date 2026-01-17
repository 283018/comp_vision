import sys
from operator import index
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import cfg
from data.augmentation import augment, make_lr_hr_pair
from data.collector import RunMode, build_dataset, image_iterator, load_image


def show_before_after():
    patches_per_img = 4
    ds, _ = build_dataset(
        run_mode=RunMode.TRAIN,
        batch_size=8,
        patches_per_image=patches_per_img,
        seed=42,
    )
    lr_batch, hr_batch = next(iter(ds))

    lr_batch = lr_batch.numpy()
    hr_batch = hr_batch.numpy()

    lr_batch = np.clip(lr_batch, 0.0, 1.0)
    hr_batch = np.clip(hr_batch, 0.0, 1.0)

    fig, axes = plt.subplots(patches_per_img, 2, figsize=(6, 3 * patches_per_img))

    #! shows already augmented images
    for i in range(patches_per_img):
        axes[i, 0].imshow(lr_batch[i])
        axes[i, 0].set_title("LR (augmented)")

        axes[i, 1].imshow(hr_batch[i])
        axes[i, 1].set_title("HR (augmented)")

        for ax in axes[i]:
            ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    files = image_iterator(
        RunMode.TEST,
        (0.9999, 0.0001),
        data_dir=cfg.DATA_DIR,
        index_path=cfg.DATA_DIR / "index.json",
        seed=123321,
        regenerate_index=False,
    )
    files = list(files)

    for f in files:
        print(f"{cfg.DATA_DIR / f}")
        show_before_after()
