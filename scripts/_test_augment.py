import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import cfg
from data.augmentation import augment, make_lr_hr_pair
from data.collector import load_image


def show_before_after(image_path):
    img = load_image(Path(image_path))
    lr, hr = make_lr_hr_pair(img)

    #!
    lr_aug, hr_aug = augment(
        lr,
        hr,
        lr_flip_prob = 0,
        ud_flip_prob = 0.,
        rot90_prob = 0.,
    )
    lr_aug = np.clip(lr_aug, 0.0, 1.0)
    hr_aug = np.clip(hr_aug, 0.0, 1.0)

    lr, hr = lr.numpy(), hr.numpy()
    lr = np.clip(lr, 0.0, 1.0)
    hr = np.clip(hr, 0.0, 1.0)

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    axes[0, 0].imshow(lr)
    axes[0, 0].set_title("LR (before)")
    axes[0, 1].imshow(hr)
    axes[0, 1].set_title("HR (before)")

    axes[1, 0].imshow(lr_aug)
    axes[1, 0].set_title("LR (after)")
    axes[1, 1].imshow(hr_aug)
    axes[1, 1].set_title("HR (after)")

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    files = [
        "div2k/test/0014.png",
        "div2k/test/0018.png",
        "div2k/test/0021.png",
        "div2k/test/0055.png",
        "div2k/test/0089.png",
        "div2k/test/0128.png",
        "div2k/test/0158.png",
        "div2k/test/0191.png",
        "div2k/test/0261.png",
        "div2k/test/0305.png",
        "div2k/test/0335.png",
        "div2k/test/0398.png",
        "div2k/test/0443.png",
        "div2k/test/0519.png",
        "div2k/test/0616.png",
        "div2k/test/0688.png",
        "div2k/test/0820.png",
        "div2k/test/0897.png",
        "div2k/test/0492.png",
        "div2k/test/0859.png",
    ]

    for f in files:
        print(f"{cfg.DATA_DIR / f}")
        show_before_after(cfg.DATA_DIR / f)
