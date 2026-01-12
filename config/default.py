import os
from dataclasses import dataclass
from pathlib import Path

import tensorflow as tf
from dotenv import load_dotenv

load_dotenv(Path("../home.env"))
USER_HOME_DIR = os.getenv("USER_HOME_DIR") or Path.home()


@dataclass(slots=True, frozen=True)
class Config:
    UPSCALE: int = 4
    HR_PATCH: int = 128
    LR_PATCH: int = HR_PATCH // UPSCALE

    BATCH_SIZE: int = 32  # 16 is safe, but should be ok
    AUTOTUNE = tf.data.AUTOTUNE
    DATA_DIR = Path(USER_HOME_DIR) / Path("image_data")
    EPOCHS = 120
    ALLOWED_EXTS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".webp"})

    LOG_ROOT = Path("training_logs")
    LOG_ROOT.mkdir(exist_ok=True)
    BAD_FILES_LOG = LOG_ROOT / Path("bar_images.txt")
