import os
from dataclasses import dataclass
from pathlib import Path

import tensorflow as tf
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
ENV_FILE = PROJECT_ROOT / "home.env"
load_dotenv(ENV_FILE)
USER_HOME_DIR = os.getenv("USER_HOME_DIR")

@dataclass(slots=True, frozen=True)
class Config:
    UPSCALE: int = 4
    HR_PATCH: int = 128
    LR_PATCH: int = HR_PATCH // UPSCALE

    BATCH_SIZE: int = 32  # 16 is safe, but should be ok
    AUTOTUNE = tf.data.AUTOTUNE
    DATA_DIR = Path(USER_HOME_DIR).resolve() / "image_data"
    EPOCHS = 120
    ALLOWED_EXTS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".webp"})

    LOG_ROOT = PROJECT_ROOT / "training_logs"
    LOG_ROOT.mkdir(exist_ok=True)
    BAD_FILES_LOG = LOG_ROOT / Path("bar_images.txt")
