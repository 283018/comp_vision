import os
from dataclasses import dataclass
from pathlib import Path

import tensorflow as tf
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
ENV_FILE = PROJECT_ROOT / "home.env"
load_dotenv(ENV_FILE)
USER_HOME_DIR = os.getenv("USER_HOME_DIR")

if USER_HOME_DIR is None:
    msg = "User home directory not found, check your home.env!"
    raise ValueError(msg)


@dataclass(slots=True)
class Config:
    UPSCALE: int
    HR_PATCH: int

    BATCH_SIZE: int
    EPOCHS: int
    
    LR_PATCH: int | None = None
    AUTOTUNE = tf.data.AUTOTUNE
    DATA_DIR = Path(USER_HOME_DIR).resolve() / "image_data"
    ALLOWED_EXTS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".webp"})

    LOG_ROOT = PROJECT_ROOT / "logs"
    BAD_FILES_LOG = LOG_ROOT / Path("bar_images.txt")

    def __post_init__(self):
        if self.LR_PATCH is None:
            self.LR_PATCH = self.HR_PATCH // self.UPSCALE
        self.LOG_ROOT.mkdir(exist_ok=True)
