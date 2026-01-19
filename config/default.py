from .common import Config

CFG_DEFAULT = Config(
    UPSCALE=4,
    HR_PATCH=128,
    BATCH_SIZE=32,  # 16 is safe but should be ok
    EPOCHS=120,
)
