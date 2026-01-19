import math
from datetime import datetime

import absl.logging
import keras
import pytz
import tensorflow as tf
from keras import mixed_precision

from config import cfg
from data import RunMode, build_dataset
from training import compile_and_train

absl.logging.set_verbosity(absl.logging.ERROR)
# tf.config.optimizer.set_jit(False)

mixed_precision.set_global_policy("mixed_float16")

epochs_to_save = [1, 5, 7, 10, 13, 15, 17, 18, 20, 23, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 38, 40, 42, 45, 53, 55, 57, 60, 63, 65, 67, 70, 73, 75]  # fmt: skip

if __name__ == "__main__":
    train_ds, n_train = build_dataset(
        run_mode=RunMode.TRAIN,
        data_dir=cfg.DATA_DIR,
        batch_size=cfg.BATCH_SIZE,
        patches_per_image=4,
        seed=123321,
    )

    # should be ok
    # if not try assert_cardinality??
    steps_per_epoch = math.ceil(n_train / cfg.BATCH_SIZE)

    model, history = compile_and_train(
        train_ds,
        steps_per_epoch=steps_per_epoch,
        make_snapshots_on=epochs_to_save,
        pretrain_epochs=5,
    )
    model.save("first_srgan_test.keras")

    end = datetime.now(pytz.timezone("Poland"))

    # pass
