import json
from datetime import datetime
from pathlib import Path

import keras
import pandas as pd
import tensorflow as tf
from keras import mixed_precision

from config import cfg
from models import build_unet_upscaler
from training import SnapshotOnEpoch, SnapshotOnPlateau, TrainingMonitor


def compile_and_train(train_ds, log_root=cfg.LOG_ROOT, steps_per_epoch=None, val_ds=None):
    model = build_unet_upscaler()

    # metrics wrappers
    def psnr_metric(y_true, y_pred):
        return tf.image.psnr(tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32), max_val=1.0) # cast for mixed float stability

    def ssim_metric(y_true, y_pred):
        return tf.image.ssim(tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32), max_val=1.0)

    model.compile(
        optimizer=mixed_precision.LossScaleOptimizer(
            tf.keras.optimizers.Adam(1e-4),
        ),
        loss=tf.keras.losses.MeanAbsoluteError(),  # TODO: change metrics for better psnr?
        metrics=[psnr_metric, ssim_metric],
    )
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            "unet_best.keras",
            save_best_only=True,
            monitor="val_loss" if val_ds else "loss",
        ),
        # TODO: try without reduce?
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss" if val_ds else "loss", factor=0.5, patience=6),
        # TODO: snapshot vs early stop
        # tf.keras.callbacks.EarlyStopping(
        #     monitor="val_loss" if val_ds else "loss",
        #     patience=12,
        #     restore_best_weights=True,
        # ),
        SnapshotOnPlateau(
            monitor="val_loss" if val_ds else "loss",
            patience=12,
            save_dir=log_root / "plateau_snapshots",
        ),
        SnapshotOnEpoch([20, 30, 80, 90, 100, 110], save_dir=log_root / Path("model_epoch_snapshots")),
    ]

    csv_logger = tf.keras.callbacks.CSVLogger(str(log_root / "metrics.csv"), append=True)
    callbacks.append(csv_logger)

    # TensorBoard
    tb_dir = log_root / "tensorboard"
    callbacks.append(
        tf.keras.callbacks.TensorBoard(
            log_dir=str(tb_dir),
            # profile_batch=(20, 50),     # only for profiling
            histogram_freq=1,
        ),
    )

    monitor = TrainingMonitor(log_root=log_root, save_freq_epochs=5, max_samples=3)
    callbacks.append(monitor)

    history = model.fit(
        train_ds,
        epochs=cfg.EPOCHS,
        validation_data=val_ds,
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch,
    )

    (log_root / "history.json").write_text(json.dumps(history.history, indent=2))

    df = pd.DataFrame(history.history)
    df.index = df.index + 1
    df.index.name = "epoch"
    df.to_csv(log_root / "history.csv")

    model.save(str(log_root / "final_unet.keras"))

    return model, history
