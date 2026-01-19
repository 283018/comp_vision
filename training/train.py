import json
from datetime import datetime
from pathlib import Path

import keras
import pandas as pd
import tensorflow as tf
from keras import mixed_precision
from tensorflow.train import Checkpoint, CheckpointManager

from config import cfg
from models import SRGAN, build_discriminator, build_generator, build_vgg_feature_extractor
from training import GeneratorCheckpoint, SnapshotOnEpoch, SnapshotOnPlateau, TrainingMonitor


def compile_and_train(train_ds, log_root=cfg.LOG_ROOT, steps_per_epoch=None, val_ds=None):
    gen = build_generator(lr_shape=(cfg.LR_PATCH, cfg.LR_PATCH, 3), upscale=cfg.UPSCALE)
    disc = build_discriminator(hr_shape=(cfg.HR_PATCH, cfg.HR_PATCH, 3))
    vgg = build_vgg_feature_extractor(layer_name="block5_conv4", hr_shape=(cfg.HR_PATCH, cfg.HR_PATCH, 3))

    # g_opt = mixed_precision.LossScaleOptimizer(tf.keras.optimizers.Adam(1e-4))
    # d_opt = mixed_precision.LossScaleOptimizer(tf.keras.optimizers.Adam(1e-4))

    g_opt = tf.keras.optimizers.Adam(1e-4)
    d_opt = tf.keras.optimizers.Adam(1e-4)

    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    mse = tf.keras.losses.MeanSquaredError()
    mae = tf.keras.losses.MeanAbsoluteError()

    # metrics wrappers
    def psnr_metric(y_true, y_pred):
        return tf.image.psnr(
            tf.cast(y_true, tf.float32),
            tf.cast(y_pred, tf.float32),
            max_val=1.0,
        )

    def ssim_metric(y_true, y_pred):
        return tf.image.ssim(tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32), max_val=1.0)

    srgan = SRGAN(
        gen,
        disc,
        vgg,
    )

    srgan.compile(
        g_optimizer=g_opt,
        d_optimizer=d_opt,
        content_loss_fn=mse,
        adv_loss_fn=bce,
        pixel_loss_fn=mae,
    )

    ckpt_dir = Path(log_root) / "training" / "training_checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt = Checkpoint(
        generator=gen,
        discriminator=disc,
        g_optimizer=g_opt,
        d_optimizer=d_opt,
        epoch=tf.Variable(0, trainable=False, dtype=tf.int64), # type: ignore
    )

    ckpt_manager = CheckpointManager(
        ckpt,
        directory=str(ckpt_dir),
        max_to_keep=15,
    )

    # building dummy models for restoration
    _ = gen(tf.zeros([1, cfg.LR_PATCH, cfg.LR_PATCH, 3]))
    _ = disc(tf.zeros([1, cfg.HR_PATCH, cfg.HR_PATCH, 3]))
    
    start_epoch = 0
    if try_resume:
        latest = ckpt_manager.latest_checkpoint # type: ignore
        if latest:
            ckpt.restore(latest).expect_partial()
            start_epoch = int(ckpt.epoch.numpy())
            tf.get_logger().info(
                f"Restored checkpoint from {latest}; resuming from epoch {start_epoch}",
            )

    pretrain_epochs = 1
    gen.compile(
        optimizer=g_opt,
        loss=mae,
        metrics=[psnr_metric, ssim_metric],
    )
    if start_epoch == 0:
        gen.fit(
            train_ds,
            steps_per_epoch=steps_per_epoch,
            epochs=pretrain_epochs,
            validation_data=val_ds,
            callbacks=[],
        )

    callbacks = [
        GeneratorCheckpoint(
            monitor="val_psnr_metric",
            mode="max",
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_psnr_metric",
            mode="max",
            factor=0.5,
            patience=6,
        ),
        SnapshotOnEpoch(
            [1, 5, 7, 10, 13, 15, 17, 18, 20, 23, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 38, 40],
            save_dir_root=log_root / Path("model_epoch_snapshots"),
            ckpt=ckpt,
            ckpt_manager=ckpt_manager,
        ),
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

    monitor = TrainingMonitor(logs_dir=log_root, save_freq_epochs=5, max_samples=3)
    callbacks.append(monitor)

    history = srgan.fit(
        train_ds,
        epochs=cfg.EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        callbacks=callbacks,
        initial_epoch=start_epoch,
    )

    (log_root / "history.json").write_text(json.dumps(history.history, indent=2))

    df = pd.DataFrame(history.history)
    df.index = df.index + 1
    df.index.name = "epoch"
    df.to_csv(log_root / "history.csv")

    gen.save(str(log_root / "generator_final.keras"))
    disc.save(str(log_root / "discriminator_final.keras"))

    return srgan, history
