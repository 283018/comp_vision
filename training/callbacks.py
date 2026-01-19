import contextlib
import json
from datetime import datetime
from pathlib import Path

import pytz
import tensorflow as tf
from tensorflow.train import Checkpoint, CheckpointManager

from config import cfg


class TrainingMonitor(tf.keras.callbacks.Callback):
    def __init__(
        self,
        logs_dir=cfg.LOG_ROOT,
        save_freq_epochs=5,
        max_samples=3,
        sample_ds=None,
    ):
        super().__init__()
        self.sample_ds = sample_ds
        self.log_root = Path(logs_dir) / "training"
        self.save_freq_epochs = save_freq_epochs
        self.max_samples = max_samples
        self.metrics_csv = self.log_root / "metrics.csv"
        self.samples_dir = self.log_root / "samples"
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        if not self.metrics_csv.exists():
            with Path.open(self.metrics_csv, "w") as f:
                f.write("epoch,datetime,loss,val_loss,psnr_metric,val_psnr_metric,ssim_metric,val_ssim_metric\n")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # workaround for new splitted metrics
        def _get(log_key_variants, default=""):
            for k in log_key_variants:
                if k in logs:
                    return logs.get(k)
            return default

        row = {
            "epoch": epoch + 1,
            "datetime": datetime.now(pytz.timezone("Poland")).isoformat(),
            "loss": _get(["loss", "g_loss", "generator_loss"], ""),
            "val_loss": logs.get("val_loss", ""),
            "psnr_metric": logs.get("psnr_metric", ""),
            "val_psnr_metric": logs.get("val_psnr_metric", ""),
            "ssim_metric": logs.get("ssim_metric", ""),
            "val_ssim_metric": logs.get("val_ssim_metric", ""),
        }
        with Path.open(self.metrics_csv, "a") as f:
            f.write(
                ",".join(
                    str(row[k])
                    for k in [
                        "epoch",
                        "datetime",
                        "loss",
                        "val_loss",
                        "psnr_metric",
                        "val_psnr_metric",
                        "ssim_metric",
                        "val_ssim_metric",
                    ]
                )
                + "\n",
            )

        snapshot = {
            "epoch": epoch + 1,
            "logs": {k: float(v) if isinstance(v, (int, float)) else v for k, v in logs.items()},
        }
        with Path.open(self.log_root / "last_epoch_snapshot.json", "w") as f:
            json.dump(snapshot, f, indent=2)


class SnapshotOnPlateau(tf.keras.callbacks.Callback):
    def __init__(
        self,
        monitor="val_loss",
        patience=12,
        save_dir_root=cfg.LOG_ROOT,
        snapshot_dir="plateau_snapshots",
        verbose=1,
    ):
        super().__init__()
        self.monitor = monitor
        self.patience = int(patience)
        self.save_dir = Path(save_dir_root) / "training" / snapshot_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = int(verbose)

        if "acc" in self.monitor or "accuracy" in self.monitor or "psnr" in self.monitor or "ssim" in self.monitor:
            self.monitor_op = lambda a, b: a > b
            self.best = -float("inf")
        else:
            self.monitor_op = lambda a, b: a < b
            self.best = float("inf")

        self.wait = 0
        self.best_weights = None

    def on_train_begin(self, logs=None):  # noqa: ARG002
        self.wait = 0
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            return

        try:
            current_val = float(current)
        except Exception:  # noqa: BLE001
            return

        if self.monitor_op(current_val, self.best):
            self.best = current_val
            try:
                self.best_weights = self.model.get_weights()
            except Exception:  # noqa: BLE001
                self.best_weights = None
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                ts = datetime.now(pytz.timezone("Poland")).isoformat(timespec="seconds")
                safe_metric = self.monitor.replace("/", "_")
                fname = f"snapshot_epoch{epoch + 1}_{safe_metric}_{current_val:.6f}_{ts}"
                try:
                    if hasattr(self.model, "generator"):
                        try:
                            self.best_weights = self.model.generator.get_weights()
                        except Exception:  # noqa: BLE001
                            self.best_weights = None
                    else:
                        try:
                            self.best_weights = self.model.get_weights()
                        except Exception:  # noqa: BLE001
                                self.best_weights = None

                # double except nesting lets goooo
                except Exception as e:  # noqa: BLE001
                    wtarget = str(self.save_dir / (fname + ".h5"))
                    print(f"\n[SnapshotOnPlateau] Save failed ({e}). Falling back to weights only save at {wtarget}")
                    try:
                        self.model.save_weights(wtarget)
                    except Exception as e2:  # noqa: BLE001
                        print(f"[SnapshotOnPlateau] Fallback save also failed: {e2}")

                self.wait = 0
                self.best = current_val


class SnapshotOnEpoch(tf.keras.callbacks.Callback):
    def __init__(  # noqa: PLR0913
        self,
        epochs,
        save_dir_root: str | Path = f"{cfg.LOG_ROOT}/training/model_epoch_snapshots",
        filename_template="model_epoch_{epoch}.keras",
        *,
        overwrite=True,
        ckpt: Checkpoint | None = None,
        ckpt_manager: CheckpointManager | None = None,
    ):
        super().__init__()
        self.epochs_to_save = {int(e) for e in epochs}
        self.save_dir = Path(save_dir_root)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.overwrite = bool(overwrite)
        self.filename_template = filename_template
        
        self.ckpt = ckpt
        self.ckpt_manager = ckpt_manager

    # messed something up here
    def ____on_train_begin(self, logs=None):  # noqa: ARG002
        if hasattr(self.model, "generator"):
            self.ckpt = Checkpoint(
                generator=self.model.generator,
                discriminator=self.model.discriminator,
                g_optimizer=self.model.g_optimizer,
                d_optimizer=self.model.d_optimizer,
                epoch=tf.Variable(0, trainable=False, dtype=tf.int64), # type: ignore
            )
    
    def _unique_path(self, path: Path) -> Path:
        if self.overwrite or not path.exists():
            return path
        base = path.stem
        suf = path.suffix
        i = 1
        while True:
            candidate = path.with_name(f"{base}_{i}{suf}")
            if not candidate.exists():
                return candidate
            i += 1

    def on_epoch_end(self, epoch, logs=None):  # noqa: ARG002
        current = int(epoch) + 1
        if current in self.epochs_to_save:
            fname = self.filename_template.format(epoch=current)
            target = self.save_dir / fname
            target = self._unique_path(target)

            try:
                if hasattr(self.model, "generator"):
                    gen_target = target.with_name(target.stem + "_generator" + target.suffix)
                    self.model.generator.save(str(gen_target))
                    tf.get_logger().info(f"Saved generator snapshot for epoch {current} -> {gen_target}")

                    if self.ckpt is not None:
                        with contextlib.suppress(Exception):
                            self.ckpt.epoch.assign(current)

                        if self.ckpt_manager is not None:
                            save_path = self.ckpt_manager.save() # type: ignore
                            tf.get_logger().info(f"Saved full training checkpoint (managed) -> {save_path}")
                        else:
                            ckpt_dir = target.with_name(target.stem + "_ckpt")
                            ckpt_dir.mkdir(parents=True, exist_ok=True)
                            # this save returns a path with numeric suffix
                            path = self.ckpt.save(str(ckpt_dir / "ckpt"))
                            tf.get_logger().info(f"Saved full training checkpoint -> {path}")
                    else:
                        tf.get_logger().warning("Checkpoint object not initialized; skipping full checkpoint save.")
                else:
                    self.model.save(str(target))
                    tf.get_logger().info(f"Saved snapshot for epoch {current} -> {target}")
            except Exception as e:  # noqa: BLE001
                tf.get_logger().error(f"Failed to save model for epoch {current}: {e}")

class GeneratorCheckpoint(tf.keras.callbacks.Callback):
    def __init__(
        self,
        monitor="val_psnr_metric",
        mode="max",
        save_dir=cfg.LOG_ROOT / "training" / "gen_checkpoints",
    ):
        super().__init__()
        self.monitor = monitor
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.best = -float("inf") if mode == "max" else float("inf")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val = logs.get(self.monitor)
        if val is None:
            return
        try:
            val = float(val)
        except Exception:  # noqa: BLE001
            return

        improved = val > self.best if self.mode == "max" else val < self.best
        if improved:
            self.best = val
            ts = datetime.now(pytz.timezone("Poland")).isoformat(timespec="seconds")
            fname = f"generator_epoch{epoch + 1}_{self.monitor}_{val:.6f}_{ts}.keras"
            target = self.save_dir / fname
            if hasattr(self.model, "generator"):
                try:
                    self.model.generator.save(str(target))
                    tf.get_logger().info(f"Saved improved generator -> {target}")
                except Exception as e:  # noqa: BLE001
                    tf.get_logger().warning(f"Failed to save generator: {e}")
