import json
from datetime import datetime
from pathlib import Path

import pytz
import tensorflow as tf

from config import cfg


class TrainingMonitor(tf.keras.callbacks.Callback):
    def __init__(
        self,
        log_root=cfg.LOG_ROOT,
        save_freq_epochs=5,
        max_samples=3,
        sample_ds=None,
    ):
        super().__init__()
        self.sample_ds = sample_ds
        self.log_root = Path(log_root)
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
        row = {
            "epoch": epoch + 1,
            "datetime": datetime.now(pytz.timezone("Poland")).isoformat(),
            "loss": logs.get("loss", ""),
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
        save_dir=cfg.LOG_ROOT,
        verbose=1,
    ):
        super().__init__()
        self.monitor = monitor
        self.patience = int(patience)
        self.save_dir = Path(save_dir)
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
                    target = str(self.save_dir / (fname + ".keras"))
                    print(f"\n[SnapshotOnPlateau] No improvement for {self.patience} epochs. Saving model to {target}")
                    self.model.save(target)

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
    def __init__(
        self,
        epochs,
        save_dir: str | Path = "/model_epoch_snapshots",
        filename_template="model_epoch_{epoch}.keras",
        *,
        overwrite=True,
    ):
        super().__init__()
        self.epochs_to_save = {int(e) for e in epochs}
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.overwrite = bool(overwrite)
        self.filename_template = filename_template

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
                self.model.save(str(target))
                tf.get_logger().info(f"Saved snapshot for epoch {current} -> {target}")
            except Exception as e:  # noqa: BLE001
                tf.get_logger().error(f"Failed to save model for epoch {current}: {e}")
