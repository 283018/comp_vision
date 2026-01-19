import json
import sys
import tempfile
from pathlib import Path

from icecream import ic  # noqa: PLC0415
from numpy import full

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import cfg
from data.collector import RunMode, _ensure_index, build_dataset, image_iterator


def test_iterator_consistency():
    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = Path(tmpdir) / "test_index.json"

        full_rel_paths = [
            str(p.relative_to(cfg.DATA_DIR))
            for sub in cfg.DATA_DIR.iterdir()
            for p in sub.rglob("*")
            if p.is_file() and not p.name.startswith(".")
        ]
        full_rel_set = set(full_rel_paths)

        train_abs_files = list(
            image_iterator(
                RunMode.TRAIN,
                index_path=index_path,
                split_ratios=(0.7, 0.2, 0.1),
                data_dir=cfg.DATA_DIR,
                seed=123321,
                regenerate_index=True,
            ),
        )
        # Second: Create test iterator
        test_abs_files_first = list(
            image_iterator(
                RunMode.TEST,
                index_path=index_path,
                split_ratios=(0.7, 0.2, 0.1),
                data_dir=cfg.DATA_DIR,
                seed=123321,
                regenerate_index=False,
            ),
        )
        test_abs_files_second = list(
            image_iterator(
                RunMode.TEST,
                index_path=index_path,
                split_ratios=(0.7, 0.2, 0.1),
                data_dir=cfg.DATA_DIR,
                seed=123321,
                regenerate_index=False,
            ),
        )
        eval_abs_files_first = list(
            image_iterator(
                RunMode.EVAL,
                index_path=index_path,
                split_ratios=(0.7, 0.2, 0.1),
                data_dir=cfg.DATA_DIR,
                seed=123321,
                regenerate_index=False,
            ),
        )
        eval_abs_files_second = list(
            image_iterator(
                RunMode.EVAL,
                index_path=index_path,
                split_ratios=(0.7, 0.2, 0.1),
                data_dir=cfg.DATA_DIR,
                seed=123321,
                regenerate_index=False,
            ),
        )

        train_rel_set = {str(Path(p).relative_to(cfg.DATA_DIR)) for p in train_abs_files}
        test_rel_set_first = {str(Path(p).relative_to(cfg.DATA_DIR)) for p in test_abs_files_first}
        test_rel_set_second = {str(Path(p).relative_to(cfg.DATA_DIR)) for p in test_abs_files_second}
        eval_rel_set_first = {str(Path(p).relative_to(cfg.DATA_DIR)) for p in eval_abs_files_first}
        eval_rel_set_second = {str(Path(p).relative_to(cfg.DATA_DIR)) for p in eval_abs_files_second}

        assert test_rel_set_first == test_rel_set_second, "test files not identical across creations"  # noqa: S101
        assert eval_rel_set_first == eval_rel_set_second, "eval files not identical across creations"  # noqa: S101

        assert not (train_rel_set & test_rel_set_first), "train and test overlap"  # noqa: S101
        assert not (train_rel_set & eval_rel_set_first), "train and eval overlap"  # noqa: S101
        assert not (test_rel_set_first & eval_rel_set_first), "test and eval overlap"  # noqa: S101

        combined_rel = train_rel_set | test_rel_set_first | eval_rel_set_first
        missing_rel = full_rel_set - combined_rel
        assert len(missing_rel) <= 3, f"some files missing: {len(missing_rel)}"  # noqa: S101

        with Path.open(index_path) as f:
            index = json.load(f)

        indexed_rel_paths = set(index["files"].keys())
        assert indexed_rel_paths - full_rel_set == set(), "index contains files not in dataset"  # noqa: S101

        unused_files = [f for f, s in index["files"].items() if s == str(RunMode.UNUSED)]
        assert len(unused_files) == 0, f"unused files remain: {len(unused_files)}"  # noqa: S101

        total_assigned = len(combined_rel)
        train_ratio = len(train_rel_set) / total_assigned
        test_ratio = len(test_rel_set_first) / total_assigned
        eval_ratio = len(eval_rel_set_first) / total_assigned

        assert 0.65 <= train_ratio <= 0.75, f"train ratio {train_ratio:.2f} outside range"  # noqa: S101
        assert 0.15 <= test_ratio <= 0.25, f"test ratio {test_ratio:.2f} outside range"  # noqa: S101
        assert 0.05 <= eval_ratio <= 0.15, f"eval ratio {eval_ratio:.2f} outside range"  # noqa: S101

    print("\nAll tests passed")


if __name__ == "__main__":
    test_iterator_consistency()
