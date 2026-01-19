#!/usr/bin/env -S uv --quiet run --script
# /// script
# requires-python = "~=3.12"
# dependencies = [
#   "pillow",
# ]
# ///
import argparse
import sys
from pathlib import Path

from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def human_size(size_bytes):
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def analyze_directory(directory):
    count = total_size = total_width = total_height = 0

    for img_path in directory.iterdir():
        if img_path.suffix.lower() not in IMAGE_EXTS or not img_path.is_file():
            continue

        try:
            file_size = img_path.stat().st_size
            with Image.open(img_path) as img:
                width, height = img.size

            count += 1
            total_size += file_size
            total_width += width
            total_height += height
        except Exception as e:  # noqa: BLE001
            print(f"Skipping {img_path}: {e!s}", file=sys.stderr)

    avg_width = total_width / count if count else 0
    avg_height = total_height / count if count else 0

    return {
        "count": count,
        "total_size": total_size,
        "avg_resolution": (avg_width, avg_height),
    }


def main():  # noqa: PLR0915
    parser = argparse.ArgumentParser(description="Get image dataset stats")
    parser.add_argument(
        "root_dir",
        nargs="?",
        default=".",
        help="Root dir with datasets",
    )
    args = parser.parse_args()

    root = Path(args.root_dir)
    found_datasets = False

    global_train_count = global_test_count = 0
    global_train_size = global_test_size = 0
    global_train_width = global_train_height = 0.0
    global_test_width = global_test_height = 0.0

    for dataset_dir in root.iterdir():
        if not dataset_dir.is_dir():
            continue

        train_dir = dataset_dir / "train"
        test_dir = dataset_dir / "test"

        if not (train_dir.exists() and test_dir.exists()):
            continue

        found_datasets = True
        print(f"\n{'-' * 50}")
        print(f"DATASET: {dataset_dir.name}")
        print(f"{'-' * 50}")

        train_stats = analyze_directory(train_dir)
        test_stats = analyze_directory(test_dir)

        global_train_count += train_stats["count"]
        global_train_size += train_stats["total_size"]
        global_train_width += train_stats["avg_resolution"][0] * train_stats["count"]
        global_train_height += train_stats["avg_resolution"][1] * train_stats["count"]

        global_test_count += test_stats["count"]
        global_test_size += test_stats["total_size"]
        global_test_width += test_stats["avg_resolution"][0] * test_stats["count"]
        global_test_height += test_stats["avg_resolution"][1] * test_stats["count"]

        total_count = train_stats["count"] + test_stats["count"]
        total_size = train_stats["total_size"] + test_stats["total_size"]

        if total_count:
            combined_avg_width = (
                train_stats["avg_resolution"][0] * train_stats["count"]
                + test_stats["avg_resolution"][0] * test_stats["count"]
            ) / total_count

            combined_avg_height = (
                train_stats["avg_resolution"][1] * train_stats["count"]
                + test_stats["avg_resolution"][1] * test_stats["count"]
            ) / total_count
        else:
            combined_avg_width = combined_avg_height = 0

        print("\nTRAIN:")
        print(f"  Files: {train_stats['count']}")
        print(f"  Total size: {human_size(train_stats['total_size'])}")
        print(f"  Avg res: {train_stats['avg_resolution'][0]:.0f}x{train_stats['avg_resolution'][1]:.0f}")

        print("\nTEST:")
        print(f"  Files: {test_stats['count']}")
        print(f"  Total size: {human_size(test_stats['total_size'])}")
        print(f"  Avg res: {test_stats['avg_resolution'][0]:.0f}x{test_stats['avg_resolution'][1]:.0f}")

        print("\nCOMBINED:")
        print(f"  Total files: {total_count}")
        print(f"  Total size: {human_size(total_size)}")
        print(f"  Avg res: {combined_avg_width:.0f}x{combined_avg_height:.0f}")

    if not found_datasets:
        print("No valid datasets found. Directories must contain both 'train' and 'test' subdirs.")
        sys.exit(1)

    global_train_avg_w = global_train_width / global_train_count if global_train_count else 0
    global_train_avg_h = global_train_height / global_train_count if global_train_count else 0

    global_test_avg_w = global_test_width / global_test_count if global_test_count else 0
    global_test_avg_h = global_test_height / global_test_count if global_test_count else 0

    global_combined_count = global_train_count + global_test_count
    global_combined_size = global_train_size + global_test_size
    global_combined_width = global_train_width + global_test_width
    global_combined_height = global_train_height + global_test_height

    global_combined_avg_w = global_combined_width / global_combined_count if global_combined_count else 0
    global_combined_avg_h = global_combined_height / global_combined_count if global_combined_count else 0

    print(f"\n\n\n{'=' * 60}")
    print("GLOBAL SUMMARY")
    print(f"{'=' * 60}")

    print("\nTRAIN COMBINED:")
    print(f"  Total files: {global_train_count}")
    print(f"  Total size: {human_size(global_train_size)}")
    print(f"  Avg res: {global_train_avg_w:.0f}x{global_train_avg_h:.0f}")

    print("\nTEST COMBINED:")
    print(f"  Total files: {global_test_count}")
    print(f"  Total size: {human_size(global_test_size)}")
    print(f"  Avg res: {global_test_avg_w:.0f}x{global_test_avg_h:.0f}")

    print("\nOVERALL:")
    print(f"  Total files: {global_combined_count}")
    print(f"  Total size: {human_size(global_combined_size)}")
    print(f"  Avg res: {global_combined_avg_w:.0f}x{global_combined_avg_h:.0f}")


if __name__ == "__main__":
    main()
