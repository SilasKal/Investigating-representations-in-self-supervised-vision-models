import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_BASE_DIR = PROJECT_DIR / "MAPS_metrics_training"

BACKBONES = ("resnet18", "resnet50")
MODEL_TYPES = ("action", "ssl", "supervised")
MIN_EPOCH = 0
MAX_EPOCH = 50
COLORS = {
    "action": "tab:green",
    "ssl": "tab:orange",
    "supervised": "tab:blue",
}


def parse_progress_file(file_path: Path):
    train_points = {}
    test_points = {}

    with file_path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if not row:
                continue

            epoch_raw = (row.get("epoch") or "").strip()
            if not epoch_raw:
                continue

            try:
                epoch = int(float(epoch_raw))
            except ValueError:
                continue

            lin_acc_raw = (row.get("lin_acc0") or "").strip()
            test_acc_raw = (row.get("test_acc0") or "").strip()

            if lin_acc_raw:
                try:
                    train_points[epoch] = float(lin_acc_raw)
                except ValueError:
                    pass

            if test_acc_raw:
                try:
                    test_points[epoch] = float(test_acc_raw)
                except ValueError:
                    pass

    return train_points, test_points


def parse_supervised_file(file_path: Path):
    train_points = {}
    test_points = {}

    pattern = re.compile(
        r"\[sup\]\s+epoch=(\d+)\s+.*?train_acc=([0-9]*\.?[0-9]+)\s+test_acc=([0-9]*\.?[0-9]+)"
    )

    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = pattern.search(line)
            if not match:
                continue
            epoch = int(match.group(1))
            train_points[epoch] = float(match.group(2))
            test_points[epoch] = float(match.group(3))

    return train_points, test_points


def parse_run_file(file_path: Path, model_type: str):
    if model_type == "supervised":
        return parse_supervised_file(file_path)
    return parse_progress_file(file_path)


def collect_runs(backbone: str, model_type: str, base_dir: Path):
    model_dir = base_dir / backbone / model_type
    if model_type == "supervised":
        # Supports both naming styles: supervised_resnet*_v*.out and MAPS_supervised_*_*.out
        files = sorted(model_dir.glob("*.out"))
    else:
        files = sorted(model_dir.glob("progress_v*.txt"))

    run_data = {}
    for file_path in files:
        run_match = re.search(r"(?:_v|_)(\d+)$", file_path.stem)
        run_id = int(run_match.group(1)) if run_match else len(run_data) + 1
        train_points, test_points = parse_run_file(file_path, model_type)
        if train_points or test_points:
            run_data[run_id] = {"train": train_points, "test": test_points}

    return run_data


def aggregate_split(run_data, split: str):
    by_epoch = defaultdict(list)

    for run_metrics in run_data.values():
        for epoch, value in run_metrics[split].items():
            # Keep only the intended training horizon.
            if MIN_EPOCH <= epoch <= MAX_EPOCH:
                by_epoch[epoch].append(value)

    epochs = sorted(by_epoch)
    if not epochs:
        return np.array([]), np.array([]), np.array([])

    means = np.array([np.mean(by_epoch[epoch]) for epoch in epochs], dtype=float)
    stds = np.array([np.std(by_epoch[epoch], ddof=0) for epoch in epochs], dtype=float)
    return np.array(epochs, dtype=int), means, stds


def plot_backbone(backbone: str, base_dir: Path):
    fig, ax = plt.subplots(figsize=(12, 7))
    max_test_accs = {}

    for model_type in MODEL_TYPES:
        run_data = collect_runs(backbone, model_type, base_dir)
        if not run_data:
            continue

        color = COLORS[model_type]

        epochs_train, mean_train, std_train = aggregate_split(run_data, "train")
        epochs_test, mean_test, std_test = aggregate_split(run_data, "test")
        
        # Track max test accuracy
        if len(epochs_test) > 0:
            max_test_accs[model_type] = np.max(mean_test)
        
        if len(epochs_train) > 0:
            ax.plot(
                epochs_train,
                mean_train,
                color=color,
                linewidth=2,
                linestyle="-",
                label=f"{model_type} train (mean)",
            )
            ax.plot(
                epochs_train,
                np.clip(mean_train + std_train, 0.0, 1.0),
                color=color,
                linewidth=1,
                linestyle=":",
                alpha=0.6,
            )
            ax.plot(
                epochs_train,
                np.clip(mean_train - std_train, 0.0, 1.0),
                color=color,
                linewidth=1,
                linestyle=":",
                alpha=0.6,
            )
            ax.fill_between(
                epochs_train,
                np.clip(mean_train - std_train, 0.0, 1.0),
                np.clip(mean_train + std_train, 0.0, 1.0),
                color=color,
                alpha=0.15,
            )

        if len(epochs_test) > 0:
            ax.plot(
                epochs_test,
                mean_test,
                color=color,
                linewidth=2,
                linestyle="--",
                label=f"{model_type} test (mean)",
            )
            ax.plot(
                epochs_test,
                np.clip(mean_test + std_test, 0.0, 1.0),
                color=color,
                linewidth=1,
                linestyle=":",
                alpha=0.6,
            )
            ax.plot(
                epochs_test,
                np.clip(mean_test - std_test, 0.0, 1.0),
                color=color,
                linewidth=1,
                linestyle=":",
                alpha=0.6,
            )
            ax.fill_between(
                epochs_test,
                np.clip(mean_test - std_test, 0.0, 1.0),
                np.clip(mean_test + std_test, 0.0, 1.0),
                color=color,
                alpha=0.15,
            )

    # Title intentionally omitted per request.
    ax.set_xlabel("Epoch", fontsize=30)
    ax.set_ylabel("Accuracy", fontsize=30)
    ax.set_xlim(MIN_EPOCH, MAX_EPOCH)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(np.arange(MIN_EPOCH, MAX_EPOCH + 1, 5))
    # Set tick label font size for both axes
    ax.tick_params(axis="both", which="major", labelsize=20)
    ax.grid(True, alpha=0.25)

    output_file = base_dir / f"{backbone}_accuracy_mean_std.png"
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot: {output_file}")
    
    return max_test_accs


def main():
    parser = argparse.ArgumentParser(description="Plot MAPS training metrics with mean/std over runs.")
    parser.add_argument(
        "--base-dir",
        dest="base_dirs",
        action="append",
        help="Input metrics directory (absolute path or folder name relative to project root). Can be used multiple times.",
    )
    args = parser.parse_args()

    selected_dirs = args.base_dirs or [str(DEFAULT_BASE_DIR)]

    for base_dir_raw in selected_dirs:
        base_dir = Path(base_dir_raw)
        if not base_dir.is_absolute():
            base_dir = PROJECT_DIR / base_dir

        if not base_dir.exists():
            print(f"Skipping missing directory: {base_dir}")
            continue

        print(f"Using metrics directory: {base_dir}")
        print("\n" + "="*60)
        print(f"Max Test Accuracy for {base_dir.name}")
        print("="*60)
        
        for backbone in BACKBONES:
            max_test_accs = plot_backbone(backbone, base_dir)
            print(f"\n{backbone}:")
            for model_type in MODEL_TYPES:
                if model_type in max_test_accs:
                    print(f"  {model_type}: {max_test_accs[model_type]:.4f}")
                else:
                    print(f"  {model_type}: No data")


if __name__ == "__main__":
    main()
