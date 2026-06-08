import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.colors as mcolors

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_BASE_DIR = PROJECT_DIR / "results_params"
BACKBONES = ("resnet18", "resnet50")
# PARAM_GROUPS = ("all", "background", "camera", "light", "ssl", "supervised")
PARAM_GROUPS = ("all","background_saturation",  "camera_elevation", "camera_distance", "camera", "supervised",)

MIN_EPOCH = 0
MAX_EPOCH = 50

# Assign a unique color for each parameter group using the tab20 colormap.
PALETTE = [plt.get_cmap("tab20")(i) for i in range(20)]
COLORS = {pg: PALETTE[i % len(PALETTE)] for i, pg in enumerate(PARAM_GROUPS)}

# Keep fixed colors for the most important reference groups.
COLORS.update({
    "supervised": mcolors.to_rgba("tab:blue"),
    "ssl": mcolors.to_rgba("tab:orange"),
    "all": mcolors.to_rgba("tab:green"),
    "camera": mcolors.to_rgba("tab:red"),
    "camera_distance": mcolors.to_rgba("tab:brown"),
    "camera_elevation": mcolors.to_rgba("tab:purple"),
    # "background_saturation": mcolors.to_rgba("tab:blue"),
    "background_saturation": mcolors.to_rgba("turquoise"),
    "background": mcolors.to_rgba("#17becf"),
    "light": mcolors.to_rgba("#9467bd"),
})


LINESTYLES = {
    "train": "-",
    "test": "--",
}

LEGEND_LABELS = {
    "all": "All Parameters",
    "background": "Background",
    "camera": "All Camera Parameters",
    "light": "Light",
    "ssl": "SSL",
    "supervised": "Supervised",
    "camera_distance": "Camera Distance",
    "background_saturation" : "Background Saturation",
    "camera_elevation": "Camera Elevation"
}


def parse_supervised_file(file_path: Path):
    train_points = {}
    test_points = {}

    pattern = re.compile(
        r"\[sup]\s+epoch=(\d+)\s+.*?train_acc=([0-9]*\.?[0-9]+)\s+test_acc=([0-9]*\.?[0-9]+)"
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


def parse_metrics_file(file_path: Path):
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

            train_raw = (row.get("lin_acc0") or row.get("train_acc") or "").strip()
            test_raw = (row.get("test_acc0") or row.get("test_acc") or "").strip()

            if train_raw:
                try:
                    train_points[epoch] = float(train_raw)
                except ValueError:
                    pass

            if test_raw:
                try:
                    test_points[epoch] = float(test_raw)
                except ValueError:
                    pass

    return train_points, test_points


def collect_runs(backbone_dir: Path, param_group: str):
    if param_group == "supervised":
        run_files = sorted(backbone_dir.glob("*.out"))
        run_data = []

        for file_path in run_files:
            train_points, test_points = parse_supervised_file(file_path)

            if train_points or test_points:
                run_data.append(
                    {
                        "file": file_path,
                        "train": train_points,
                        "test": test_points,
                    }
                )

        return run_data

    progress_files = sorted(backbone_dir.rglob("progress_*.txt"))

    if not progress_files:
        progress_files = sorted(backbone_dir.rglob("linear_progressprojector0_*.txt"))

    run_data = []

    for file_path in progress_files:
        train_points, test_points = parse_metrics_file(file_path)

        if train_points or test_points:
            run_data.append(
                {
                    "file": file_path,
                    "train": train_points,
                    "test": test_points,
                }
            )

    return run_data


def aggregate_split(run_data, split: str):
    by_epoch = defaultdict(list)

    for run_metrics in run_data:
        for epoch, value in run_metrics[split].items():
            if MIN_EPOCH <= epoch <= MAX_EPOCH:
                by_epoch[epoch].append(value)

    epochs = sorted(by_epoch)

    if not epochs:
        return np.array([]), np.array([]), np.array([])

    means = np.array([np.mean(by_epoch[epoch]) for epoch in epochs], dtype=float)
    stds = np.array([np.std(by_epoch[epoch], ddof=0) for epoch in epochs], dtype=float)

    return np.array(epochs, dtype=int), means, stds


def plot_param_group(backbone: str, param_group: str, base_dir: Path):
    param_dir = base_dir / backbone / param_group

    if not param_dir.exists():
        print(f"Skipping {backbone}/{param_group}: directory does not exist")
        return None

    run_data = collect_runs(param_dir, param_group)

    if not run_data:
        print(f"Skipping {backbone}/{param_group}: no metric files found")
        return None

    return run_data


def plot_backbone_split(backbone: str, base_dir: Path, split: str):
    fig, ax = plt.subplots(figsize=(12, 7))

    has_data = False
    lines = []
    labels = []

    for param_group in PARAM_GROUPS:
        run_data = plot_param_group(backbone, param_group, base_dir)

        if run_data is None:
            continue

        has_data = True
        color = COLORS.get(param_group, "tab:gray")

        epochs, mean_values, std_values = aggregate_split(run_data, split)

        if len(epochs) == 0:
            continue

        legend_label = LEGEND_LABELS.get(param_group, param_group)

        line = ax.plot(
            epochs,
            mean_values,
            color=color,
            linewidth=5,
            linestyle=LINESTYLES[split],
            label=legend_label,
        )

        lines.extend(line)
        labels.append(legend_label)

        ax.fill_between(
            epochs,
            np.clip(mean_values - std_values, 0.0, 1.0),
            np.clip(mean_values + std_values, 0.0, 1.0),
            color=color,
            alpha=0.15,
        )

        if split == "test":
            max_acc = np.max(mean_values)
            print(f"{backbone}/{param_group}: max test acc = {max_acc:.4f}")

    if not has_data:
        print(f"Skipping {backbone}: no data found for any parameter group")
        plt.close(fig)
        return

    # Axis labels with fontsize 30
    ax.set_xlabel("Epoch", fontsize=20)
    ax.set_ylabel("Accuracy", fontsize=20)

    ax.set_xlim(MIN_EPOCH, MAX_EPOCH)
    ax.set_ylim(0.0, 0.75)
    ax.set_xticks(np.arange(MIN_EPOCH, MAX_EPOCH + 1, 5))

    # Tick labels with fontsize 20
    ax.tick_params(axis="both", which="major", labelsize=15)

    ax.grid(True, alpha=0.25)

    output_dir = base_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{backbone}_{split}_accuracy_mean_std.png"
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot: {output_file}")

    # # Separate legend figure.
    # if lines and labels:
    #     legend_fontsize = 18
    #
    #     fig_legend, ax_legend = plt.subplots(figsize=(9.5, 1.6))
    #     ax_legend.axis("off")
    #
    #     ax_legend.legend(
    #         lines,
    #         labels,
    #         loc="center",
    #         fontsize=legend_fontsize,
    #         frameon=True,
    #         ncol=3,  # 3 entries per row -> 2 rows for 6 labels
    #         handlelength=2.2,
    #         columnspacing=1.2,
    #         handletextpad=0.5,
    #         borderaxespad=0.0,
    #     )
    #
    #     legend_file = output_dir / f"{backbone}_{split}_legend.png"
    #     fig_legend.savefig(
    #         legend_file,
    #         dpi=300,
    #         bbox_inches="tight",
    #         pad_inches=0.03,
    #     )
    #     plt.close(fig_legend)
    #
    #     print(f"Saved legend: {legend_file}")
    # Separate horizontal legend figure.
    if lines and labels:
        legend_fontsize = 24

        # Wider and slightly taller figure to fit larger font
        fig_legend, ax_legend = plt.subplots(figsize=(20, 1.8))
        ax_legend.axis("off")

        ax_legend.legend(
            lines,
            labels,
            loc="center",
            fontsize=legend_fontsize,
            frameon=True,
            ncol=len(labels),  # one horizontal row
            handlelength=2.5,
            columnspacing=1.0,
            handletextpad=0.5,
            borderaxespad=0.0,
        )

        legend_file = output_dir / f"{backbone}_{split}_legend.png"
        fig_legend.savefig(
            legend_file,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.03,
        )
        plt.close(fig_legend)

        print(f"Saved legend: {legend_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot mean/std training and test accuracy for results_params."
    )

    parser.add_argument(
        "--base-dir",
        default=str(DEFAULT_BASE_DIR),
        help="Results directory, absolute path or folder name relative to the project root.",
    )

    args = parser.parse_args()

    base_dir = Path(args.base_dir)

    if not base_dir.is_absolute():
        base_dir = PROJECT_DIR / base_dir

    if not base_dir.exists():
        raise FileNotFoundError(f"Metrics directory does not exist: {base_dir}")

    print(f"Using metrics directory: {base_dir}")

    for backbone in BACKBONES:
        plot_backbone_split(backbone, base_dir, "train")
        plot_backbone_split(backbone, base_dir, "test")


if __name__ == "__main__":
    main()