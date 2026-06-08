from __future__ import annotations

import argparse
import csv
import math
import re
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# GLOBAL STYLE SETTINGS
# Change all plot/legend/font sizes here.
# ============================================================

FIG_DPI = 300

# Axis labels
X_AXIS_LABEL_FONTSIZE = 55
Y_AXIS_LABEL_FONTSIZE = 55

# Tick labels
X_TICK_LABEL_FONTSIZE = 40
Y_TICK_LABEL_FONTSIZE = 40

# Legend
LEGEND_FONTSIZE = 75
LEGEND_TITLE_FONTSIZE = 20
LEGEND_FIGSIZE = (18, 1.8)

# Lines / grid / uncertainty bands
LINEWIDTH = 20
STD_ALPHA = 0.18
GRID_ALPHA = 0

# Figure sizes
TRAIN_TEST_FIGSIZE = (20, 10)
SINGLE_FIGSIZE = (20, 15)

# Axis label text
X_LABEL = "Epoch"
Y_LABEL = "Accuracy"


MIXTURE_ORDER = ["100_0", "75_25", "50_50", "25_75", "15_85", "10_90", "0_100"]

# Manual color mapping for maximum distinguishability of bottom 4-5 mixtures.
# Start at very dark purple, progressing through dark blue -> light blue -> dark green -> green -> yellow-green -> yellow.
_MANUAL_COLOR_MAP = {
    "0_100": "#1A0033",      # Very dark purple
    "10_90": "#00008B",     # Dark blue
    "15_85": "#87CEEB",     # Light blue
    "25_75": "#228B22",     # Dark green
    "50_50": "#32CD32",     # Green
    "75_25": "#ADFF2F",     # Yellow-green
    "100_0": "#FFD700",     # Yellow/gold
}


@dataclass
class MixMetrics:
    epochs: list[int]
    train_acc: list[float]
    imagenet_test_acc: list[float]
    maps_test_acc: list[float]


@dataclass
class AggregatedMixMetrics:
    epochs: list[int]
    train_mean: list[float]
    train_std: list[float]
    imagenet_test_mean: list[float]
    imagenet_test_std: list[float]
    maps_test_mean: list[float]
    maps_test_std: list[float]
    n_runs: int


_EPOCH_RE = re.compile(re.escape("[mix]") + r"\s*epoch=(\d+)", re.IGNORECASE)
_TRAIN_ACC_RE = re.compile(r"train_acc=([0-9]*\.?[0-9]+)", re.IGNORECASE)
_IMAGENET_TEST_ACC_RE = re.compile(r"imagenet_test_acc=([0-9]*\.?[0-9]+)", re.IGNORECASE)
_MAPS_TEST_ACC_RE = re.compile(r"maps_test_acc=([0-9]*\.?[0-9]+)", re.IGNORECASE)


def _read_lines(path: Path) -> list[str]:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            with path.open("r", encoding=encoding, errors="replace") as handle:
                return handle.readlines()
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Could not read {path} with fallback encodings.")


def parse_mix_log(path: Path) -> MixMetrics:
    epochs: list[int] = []
    train_acc: list[float] = []
    imagenet_test_acc: list[float] = []
    maps_test_acc: list[float] = []

    for line in _read_lines(path):
        epoch_match = _EPOCH_RE.search(line)
        train_match = _TRAIN_ACC_RE.search(line)
        imagenet_test_match = _IMAGENET_TEST_ACC_RE.search(line)
        maps_test_match = _MAPS_TEST_ACC_RE.search(line)

        if not (epoch_match and train_match and imagenet_test_match and maps_test_match):
            continue

        epochs.append(int(epoch_match.group(1)))
        train_acc.append(float(train_match.group(1)))
        imagenet_test_acc.append(float(imagenet_test_match.group(1)))
        maps_test_acc.append(float(maps_test_match.group(1)))

    if not epochs:
        raise ValueError(
            f"No mix metrics found in {path}. Check that the log file contains '[mix]' lines "
            "with epoch, train_acc, imagenet_test_acc, and maps_test_acc metrics."
        )

    # Remove duplicates and keep latest value for duplicated epochs.
    deduped: dict[int, tuple[float, float, float]] = {}
    for epoch, train, imagenet_test, maps_test in zip(epochs, train_acc, imagenet_test_acc, maps_test_acc):
        deduped[epoch] = (train, imagenet_test, maps_test)

    ordered = sorted(deduped.items())
    epochs = [item[0] for item in ordered]
    train_acc = [item[1][0] for item in ordered]
    imagenet_test_acc = [item[1][1] for item in ordered]
    maps_test_acc = [item[1][2] for item in ordered]

    return MixMetrics(
        epochs=epochs,
        train_acc=train_acc,
        imagenet_test_acc=imagenet_test_acc,
        maps_test_acc=maps_test_acc,
    )


def _is_numbered_run(path: Path) -> bool:
    """Check if this file is a numbered run.

    Examples:
      - 25_75_1.out -> True
      - 25_75_scratch_1.out -> True
      - 25_75_resnet50_1.out -> True
      - 25_75.out -> False
      - 25_75_scratch.out -> False
      - 25_75_resnet50.out -> False
    """
    name = path.stem
    match = re.match(r"^(\d+_\d+)(?:_(.*))?$", name)
    if not match:
        return False

    suffix = match.group(2)
    if not suffix:
        return False

    known_modifiers = {"scratch", "resnet50"}
    tokens = suffix.split("_")

    if tokens[-1].isdigit():
        if len(tokens) == 1:
            return True

        modifiers = set(tokens[:-1])
        return modifiers <= known_modifiers

    return False


def _parse_file_attributes(path: Path) -> tuple[str, bool, bool]:
    name = path.stem

    # Handles:
    # - 100_0
    # - 100_0_scratch
    # - 100_0_scratch_1
    # - 100_0_resnet50
    # - 100_0_resnet50_1
    match = re.match(r"^(\d+_\d+)(?:_(.*))?$", name)
    if not match:
        return name, False, False

    mix_name = match.group(1)
    suffix = match.group(2) or ""
    tokens = [token for token in suffix.split("_") if token]

    # Remove run numbers.
    tokens = [token for token in tokens if not token.isdigit()]

    is_scratch = "scratch" in tokens
    is_resnet50 = "resnet50" in tokens

    return mix_name, is_scratch, is_resnet50


def _mixture_key(path: Path) -> str:
    mix_name, _, _ = _parse_file_attributes(path)
    return mix_name


def _is_scratch(path: Path) -> bool:
    _, is_scratch, _ = _parse_file_attributes(path)
    return is_scratch


def _is_resnet50(path: Path) -> bool:
    _, _, is_resnet50 = _parse_file_attributes(path)
    return is_resnet50


def _sorted_mix_files(paths: Iterable[Path]) -> list[Path]:
    order = {name: idx for idx, name in enumerate(MIXTURE_ORDER)}

    def sort_key(path: Path) -> tuple[int, str]:
        mix_name = _mixture_key(path)
        return order.get(mix_name, len(order)), mix_name

    return sorted(paths, key=sort_key)


def _color_for_mixture(mix_name: str) -> str:
    if mix_name in _MANUAL_COLOR_MAP:
        return _MANUAL_COLOR_MAP[mix_name]
    return "#808080"


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0

    mean = sum(values) / len(values)

    if len(values) == 1:
        return mean, 0.0

    # Sample standard deviation across runs.
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)

    return mean, math.sqrt(variance)


def _aggregate_runs(files: list[Path]) -> dict[str, AggregatedMixMetrics]:
    grouped_files: dict[str, list[Path]] = defaultdict(list)

    for file_path in files:
        grouped_files[_mixture_key(file_path)].append(file_path)

    aggregated: dict[str, AggregatedMixMetrics] = {}

    for mix_name, mix_files in grouped_files.items():
        # Prefer numbered runs over base files when both exist.
        numbered_files = [file_path for file_path in mix_files if _is_numbered_run(file_path)]
        if numbered_files:
            mix_files = numbered_files

        mix_files = sorted(mix_files, key=lambda path: path.name)

        train_by_epoch: dict[int, list[float]] = defaultdict(list)
        imagenet_by_epoch: dict[int, list[float]] = defaultdict(list)
        maps_by_epoch: dict[int, list[float]] = defaultdict(list)

        successfully_parsed_count = 0

        for file_path in mix_files:
            try:
                metrics = parse_mix_log(file_path)
            except ValueError as exc:
                warnings.warn(f"Skipping {file_path.name}: {exc}")
                continue

            successfully_parsed_count += 1

            run_train_by_epoch: dict[int, float] = {}
            run_imagenet_by_epoch: dict[int, float] = {}
            run_maps_by_epoch: dict[int, float] = {}

            for epoch, train, imagenet_test, maps_test in zip(
                metrics.epochs,
                metrics.train_acc,
                metrics.imagenet_test_acc,
                metrics.maps_test_acc,
            ):
                run_train_by_epoch[epoch] = train
                run_imagenet_by_epoch[epoch] = imagenet_test
                run_maps_by_epoch[epoch] = maps_test

            for epoch in sorted(run_train_by_epoch.keys()):
                train_by_epoch[epoch].append(run_train_by_epoch[epoch])
                imagenet_by_epoch[epoch].append(run_imagenet_by_epoch[epoch])
                maps_by_epoch[epoch].append(run_maps_by_epoch[epoch])

        if not train_by_epoch:
            warnings.warn(f"Mix '{mix_name}' has no valid runs; skipping.")
            continue

        if successfully_parsed_count != 5:
            warnings.warn(
                f"Mix '{mix_name}' has {successfully_parsed_count} runs; "
                "std is computed across available runs (expected 5)."
            )

        epochs = sorted(train_by_epoch.keys())

        train_mean: list[float] = []
        train_std: list[float] = []
        imagenet_test_mean: list[float] = []
        imagenet_test_std: list[float] = []
        maps_test_mean: list[float] = []
        maps_test_std: list[float] = []

        for epoch in epochs:
            mean, std = _mean_std(train_by_epoch[epoch])
            train_mean.append(mean)
            train_std.append(std)

            mean, std = _mean_std(imagenet_by_epoch[epoch])
            imagenet_test_mean.append(mean)
            imagenet_test_std.append(std)

            mean, std = _mean_std(maps_by_epoch[epoch])
            maps_test_mean.append(mean)
            maps_test_std.append(std)

        aggregated[mix_name] = AggregatedMixMetrics(
            epochs=epochs,
            train_mean=train_mean,
            train_std=train_std,
            imagenet_test_mean=imagenet_test_mean,
            imagenet_test_std=imagenet_test_std,
            maps_test_mean=maps_test_mean,
            maps_test_std=maps_test_std,
            n_runs=successfully_parsed_count,
        )

    return aggregated


def _sorted_mix_names(names: Iterable[str]) -> list[str]:
    order = {name: idx for idx, name in enumerate(MIXTURE_ORDER)}
    return sorted(names, key=lambda name: (order.get(name, len(order)), name))


def _format_axis(ax: plt.Axes) -> None:
    ax.set_xlabel(X_LABEL, fontsize=X_AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(Y_LABEL, fontsize=Y_AXIS_LABEL_FONTSIZE)

    ax.tick_params(axis="x", labelsize=X_TICK_LABEL_FONTSIZE)
    ax.tick_params(axis="y", labelsize=Y_TICK_LABEL_FONTSIZE)

    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=GRID_ALPHA)


def _save_figure(fig: plt.Figure, output_path: Path) -> None:
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def save_separate_legend(files: list[Path], output_path: Path) -> bool:
    """Save the mixture legend as a separate image.

    Returns True if a legend was created and False if no valid metrics were available.
    """
    if not files:
        return False

    grouped_metrics = _aggregate_runs(files)
    if not grouped_metrics:
        return False

    fig, ax = plt.subplots(figsize=LEGEND_FIGSIZE)

    handles = []
    labels = []

    for mix_name in _sorted_mix_names(grouped_metrics.keys()):
        color = _color_for_mixture(mix_name)
        n_runs = grouped_metrics[mix_name].n_runs

        handles.append(
            plt.Line2D(
                [0],
                [0],
                color=color,
                linewidth=LINEWIDTH,
            )
        )
        # Make the legend label more readable by showing mixtures with a slash
        # e.g. show "color 100/0 (n=5)" instead of "100_0 (n=5)".
        pretty_mix = mix_name.replace("_", "/")
        labels.append(f"{pretty_mix} (n={n_runs})")

    ax.legend(
        handles,
        labels,
        loc="center",
        ncol=len(labels),
        fontsize=LEGEND_FONTSIZE,
        title=None,
        title_fontsize=LEGEND_TITLE_FONTSIZE,
        frameon=False,
    )
    ax.axis("off")

    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)

    return True


def _plot_group(ax_train: plt.Axes, ax_test: plt.Axes, files: list[Path], title: str) -> None:
    if not files:
        ax_train.set_visible(False)
        ax_test.set_visible(False)
        return

    grouped_metrics = _aggregate_runs(files)

    for mix_name in _sorted_mix_names(grouped_metrics.keys()):
        metrics = grouped_metrics[mix_name]
        color = _color_for_mixture(mix_name)

        train_lower = [max(0.0, mean - std) for mean, std in zip(metrics.train_mean, metrics.train_std)]
        train_upper = [min(1.0, mean + std) for mean, std in zip(metrics.train_mean, metrics.train_std)]

        imagenet_lower = [
            max(0.0, mean - std)
            for mean, std in zip(metrics.imagenet_test_mean, metrics.imagenet_test_std)
        ]
        imagenet_upper = [
            min(1.0, mean + std)
            for mean, std in zip(metrics.imagenet_test_mean, metrics.imagenet_test_std)
        ]

        ax_train.plot(
            metrics.epochs,
            metrics.train_mean,
            color=color,
            linewidth=LINEWIDTH,
        )
        ax_train.fill_between(
            metrics.epochs,
            train_lower,
            train_upper,
            color=color,
            alpha=STD_ALPHA,
        )

        ax_test.plot(
            metrics.epochs,
            metrics.imagenet_test_mean,
            color=color,
            linewidth=LINEWIDTH,
        )
        ax_test.fill_between(
            metrics.epochs,
            imagenet_lower,
            imagenet_upper,
            color=color,
            alpha=STD_ALPHA,
        )

    _format_axis(ax_train)
    _format_axis(ax_test)


def _plot_imagenet_test_group(ax_test: plt.Axes, files: list[Path], title: str) -> None:
    if not files:
        ax_test.set_visible(False)
        return

    grouped_metrics = _aggregate_runs(files)

    for mix_name in _sorted_mix_names(grouped_metrics.keys()):
        metrics = grouped_metrics[mix_name]
        color = _color_for_mixture(mix_name)

        imagenet_lower = [
            max(0.0, mean - std)
            for mean, std in zip(metrics.imagenet_test_mean, metrics.imagenet_test_std)
        ]
        imagenet_upper = [
            min(1.0, mean + std)
            for mean, std in zip(metrics.imagenet_test_mean, metrics.imagenet_test_std)
        ]

        ax_test.plot(
            metrics.epochs,
            metrics.imagenet_test_mean,
            color=color,
            linewidth=LINEWIDTH,
        )
        ax_test.fill_between(
            metrics.epochs,
            imagenet_lower,
            imagenet_upper,
            color=color,
            alpha=STD_ALPHA,
        )

    _format_axis(ax_test)


def _plot_maps_test_group(ax_maps: plt.Axes, files: list[Path], title: str) -> None:
    if not files:
        ax_maps.set_visible(False)
        return

    grouped_metrics = _aggregate_runs(files)

    for mix_name in _sorted_mix_names(grouped_metrics.keys()):
        metrics = grouped_metrics[mix_name]
        color = _color_for_mixture(mix_name)

        maps_lower = [
            max(0.0, mean - std)
            for mean, std in zip(metrics.maps_test_mean, metrics.maps_test_std)
        ]
        maps_upper = [
            min(1.0, mean + std)
            for mean, std in zip(metrics.maps_test_mean, metrics.maps_test_std)
        ]

        ax_maps.plot(
            metrics.epochs,
            metrics.maps_test_mean,
            color=color,
            linewidth=LINEWIDTH,
        )
        ax_maps.fill_between(
            metrics.epochs,
            maps_lower,
            maps_upper,
            color=color,
            alpha=STD_ALPHA,
        )

    _format_axis(ax_maps)


def _collect_files(metrics_dir: Path, scratch: bool, resnet50: bool) -> list[Path]:
    files: list[Path] = []

    # Recursively collect logs to support nested layouts, e.g. pretrained/resnet50/*.out.
    for path in metrics_dir.rglob("*.out"):
        if not path.is_file():
            continue

        if _is_scratch(path) != scratch:
            continue

        if _is_resnet50(path) != resnet50:
            continue

        files.append(path)

    return files


def _default_metrics_dir() -> Path:
    base = Path(__file__).resolve().parent

    for dirname in ("training_metrics_mix", "training_mtrics_mix"):
        candidate = base / dirname
        if candidate.exists():
            return candidate

    return base / "training_metrics_mix"


def _save_train_accuracy_data(files: list[Path], output_file: Path) -> None:
    if not files:
        return

    grouped_metrics = _aggregate_runs(files)

    with output_file.open("w", newline="") as csvfile:
        fieldnames = [
            "mixture",
            "n_runs",
            "epoch",
            "train_mean",
            "train_std",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for mix_name in _sorted_mix_names(grouped_metrics.keys()):
            metrics = grouped_metrics[mix_name]

            for epoch, train_mean, train_std in zip(
                metrics.epochs,
                metrics.train_mean,
                metrics.train_std,
            ):
                writer.writerow(
                    {
                        "mixture": mix_name,
                        "n_runs": metrics.n_runs,
                        "epoch": epoch,
                        "train_mean": f"{train_mean:.6f}",
                        "train_std": f"{train_std:.6f}",
                    }
                )


def _save_imagenet_test_accuracy_data(files: list[Path], output_file: Path) -> None:
    if not files:
        return

    grouped_metrics = _aggregate_runs(files)

    with output_file.open("w", newline="") as csvfile:
        fieldnames = [
            "mixture",
            "n_runs",
            "epoch",
            "imagenet_test_mean",
            "imagenet_test_std",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for mix_name in _sorted_mix_names(grouped_metrics.keys()):
            metrics = grouped_metrics[mix_name]

            for epoch, imagenet_mean, imagenet_std in zip(
                metrics.epochs,
                metrics.imagenet_test_mean,
                metrics.imagenet_test_std,
            ):
                writer.writerow(
                    {
                        "mixture": mix_name,
                        "n_runs": metrics.n_runs,
                        "epoch": epoch,
                        "imagenet_test_mean": f"{imagenet_mean:.6f}",
                        "imagenet_test_std": f"{imagenet_std:.6f}",
                    }
                )


def _save_maps_test_accuracy_data(files: list[Path], output_file: Path) -> None:
    if not files:
        return

    grouped_metrics = _aggregate_runs(files)

    with output_file.open("w", newline="") as csvfile:
        fieldnames = [
            "mixture",
            "n_runs",
            "epoch",
            "maps_test_mean",
            "maps_test_std",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for mix_name in _sorted_mix_names(grouped_metrics.keys()):
            metrics = grouped_metrics[mix_name]

            for epoch, maps_mean, maps_std in zip(
                metrics.epochs,
                metrics.maps_test_mean,
                metrics.maps_test_std,
            ):
                writer.writerow(
                    {
                        "mixture": mix_name,
                        "n_runs": metrics.n_runs,
                        "epoch": epoch,
                        "maps_test_mean": f"{maps_mean:.6f}",
                        "maps_test_std": f"{maps_std:.6f}",
                    }
                )


def _append_if_exists(paths: list[Path], path: Path) -> None:
    if path.exists():
        paths.append(path)


def make_mix_plots(metrics_dir: Path, output_dir: Path, debug: bool = False) -> tuple[Path, ...]:
    base_pretrained_files = _collect_files(metrics_dir, scratch=False, resnet50=False)
    base_scratch_files = _collect_files(metrics_dir, scratch=True, resnet50=False)
    resnet50_pretrained_files = _collect_files(metrics_dir, scratch=False, resnet50=True)
    resnet50_scratch_files = _collect_files(metrics_dir, scratch=True, resnet50=True)

    if debug:
        print(f"[DEBUG] Base pretrained files found: {len(base_pretrained_files)}")
        for file_path in sorted(base_pretrained_files, key=lambda x: x.name):
            print(f"  - {file_path.name}")

        print(f"[DEBUG] Base scratch files found: {len(base_scratch_files)}")
        for file_path in sorted(base_scratch_files, key=lambda x: x.name):
            print(f"  - {file_path.name}")

        print(f"[DEBUG] ResNet50 pretrained files found: {len(resnet50_pretrained_files)}")
        for file_path in sorted(resnet50_pretrained_files, key=lambda x: x.name):
            print(f"  - {file_path.name}")

        print(f"[DEBUG] ResNet50 scratch files found: {len(resnet50_scratch_files)}")
        for file_path in sorted(resnet50_scratch_files, key=lambda x: x.name):
            print(f"  - {file_path.name}")

    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths: list[Path] = []

    base_pretrained_out = output_dir / "pretrained_mix_accuracy.png"
    base_scratch_out = output_dir / "scratch_mix_accuracy.png"
    resnet50_pretrained_out = output_dir / "resnet50_pretrained_mix_accuracy.png"
    resnet50_scratch_out = output_dir / "resnet50_scratch_mix_accuracy.png"

    base_pretrained_maps_out = output_dir / "pretrained_mix_maps_test_accuracy.png"
    base_scratch_maps_out = output_dir / "scratch_mix_maps_test_accuracy.png"
    resnet50_pretrained_maps_out = output_dir / "resnet50_pretrained_mix_maps_test_accuracy.png"
    resnet50_scratch_maps_out = output_dir / "resnet50_scratch_mix_maps_test_accuracy.png"

    base_pretrained_imagenet_out = output_dir / "pretrained_mix_imagenet_test_accuracy.png"
    base_scratch_imagenet_out = output_dir / "scratch_mix_imagenet_test_accuracy.png"
    resnet50_pretrained_imagenet_out = output_dir / "resnet50_pretrained_mix_imagenet_test_accuracy.png"
    resnet50_scratch_imagenet_out = output_dir / "resnet50_scratch_mix_imagenet_test_accuracy.png"

    base_pretrained_legend_out = output_dir / "pretrained_mix_legend.png"
    base_scratch_legend_out = output_dir / "scratch_mix_legend.png"
    resnet50_pretrained_legend_out = output_dir / "resnet50_pretrained_mix_legend.png"
    resnet50_scratch_legend_out = output_dir / "resnet50_scratch_mix_legend.png"

    base_pretrained_train_csv = output_dir / "pretrained_mix_train_accuracy.csv"
    base_pretrained_test_csv = output_dir / "pretrained_mix_imagenet_test_accuracy.csv"
    base_pretrained_maps_csv = output_dir / "pretrained_mix_maps_test_accuracy.csv"

    base_scratch_train_csv = output_dir / "scratch_mix_train_accuracy.csv"
    base_scratch_test_csv = output_dir / "scratch_mix_imagenet_test_accuracy.csv"
    base_scratch_maps_csv = output_dir / "scratch_mix_maps_test_accuracy.csv"

    resnet50_pretrained_train_csv = output_dir / "resnet50_pretrained_mix_train_accuracy.csv"
    resnet50_pretrained_test_csv = output_dir / "resnet50_pretrained_mix_imagenet_test_accuracy.csv"
    resnet50_pretrained_maps_csv = output_dir / "resnet50_pretrained_mix_maps_test_accuracy.csv"

    resnet50_scratch_train_csv = output_dir / "resnet50_scratch_mix_train_accuracy.csv"
    resnet50_scratch_test_csv = output_dir / "resnet50_scratch_mix_imagenet_test_accuracy.csv"
    resnet50_scratch_maps_csv = output_dir / "resnet50_scratch_mix_maps_test_accuracy.csv"

    # ============================================================
    # ResNet18 pretrained
    # ============================================================

    fig, axes = plt.subplots(1, 2, figsize=TRAIN_TEST_FIGSIZE)
    _plot_group(axes[0], axes[1], base_pretrained_files, "ResNet18 Pretrained")
    _save_figure(fig, base_pretrained_out)
    output_paths.append(base_pretrained_out)

    fig, ax = plt.subplots(1, 1, figsize=SINGLE_FIGSIZE)
    _plot_imagenet_test_group(ax, base_pretrained_files, "ResNet18 Pretrained")
    _save_figure(fig, base_pretrained_imagenet_out)
    output_paths.append(base_pretrained_imagenet_out)

    fig, ax = plt.subplots(1, 1, figsize=SINGLE_FIGSIZE)
    _plot_maps_test_group(ax, base_pretrained_files, "ResNet18 Pretrained")
    _save_figure(fig, base_pretrained_maps_out)
    output_paths.append(base_pretrained_maps_out)

    if save_separate_legend(base_pretrained_files, base_pretrained_legend_out):
        output_paths.append(base_pretrained_legend_out)

    _save_train_accuracy_data(base_pretrained_files, base_pretrained_train_csv)
    _save_imagenet_test_accuracy_data(base_pretrained_files, base_pretrained_test_csv)
    _save_maps_test_accuracy_data(base_pretrained_files, base_pretrained_maps_csv)

    for csv_path in (base_pretrained_train_csv, base_pretrained_test_csv, base_pretrained_maps_csv):
        _append_if_exists(output_paths, csv_path)

    # ============================================================
    # ResNet18 scratch
    # ============================================================

    fig, axes = plt.subplots(1, 2, figsize=TRAIN_TEST_FIGSIZE)
    _plot_group(axes[0], axes[1], base_scratch_files, "ResNet18 Scratch")
    _save_figure(fig, base_scratch_out)
    output_paths.append(base_scratch_out)

    fig, ax = plt.subplots(1, 1, figsize=SINGLE_FIGSIZE)
    _plot_imagenet_test_group(ax, base_scratch_files, "ResNet18 Scratch")
    _save_figure(fig, base_scratch_imagenet_out)
    output_paths.append(base_scratch_imagenet_out)

    fig, ax = plt.subplots(1, 1, figsize=SINGLE_FIGSIZE)
    _plot_maps_test_group(ax, base_scratch_files, "ResNet18 Scratch")
    _save_figure(fig, base_scratch_maps_out)
    output_paths.append(base_scratch_maps_out)

    if save_separate_legend(base_scratch_files, base_scratch_legend_out):
        output_paths.append(base_scratch_legend_out)

    _save_train_accuracy_data(base_scratch_files, base_scratch_train_csv)
    _save_imagenet_test_accuracy_data(base_scratch_files, base_scratch_test_csv)
    _save_maps_test_accuracy_data(base_scratch_files, base_scratch_maps_csv)

    for csv_path in (base_scratch_train_csv, base_scratch_test_csv, base_scratch_maps_csv):
        _append_if_exists(output_paths, csv_path)

    # ============================================================
    # ResNet50 pretrained
    # ============================================================

    fig, axes = plt.subplots(1, 2, figsize=TRAIN_TEST_FIGSIZE)
    _plot_group(axes[0], axes[1], resnet50_pretrained_files, "ResNet50 Pretrained")
    _save_figure(fig, resnet50_pretrained_out)
    output_paths.append(resnet50_pretrained_out)

    fig, ax = plt.subplots(1, 1, figsize=SINGLE_FIGSIZE)
    _plot_imagenet_test_group(ax, resnet50_pretrained_files, "ResNet50 Pretrained")
    _save_figure(fig, resnet50_pretrained_imagenet_out)
    output_paths.append(resnet50_pretrained_imagenet_out)

    fig, ax = plt.subplots(1, 1, figsize=SINGLE_FIGSIZE)
    _plot_maps_test_group(ax, resnet50_pretrained_files, "ResNet50 Pretrained")
    _save_figure(fig, resnet50_pretrained_maps_out)
    output_paths.append(resnet50_pretrained_maps_out)

    if save_separate_legend(resnet50_pretrained_files, resnet50_pretrained_legend_out):
        output_paths.append(resnet50_pretrained_legend_out)

    _save_train_accuracy_data(resnet50_pretrained_files, resnet50_pretrained_train_csv)
    _save_imagenet_test_accuracy_data(resnet50_pretrained_files, resnet50_pretrained_test_csv)
    _save_maps_test_accuracy_data(resnet50_pretrained_files, resnet50_pretrained_maps_csv)

    for csv_path in (resnet50_pretrained_train_csv, resnet50_pretrained_test_csv, resnet50_pretrained_maps_csv):
        _append_if_exists(output_paths, csv_path)

    # ============================================================
    # ResNet50 scratch
    # ============================================================

    fig, axes = plt.subplots(1, 2, figsize=TRAIN_TEST_FIGSIZE)
    _plot_group(axes[0], axes[1], resnet50_scratch_files, "ResNet50 Scratch")
    _save_figure(fig, resnet50_scratch_out)
    output_paths.append(resnet50_scratch_out)

    fig, ax = plt.subplots(1, 1, figsize=SINGLE_FIGSIZE)
    _plot_imagenet_test_group(ax, resnet50_scratch_files, "ResNet50 Scratch")
    _save_figure(fig, resnet50_scratch_imagenet_out)
    output_paths.append(resnet50_scratch_imagenet_out)

    fig, ax = plt.subplots(1, 1, figsize=SINGLE_FIGSIZE)
    _plot_maps_test_group(ax, resnet50_scratch_files, "ResNet50 Scratch")
    _save_figure(fig, resnet50_scratch_maps_out)
    output_paths.append(resnet50_scratch_maps_out)

    if save_separate_legend(resnet50_scratch_files, resnet50_scratch_legend_out):
        output_paths.append(resnet50_scratch_legend_out)

    _save_train_accuracy_data(resnet50_scratch_files, resnet50_scratch_train_csv)
    _save_imagenet_test_accuracy_data(resnet50_scratch_files, resnet50_scratch_test_csv)
    _save_maps_test_accuracy_data(resnet50_scratch_files, resnet50_scratch_maps_csv)

    for csv_path in (resnet50_scratch_train_csv, resnet50_scratch_test_csv, resnet50_scratch_maps_csv):
        _append_if_exists(output_paths, csv_path)

    return tuple(output_paths)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot pretrained and scratch mix metrics.")

    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=_default_metrics_dir(),
        help="Directory containing the mix .out logs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_default_metrics_dir(),
        help="Directory where the PNG and CSV files will be written.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug information about file collection and processing.",
    )

    return parser


def main() -> None:
    args = build_argparser().parse_args()

    output_paths = make_mix_plots(
        args.metrics_dir,
        args.output_dir,
        debug=args.debug,
    )

    for output_path in output_paths:
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
