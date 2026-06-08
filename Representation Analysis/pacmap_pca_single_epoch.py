r"""PCA plots for a single checkpoint per model.

This is the non-evolution variant of `checkpoint_pacmap_evolution.py`.
It keeps the same dataset handling and plot outputs, but loads only one
checkpoint per model group instead of a sequence of epochs.

By default it compares:
  - `action`
  - `ssl`
  - `supervised`

For each object / parameter combination in the hierarchical dataset, it writes:
  - a one-panel PCA grid
  - an overlay plot
  - a trajectory-style plot (degenerates to a single-epoch scatter)
  - a CSV with the 2D coordinates

Example (Windows / PowerShell):
  python checkpoint_pacmap_single_epoch.py `
      --dataset-root "C:\Users\silas\PycharmProjects\SimClr_MT\dataset_one_transformation" `
      --model-root "C:\Users\silas\PycharmProjects\SimClr_MT\model_files" `
      --model-arch resnet18 `
      --model-groups action ssl `
      --epoch 49 `
      --output-dir "C:\Users\silas\PycharmProjects\SimClr_MT\pacmap_single_epoch"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
from sklearn.decomposition import PCA
from torch.utils.data import Subset
from torchvision import transforms as T

from checkpoint_pacmap_evolution import (
    DEFAULT_CSV_NAME,
    DEFAULT_DATASET_ROOT,
    DEFAULT_IMAGE_SUBDIR,
    DEFAULT_INSTANCE_GLOB,
    DEFAULT_MODEL_ARCH,
    DEFAULT_MODEL_ROOT,
    SUPERVISED_MODEL_PATH,
    ImageOnlyDataset,
    build_instance_color_map,
    build_loader,
    collect_object_names,
    collect_parameter_names_for_object,
    extract_embeddings,
    plot_epoch_grid,
    plot_overlay,
    plot_trajectories,
    resolve_checkpoints,
    resolve_epoch_label,
    save_coordinates_csv,
    select_balanced_subset_indices,
    select_subset_indices,
)
from pretrained.load_mvimgnet_model import load_mv_model

DEFAULT_MODEL_GROUPS = ["action", "ssl"]
DEFAULT_EPOCH = 49

# ============================================================
# GLOBAL PLOT STYLE SETTINGS
# ============================================================

FIGURE_DPI = 300
# ============================================================
# GLOBAL PLOT STYLE SETTINGS
# ============================================================

PLOT_DPI_GRID = 200
PLOT_DPI_SINGLE = 220

AXIS_LABEL_FONTSIZE = 35
TICK_LABEL_FONTSIZE = 20
TITLE_FONTSIZE = 20
PANEL_TITLE_FONTSIZE = 20
LEGEND_FONTSIZE = 8

GRID_SCATTER_SIZE = 9
OVERLAY_SCATTER_SIZE = 10
TRAJECTORY_SCATTER_SIZE = 11
TRAJECTORY_POINT_SIZE = 16
TRAJECTORY_LINEWIDTH = 0.9

GRID_ALPHA = 0.85
OVERLAY_ALPHA = 0.20
TRAJECTORY_ALPHA = 0.10
TRAJECTORY_LINE_ALPHA = 0.16
TRAJECTORY_POINT_ALPHA = 0.48

GRID_SINGLE_FIGSIZE = (8.8, 7.6)
GRID_PANEL_FIGSIZE = (5.3, 4.8)
SINGLE_PLOT_FIGSIZE = (10, 8)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize last-layer representations for a single checkpoint per model group."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--filename-col", type=str, default="image")
    parser.add_argument("--instance-glob", type=str, default=DEFAULT_INSTANCE_GLOB)
    parser.add_argument("--image-subdir", type=str, default=DEFAULT_IMAGE_SUBDIR)
    parser.add_argument("--csv-name", type=str, default=DEFAULT_CSV_NAME)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--model-arch", type=str, default=DEFAULT_MODEL_ARCH)
    parser.add_argument("--model-groups", type=str, nargs="*", default=DEFAULT_MODEL_GROUPS)
    parser.add_argument(
        "--checkpoint-pattern",
        type=str,
        default="epoch_*.pt",
        help="Glob pattern used to discover checkpoints inside each model group directory.",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=DEFAULT_EPOCH,
        help="Which checkpoint epoch to visualize for each model group.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("pacmap_single_epoch_outputs"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-samples-per-instance", type=int, default=0)
    parser.add_argument(
        "--trajectory-samples",
        type=int,
        default=150,
        help="How many samples to include in the trajectory-style plot.",
    )
    parser.add_argument(
        "--supervised-model",
        type=Path,
        default=SUPERVISED_MODEL_PATH,
        help="Path to a supervised checkpoint to include in the comparison.",
    )
    return parser.parse_args()


def _make_transform() -> T.Compose:
    return T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def _choose_subset(dataset: ImageOnlyDataset, args: argparse.Namespace) -> np.ndarray:
    if args.max_samples_per_instance > 0:
        return select_balanced_subset_indices(dataset.instance_indices, args.max_samples_per_instance, args.seed)
    return select_subset_indices(len(dataset), args.max_samples, args.seed)


def _fit_single_checkpoint_pca(embeddings: np.ndarray, seed: int) -> tuple[np.ndarray, float, float]:
    if np.var(embeddings, axis=0).sum() <= 1e-12:
        coords = np.zeros((embeddings.shape[0], 2), dtype=np.float32)
        return coords, 0.0, 0.0

    pca = PCA(n_components=2, random_state=seed)
    coords = pca.fit_transform(embeddings)
    var_ratio = pca.explained_variance_ratio_
    pc1 = float(var_ratio[0]) if len(var_ratio) > 0 else 0.0
    pc2 = float(var_ratio[1]) if len(var_ratio) > 1 else 0.0
    return coords, pc1, pc2


def _process_single_checkpoint(
    *,
    loader,
    subset_indices: np.ndarray,
    sample_instance_names: list[str],
    sample_instance_indices: np.ndarray,
    relative_paths: list[str],
    filenames: list[str],
    instance_labels: list[str],
    color_map: dict[str, tuple[float, float, float, float]],
    checkpoint_path: Path,
    group_name: str,
    output_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    epoch_label = resolve_epoch_label(checkpoint_path)
    print(f"[Load] {group_name}: {checkpoint_path.name}")

    model, _ = load_mv_model(str(checkpoint_path), device=str(device))
    embeddings, _ = extract_embeddings(model, loader, device)

    if embeddings.shape[0] != len(subset_indices):
        raise RuntimeError(
            f"Embedding count mismatch for {checkpoint_path.name}: got {embeddings.shape[0]}, expected {len(subset_indices)}"
        )

    coords, pc1_var, pc2_var = _fit_single_checkpoint_pca(embeddings, args.seed)
    coords_by_epoch = {epoch_label: coords}
    # x_label = "Dim-1"
    # y_label = "Dim-2"
    x_label = f"PCA-1 ({pc1_var * 100:.2f}% var)"
    y_label = f"PCA-2 ({pc2_var * 100:.2f}% var)"

    group_output_dir = output_dir / f"epoch_{args.epoch}"
    group_output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"pacmap_single_{args.model_arch}_{group_name}_epoch_{args.epoch}"
    grid_path = group_output_dir / f"{prefix}_grid.png"
    overlay_path = group_output_dir / f"{prefix}_overlay.png"
    traj_path = group_output_dir / f"{prefix}_trajectories.png"
    csv_path = group_output_dir / f"{prefix}_coordinates.csv"

    plot_epoch_grid(
        coords_by_epoch,
        [epoch_label],
        sample_instance_indices,
        instance_labels,
        color_map,
        grid_path,
        "",
        show_legend=False,
        axis_fontsize=AXIS_LABEL_FONTSIZE,
        tick_fontsize=TICK_LABEL_FONTSIZE,
        x_label=x_label,
        y_label=y_label,
        show_title=False,
        show_panel_titles=False,
    )

    plot_overlay(
        coords_by_epoch,
        [epoch_label],
        sample_instance_indices,
        instance_labels,
        color_map,
        overlay_path,
        "",
        show_legend=False,
        axis_fontsize=AXIS_LABEL_FONTSIZE,
        tick_fontsize=TICK_LABEL_FONTSIZE,
        x_label=x_label,
        y_label=y_label,
        show_title=False,
    )
    traj_n = min(args.trajectory_samples, len(subset_indices))
    traj_indices = np.sort(np.random.default_rng(args.seed).choice(len(subset_indices), size=traj_n, replace=False))
    plot_trajectories(
        coords_by_epoch,
        [epoch_label],
        sample_instance_indices,
        instance_labels,
        color_map,
        traj_path,
        "",
        traj_indices,
        show_legend=False,
        axis_fontsize=AXIS_LABEL_FONTSIZE,
        tick_fontsize=TICK_LABEL_FONTSIZE,
        x_label=x_label,
        y_label=y_label,
        show_title=False,
    )
    save_coordinates_csv(
        coords_by_epoch,
        [epoch_label],
        subset_indices,
        sample_instance_names,
        sample_instance_indices,
        relative_paths,
        filenames,
        csv_path,
    )

    print(f"[Saved] {group_name}:\n  {grid_path}\n  {overlay_path}\n  {traj_path}\n  {csv_path}")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Info] Using device: {device}")
    print(f"[Info] Dataset root: {args.dataset_root}")
    print(f"[Info] Model root: {args.model_root / args.model_arch}")
    print(f"[Info] Model groups: {args.model_groups}")
    print(f"[Info] Target epoch: {args.epoch}")

    object_names = collect_object_names(args.dataset_root)
    if not object_names:
        print(f"[Warning] No objects found in dataset root: {args.dataset_root}")
        return

    supervised_exists = bool(args.supervised_model) and args.supervised_model.exists()
    print(f"[Info] Supervised model available: {supervised_exists}")

    for object_name in object_names:
        print(f"\n{'#' * 80}")
        print(f"## Object: {object_name}")
        print(f"{'#' * 80}")

        parameter_names = collect_parameter_names_for_object(args.dataset_root, object_name)
        if not parameter_names:
            print(f"[Warning] No parameters found for object: {object_name}")
            continue

        for parameter_name in parameter_names:
            print(f"\n[Info] Processing {object_name} / {parameter_name}")
            dataset = ImageOnlyDataset(
                args.dataset_root,
                filename_col=args.filename_col,
                instance_glob=args.instance_glob,
                image_subdir=args.image_subdir,
                csv_name=args.csv_name,
                transform=_make_transform(),
                object_filter=object_name,
                parameter_filter=parameter_name,
            )

            subset_indices = _choose_subset(dataset, args)
            if len(subset_indices) == 0:
                print(f"[Warning] No samples selected for {object_name} / {parameter_name}")
                continue

            sample_instance_indices = dataset.instance_indices[subset_indices]
            sample_instance_names = [dataset.instance_names[i] for i in subset_indices]
            filenames = [dataset.filenames[i] for i in subset_indices]
            relative_paths = [dataset.relative_paths[i] for i in subset_indices]
            instance_labels = dataset.instance_labels
            color_map = build_instance_color_map(instance_labels)

            subset = Subset(dataset, subset_indices.tolist())
            loader = build_loader(subset, batch_size=args.batch_size, num_workers=args.num_workers)

            for group_name in args.model_groups:
                checkpoint_dir = args.model_root / args.model_arch / group_name
                checkpoint_paths = resolve_checkpoints(None, checkpoint_dir, args.checkpoint_pattern)
                matching = [p for p in checkpoint_paths if resolve_epoch_label(p) == str(args.epoch)]
                if not matching:
                    print(
                        f"[Skip] No checkpoint for epoch {args.epoch} in {checkpoint_dir} (pattern: {args.checkpoint_pattern})"
                    )
                    continue

                group_output_dir = args.output_dir / object_name / parameter_name / args.model_arch / group_name
                _process_single_checkpoint(
                    loader=loader,
                    subset_indices=subset_indices,
                    sample_instance_names=sample_instance_names,
                    sample_instance_indices=sample_instance_indices,
                    relative_paths=relative_paths,
                    filenames=filenames,
                    instance_labels=instance_labels,
                    color_map=color_map,
                    checkpoint_path=matching[0],
                    group_name=group_name,
                    output_dir=group_output_dir,
                    args=args,
                    device=device,
                )

            if supervised_exists:
                group_output_dir = args.output_dir / object_name / parameter_name / "supervised"
                _process_single_checkpoint(
                    loader=loader,
                    subset_indices=subset_indices,
                    sample_instance_names=sample_instance_names,
                    sample_instance_indices=sample_instance_indices,
                    relative_paths=relative_paths,
                    filenames=filenames,
                    instance_labels=instance_labels,
                    color_map=color_map,
                    checkpoint_path=args.supervised_model,
                    group_name="supervised",
                    output_dir=group_output_dir,
                    args=args,
                    device=device,
                )
            else:
                print(f"[Skip] Supervised checkpoint not found: {args.supervised_model}")


if __name__ == "__main__":
    main()
