"""Plot the first two principal components of layer embeddings, colored by a parameter.

Supports checkpoints from `action`, `ssl`, or `supervised` model folders and both
flat and hierarchical dataset layouts used in this repository.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms as T

from pretrained.load_mvimgnet_model import load_mv_model

DEFAULT_DATASET_ROOT = Path(r"C:\Users\silas\PycharmProjects\SimClr_MT\dataset_one_transformation")
DEFAULT_MODEL_ROOT = Path(r"C:\Users\silas\PycharmProjects\SimClr_MT\model_files")
DEFAULT_MODEL_ARCH = "resnet18"
DEFAULT_MODEL_TYPES = ["action", "ssl", "supervised"]
# DEFAULT_LAYERS = ["layer1", "layer2", "layer3", "layer4"]
DEFAULT_LAYERS = [
    "layer1_first_relu", "layer1_last_relu",
    "layer2_first_relu", "layer2_last_relu",
    "layer3_first_relu", "layer3_last_relu",
    "layer4_first_relu", "layer4_last_relu",
]
DEFAULT_CSV_NAME = "parameters.csv"
DEFAULT_IMAGE_SUBDIR = "images"
DEFAULT_SUPERVISED_MODEL = Path(
    r"C:\Users\silas\PycharmProjects\SimClr_MT\model_files\resnet18\supervised\supervised_MAPS_resnet18_seed0_1.pt"
)


def natural_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)", path.stem)
    return (int(match.group(1)) if match else math.inf, path.name)


def infer_parameter_column(
    root: str | Path,
    filename_col: str = "image",
    csv_name: str = "parameters.csv",
    instance_glob: str = "*",
) -> str:
    """Infer the first usable numeric parameter column from the dataset."""
    root = Path(root)

    if (root / csv_name).exists() and (root / "images").exists():
        instance_dirs = [root]
    else:
        instance_dirs = sorted(
            [p for p in root.glob(instance_glob) if p.is_dir() and (p / csv_name).exists()],
            key=natural_key,
        )
        if not instance_dirs:
            instance_dirs = []
            for object_dir in sorted(root.iterdir()):
                if not object_dir.is_dir():
                    continue
                for param_dir in sorted(object_dir.iterdir()):
                    if not param_dir.is_dir():
                        continue
                    instance_dirs.extend(
                        sorted(
                            [p for p in param_dir.glob(instance_glob) if p.is_dir() and (p / csv_name).exists()],
                            key=natural_key,
                        )
                    )

    if not instance_dirs:
        raise FileNotFoundError(f"No CSV files found under {root}")

    df = pd.read_csv(instance_dirs[0] / csv_name)
    for col in df.columns:
        if col == filename_col:
            continue
        numeric = pd.to_numeric(df[col].astype(str).str.strip().str.replace(",", ".", regex=False), errors="coerce")
        if numeric.notna().any():
            return col

    fallback = [col for col in df.columns if col != filename_col]
    if fallback:
        return fallback[0]
    raise ValueError(f"Could not infer a parameter column from {instance_dirs[0] / csv_name}")


def infer_parameter_column_from_dataframe(df: pd.DataFrame, filename_col: str = "image") -> str:
    """Infer the usable numeric parameter column from a single CSV dataframe."""
    for col in df.columns:
        if col == filename_col:
            continue
        numeric = pd.to_numeric(df[col].astype(str).str.strip().str.replace(",", ".", regex=False), errors="coerce")
        if numeric.notna().any():
            return col

    fallback = [col for col in df.columns if col != filename_col]
    if fallback:
        return fallback[0]
    raise ValueError("Could not infer a parameter column from the provided dataframe")


class ImageParameterDataset(Dataset):
    """Aggregate samples and one numeric parameter from CSV files.

    Supported layouts:
    - root/parameters.csv + root/images/
    - root/instance_x/parameters.csv + images/
    - root/object_name/parameter_name/instance_x/parameters.csv + images/
    """

    def __init__(
        self,
        root: str | Path,
        parameter_col: str | None = None,
        filename_col: str = "image",
        csv_name: str = "parameters.csv",
        image_subdir: str = "images",
        instance_glob: str = "*",
        transform=None,
        drop_missing_files: bool = True,
        object_filter: str | None = None,
        parameter_filter: str | None = None,
    ) -> None:
        self.root = Path(root)
        self.parameter_col = parameter_col
        self.transform = transform or T.Compose(
            [
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        self.object_filter = object_filter
        self.parameter_filter = parameter_filter

        instance_dirs = self._discover_instance_dirs(csv_name=csv_name, instance_glob=instance_glob)
        records: list[dict[str, object]] = []

        for instance_dir in instance_dirs:
            csv_path = instance_dir / csv_name
            img_dir = instance_dir / image_subdir
            if not csv_path.exists() or not img_dir.exists():
                continue

            df = pd.read_csv(csv_path)
            if filename_col not in df.columns:
                raise ValueError(f"Missing filename column '{filename_col}' in {csv_path}")

            selected_parameter_col = parameter_col or infer_parameter_column_from_dataframe(df, filename_col=filename_col)
            if selected_parameter_col not in df.columns:
                raise ValueError(f"Missing parameter column '{selected_parameter_col}' in {csv_path}")

            # Convert German decimal commas and coerce to numeric.
            values = pd.to_numeric(
                df[selected_parameter_col].astype(str).str.strip().str.replace(",", ".", regex=False),
                errors="coerce",
            )

            for i, filename in enumerate(df[filename_col].astype(str)):
                value = values.iloc[i]
                if pd.isna(value):
                    continue
                img_path = img_dir / filename
                if drop_missing_files and not img_path.exists():
                    continue
                records.append(
                    {
                        "path": img_path,
                        "filename": img_path.name,
                        "relative_path": str(img_path.relative_to(self.root)),
                        "parameter": float(value),
                        "parameter_col": selected_parameter_col,
                    }
                )

        if not records:
            raise RuntimeError(f"No valid samples found under {self.root}")

        self.df = pd.DataFrame(records)
        self.instance_dirs = instance_dirs
        self.instance_labels = [p.name for p in instance_dirs]
        self.paths = [Path(p) for p in self.df["path"].tolist()]
        self.relative_paths = self.df["relative_path"].tolist()
        self.filenames = self.df["filename"].tolist()
        self.parameters = self.df["parameter"].to_numpy(dtype=np.float32)
        self.parameter_columns = self.df["parameter_col"].tolist()

    def _discover_instance_dirs(self, csv_name: str, instance_glob: str) -> list[Path]:
        if (self.root / csv_name).exists() and (self.root / "images").exists():
            return [self.root]

        flat_dirs = sorted(
            [p for p in self.root.glob(instance_glob) if p.is_dir() and (p / csv_name).exists()],
            key=natural_key,
        )
        if flat_dirs:
            return flat_dirs

        hierarchical_dirs: list[Path] = []
        for object_dir in sorted(self.root.iterdir()):
            if not object_dir.is_dir():
                continue
            if self.object_filter is not None and object_dir.name != self.object_filter:
                continue
            for param_dir in sorted(object_dir.iterdir()):
                if not param_dir.is_dir():
                    continue
                if self.parameter_filter is not None and param_dir.name != self.parameter_filter:
                    continue
                dirs = sorted(
                    [p for p in param_dir.glob(instance_glob) if p.is_dir() and (p / csv_name).exists()],
                    key=natural_key,
                )
                hierarchical_dirs.extend(dirs)

        if not hierarchical_dirs:
            raise FileNotFoundError(f"No instance folders found in {self.root}")
        return hierarchical_dirs

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        image = Image.open(self.paths[idx]).convert("RGB")
        image = self.transform(image)
        param = torch.tensor(self.parameters[idx], dtype=torch.float32)
        return image, param, idx


def resolve_checkpoint(args: argparse.Namespace) -> Path:
    if args.checkpoint_path:
        ckpt = args.checkpoint_path
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        return ckpt

    if args.model_type == "supervised":
        ckpt = args.supervised_model
        if not ckpt.exists():
            raise FileNotFoundError(f"Supervised checkpoint not found: {ckpt}")
        return ckpt

    checkpoint_dir = args.model_root / args.model_arch / args.model_type
    checkpoints = sorted(checkpoint_dir.glob(args.checkpoint_pattern), key=natural_key)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir} with pattern {args.checkpoint_pattern}")

    if args.epoch is None:
        return checkpoints[-1]

    epoch_pattern = re.compile(rf"epoch_{args.epoch}(?:\D|$)")
    for ckpt in checkpoints:
        if epoch_pattern.search(ckpt.stem):
            return ckpt

    raise FileNotFoundError(f"No checkpoint for epoch {args.epoch} in {checkpoint_dir}")


def get_module_by_name(model: torch.nn.Module, layer_name: str) -> torch.nn.Module:
    modules = dict(model.named_modules())
    if layer_name in modules:
        return modules[layer_name]
    if hasattr(model, layer_name):
        return getattr(model, layer_name)
    available = sorted([name for name in modules.keys() if name])
    preview = ", ".join(available[:25])
    raise ValueError(f"Layer '{layer_name}' not found. First modules: {preview}")


def get_relu_module(model: torch.nn.Module, layer_name: str, position: str) -> torch.nn.Module:
    """Extract first or last ReLU from a layer sequence.
    
    Args:
        model: The model
        layer_name: Layer name like 'layer1', 'layer2', etc.
        position: 'first' or 'last'
    
    Returns:
        The ReLU module at the specified position
    """
    layer_module = get_module_by_name(model, layer_name)
    
    # Collect all ReLU modules in sequence order
    relu_modules = []
    for name, module in layer_module.named_modules():
        if isinstance(module, torch.nn.ReLU):
            relu_modules.append((name, module))
    
    if not relu_modules:
        raise ValueError(f"No ReLU modules found in layer '{layer_name}'")
    
    if position == "first":
        return relu_modules[0][1]
    elif position == "last":
        return relu_modules[-1][1]
    else:
        raise ValueError(f"Invalid position '{position}', must be 'first' or 'last'")


def extract_embeddings_for_layer(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    layer_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    model.to(device)

    all_embeddings: list[torch.Tensor] = []
    all_params: list[torch.Tensor] = []
    all_indices: list[torch.Tensor] = []

    capture_output = layer_name in {"output", "model_output"}
    activations: dict[str, torch.Tensor] = {}
    hook = None

    # Check if this is a special ReLU extraction request
    relu_match = re.match(r"(layer\d+)_(first|last)_relu", layer_name)
    
    if not capture_output:
        if relu_match:
            # Extract for specific ReLU position
            layer_base = relu_match.group(1)
            position = relu_match.group(2)
            module = get_relu_module(model, layer_base, position)
        else:
            module = get_module_by_name(model, layer_name)

        def _hook_fn(_module, _inputs, output):
            activations["feat"] = output

        hook = module.register_forward_hook(_hook_fn)

    try:
        with torch.no_grad():
            for images, params, indices in loader:
                images = images.to(device, non_blocking=True)
                output = model(images)

                features = output if capture_output else activations.get("feat")
                if features is None:
                    raise RuntimeError(f"No activations captured for layer '{layer_name}'")

                if isinstance(features, (tuple, list)):
                    features = features[0]
                if features.ndim > 2:
                    features = F.adaptive_avg_pool2d(features, 1).flatten(1)
                features = F.normalize(features, dim=1)

                all_embeddings.append(features.detach().cpu())
                all_params.append(params.detach().cpu())
                all_indices.append(torch.as_tensor(indices, dtype=torch.long))
    finally:
        if hook is not None:
            hook.remove()

    embeddings = torch.cat(all_embeddings, dim=0).numpy()
    parameters = torch.cat(all_params, dim=0).numpy()
    indices = torch.cat(all_indices, dim=0).numpy()
    return embeddings, parameters, indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot PC1/PC2 of layer embeddings colored by one dataset parameter.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--parameter-col",
        type=str,
        default=None,
        help="CSV column used for coloring the plot. If omitted, each CSV infers its own numeric parameter column.",
    )
    parser.add_argument("--filename-col", type=str, default="image")
    parser.add_argument(
        "--model-types",
        nargs="*",
        choices=["action", "ssl", "supervised"],
        default=DEFAULT_MODEL_TYPES,
        help="Model types to analyze. Default: action ssl supervised.",
    )
    parser.add_argument(
        "--layers",
        nargs="*",
        default=DEFAULT_LAYERS,
        help="Layers to analyze. Default: layer1 layer2 layer3 layer4.",
    )
    parser.add_argument("--csv-name", type=str, default=DEFAULT_CSV_NAME)
    parser.add_argument("--image-subdir", type=str, default=DEFAULT_IMAGE_SUBDIR)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--model-arch", type=str, default=DEFAULT_MODEL_ARCH)
    parser.add_argument("--checkpoint-pattern", type=str, default="epoch_*.pt")
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--epoch", type=int, default=None, help="For action/ssl: epoch number (default uses latest checkpoint).")
    parser.add_argument("--supervised-model", type=Path, default=DEFAULT_SUPERVISED_MODEL)
    parser.add_argument("--instance-glob", type=str, default="*")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("pca_layer_outputs"))
    return parser.parse_args()


def _sanitized_name(value: str) -> str:
    return value.replace(".", "_").replace("/", "_").replace("\\", "_")


def build_instance_color_map(instance_labels: list[str] | tuple[str, ...] | np.ndarray) -> dict[str, tuple[float, float, float, float]]:
    """Assign stable colors to instance labels."""
    cmap = plt.get_cmap("tab10")
    return {label: cmap(i % cmap.N) for i, label in enumerate(instance_labels)}


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_root = args.output_dir / args.dataset_root.name
    output_root.mkdir(parents=True, exist_ok=True)

    if (args.dataset_root / args.csv_name).exists() and (args.dataset_root / args.image_subdir).exists():
        object_parameter_pairs = [(args.dataset_root.name, args.dataset_root.name)]
    else:
        object_parameter_pairs = []
        for object_dir in sorted(args.dataset_root.iterdir()):
            if not object_dir.is_dir():
                continue
            for parameter_dir in sorted(object_dir.iterdir()):
                if parameter_dir.is_dir():
                    object_parameter_pairs.append((object_dir.name, parameter_dir.name))

    if not object_parameter_pairs:
        print(f"[Warning] No object/parameter combinations found under {args.dataset_root}")
        return

    print(f"[Info] Found {len(object_parameter_pairs)} object/parameter combinations")

    transform_temp = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    first_object, first_parameter = object_parameter_pairs[0]
    try:
        dataset_temp = ImageParameterDataset(
            root=args.dataset_root,
            parameter_col=args.parameter_col,
            filename_col=args.filename_col,
            instance_glob=args.instance_glob,
            transform=transform_temp,
            object_filter=None if first_object == args.dataset_root.name and first_parameter == args.dataset_root.name else first_object,
            parameter_filter=None if first_object == args.dataset_root.name and first_parameter == args.dataset_root.name else first_parameter,
        )
        color_map = build_instance_color_map(dataset_temp.instance_labels)
    except Exception as exc:
        print(f"[Warning] Could not build instance color map from first dataset: {exc}")
        color_map = {}

    for object_name, parameter_name in object_parameter_pairs:
        print(f"\n{'#'*80}")
        print(f"## Object: {object_name} / Parameter: {parameter_name}")
        print(f"{'#'*80}")

        is_root_dataset = object_name == args.dataset_root.name and parameter_name == args.dataset_root.name and (args.dataset_root / args.csv_name).exists() and (args.dataset_root / args.image_subdir).exists()

        dataset = ImageParameterDataset(
            root=args.dataset_root,
            parameter_col=args.parameter_col,
            filename_col=args.filename_col,
            instance_glob=args.instance_glob,
            transform=transform_temp,
            object_filter=None if is_root_dataset else object_name,
            parameter_filter=None if is_root_dataset else parameter_name,
        )

        if not color_map:
            color_map = build_instance_color_map(dataset.instance_labels)

        if args.max_samples and args.max_samples > 0 and args.max_samples < len(dataset):
            rng = np.random.default_rng(args.seed)
            subset_indices = np.sort(rng.choice(np.arange(len(dataset)), size=args.max_samples, replace=False))
            subset = Subset(dataset, subset_indices.tolist())
        else:
            subset_indices = np.arange(len(dataset))
            subset = dataset

        loader = DataLoader(
            subset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False,
            pin_memory=torch.cuda.is_available(),
        )

        parameter_label = dataset.parameter_columns[0] if getattr(dataset, "parameter_columns", None) else (args.parameter_col or parameter_name)
        safe_param = _sanitized_name(parameter_label)

        for model_type in args.model_types:
            checkpoint_path = resolve_checkpoint(argparse.Namespace(**{**vars(args), "model_type": model_type}))
            model, _ = load_mv_model(str(checkpoint_path), device=str(device))

            for layer_name in args.layers:
                print(f"[Info] Running {model_type} / {layer_name}")
                embeddings, parameters, local_indices = extract_embeddings_for_layer(model, loader, device, layer_name)
                sample_indices = subset_indices[local_indices]

                pca = PCA(n_components=2, random_state=args.seed)
                coords = pca.fit_transform(embeddings)

                ckpt_tag = checkpoint_path.stem.replace(" ", "_")
                safe_layer = _sanitized_name(layer_name)
                run_output_dir = output_root / object_name / parameter_name / model_type / safe_layer
                run_output_dir.mkdir(parents=True, exist_ok=True)

                plot_path = run_output_dir / f"pca_{model_type}_{safe_layer}_{safe_param}_{ckpt_tag}.png"
                csv_path = run_output_dir / f"pca_{model_type}_{safe_layer}_{safe_param}_{ckpt_tag}.csv"

                fig, ax = plt.subplots(figsize=(9, 7.5))
                sc = ax.scatter(coords[:, 0], coords[:, 1], c=parameters, cmap="viridis", s=9, alpha=0.85, edgecolors="none")
                cbar = fig.colorbar(sc, ax=ax)
                cbar.set_label(parameter_label)

                var1, var2 = pca.explained_variance_ratio_
                ax.set_xlabel(f"PC1 ({var1 * 100:.2f}% var)")
                ax.set_ylabel(f"PC2 ({var2 * 100:.2f}% var)")
                ax.set_title(f"{object_name} | {parameter_name} | {model_type} | layer={layer_name} | {checkpoint_path.name}")
                ax.grid(True, alpha=0.2)
                fig.tight_layout()
                fig.savefig(plot_path, dpi=220)
                plt.close(fig)

                out_df = pd.DataFrame(
                    {
                        "sample_idx": sample_indices.astype(int),
                        "filename": [dataset.filenames[i] for i in sample_indices],
                        "relative_path": [dataset.relative_paths[i] for i in sample_indices],
                        parameter_label: parameters.astype(float),
                        "pc1": coords[:, 0].astype(float),
                        "pc2": coords[:, 1].astype(float),
                    }
                )
                out_df.to_csv(csv_path, index=False)

                print(f"[Info] Device: {device}")
                print(f"[Info] Checkpoint: {checkpoint_path}")
                print(f"[Info] Samples: {len(sample_indices)}, Embedding dim: {embeddings.shape[1]}")
                print(f"[Saved] {plot_path}")
                print(f"[Saved] {csv_path}")


if __name__ == "__main__":
    main()

