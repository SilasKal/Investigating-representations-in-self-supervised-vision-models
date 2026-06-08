
"""
Similar to plot_adjacent_cosines.py but plots mean adjacent cosines per parameter
across all objects (one plot per transformation parameter).
Produces separate train and test plots for each parameter.

Usage (PowerShell):
python ./compute_adjacent_cosines_by_parameter.py --data_dir "C:/Users/silas/PycharmProjects/SimClr_MT/dataset_one_transformation" --out_prefix "results/adj_param"
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import numpy as np
import torch
import matplotlib.pyplot as plt

# --- Default paths you can edit directly in this file ---
DEFAULT_DATA_DIR = Path(r"C:/Users/silas/PycharmProjects/SimClr_MT/dataset_one_transformation")
DEFAULT_OUT_PREFIX = Path(r"C:/Users/silas/PycharmProjects/SimClr_MT/results/adj_param/adj_cosine")
# Example model checkpoint paths (optional; kept for convenience)
DEFAULT_MODEL_SSL = Path(r"C:\Users\silas\PycharmProjects\SimClr_MT\model_files\resnet18\ssl\epoch_49.pt")
DEFAULT_MODEL_ACTION = Path(r"C:\Users\silas\PycharmProjects\SimClr_MT\model_files\resnet18\action\epoch_49.pt")
DEFAULT_MODEL_SUPERVISED = Path(r"C:\Users\silas\PycharmProjects\SimClr_MT\model_files\supervised_MAPS_resnet18_seed0_1.pt")
# --------------------------------------------------------
DEFAULT_LAYER = "layer4_last_relu"
DEFAULT_NORMALIZE_BY_MAGNITUDE = False


def adjacent_metric(arr: np.ndarray, params: np.ndarray, normalize_by_magnitude: bool = False) -> float:
    """Compute mean adjacent cosine similarity, or cosine distance normalized by |Δa|.

    If normalize_by_magnitude is False:
        mean(cos(z_t, z_{t+1}))
    If normalize_by_magnitude is True:
        mean((1 - cos(z_t, z_{t+1})) / (|Δa| + eps))
    """
    arr = np.asarray(arr)
    params = np.asarray(params).reshape(-1)

    if arr.ndim != 2:
        raise ValueError(f'Expected 2D array (seq_len, dim). Got shape {arr.shape}')
    if arr.shape[0] < 2:
        return float('nan')

    norms = np.linalg.norm(arr, axis=1)
    eps = 1e-8
    norms = np.maximum(norms, eps)
    arr_n = arr / norms[:, None]
    cos_vals = np.sum(arr_n[:-1] * arr_n[1:], axis=1)

    if not normalize_by_magnitude:
        return float(np.mean(cos_vals))

    if params.shape[0] != arr.shape[0]:
        raise ValueError(f'Parameters and embeddings must have same length: {params.shape[0]} vs {arr.shape[0]}')

    delta_mag = np.abs(np.diff(params))
    delta_mag = np.maximum(delta_mag, eps)
    normalized = (1.0 - cos_vals) / delta_mag
    return float(np.mean(normalized))

def compute_by_parameter(data_dir: Path, out_prefix: str, model_ssl: Path, model_action: Path, model_supervised: Path, layer_name: str = 'layer4_last_relu', batch_size: int = 128, num_workers: int = 0, normalize_by_magnitude: bool = False):
    """Compute and plot mean adjacent cosines per parameter across all objects.
    """
    print(f"\n[compute_by_parameter] Starting preprocessing and dataset loading...")

    try:
        from plot_layer_pca_by_parameter import ImageParameterDataset, extract_embeddings_for_layer
        from pretrained.load_mvimgnet_model import load_mv_model
        print(f"[compute_by_parameter] Successfully imported repo utilities")
    except Exception as e:
        raise RuntimeError('Required repo utilities not available: ensure plot_layer_pca_by_parameter and pretrained.load_mvimgnet_model are importable')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[compute_by_parameter] Device: {device}")

    # Load initial model to get preprocess
    any_model_path = model_ssl if Path(model_ssl).exists() else model_action if Path(model_action).exists() else model_supervised
    print(f"[compute_by_parameter] Loading initial model from: {any_model_path}")
    model_any, preprocess_any = load_mv_model(str(any_model_path), device=str(device))
    print(f"[compute_by_parameter] Loaded initial model; building dataset...")
    dataset = ImageParameterDataset(root=str(data_dir), transform=preprocess_any)
    print(f"[compute_by_parameter] Dataset loaded: {len(dataset)} total samples")

    # Build mapping: parameter -> object -> instance -> indices
    param_obj_inst_indices: dict[str, dict[str, dict[str, list[int]]]] = {}
    for idx, rel in enumerate(dataset.relative_paths):
        parts = Path(rel).parts
        object_name = parts[0] if len(parts) > 1 else dataset.root.name
        parameter_name = parts[1] if len(parts) > 2 else 'default'
        instance_label = None
        for lab in dataset.instance_labels:
            if lab in rel:
                instance_label = lab
                break
        if instance_label is None:
            m = re.search(r'instance[_-]?(\d)', rel, flags=re.IGNORECASE)
            instance_label = f'instance_{m.group(1)}' if m else 'unknown'

        param_obj_inst_indices.setdefault(parameter_name, {}).setdefault(object_name, {}).setdefault(instance_label, []).append(idx)

    print(f"[compute_by_parameter] Built per-parameter-object mapping for {len(param_obj_inst_indices)} parameters")
    for param, obj_dict in sorted(param_obj_inst_indices.items()):
        print(f"  - {param}: {len(obj_dict)} objects")

    param_names = sorted(param_obj_inst_indices.keys())
    object_names = sorted(set(obj for param_dict in param_obj_inst_indices.values() for obj in param_dict.keys()))
    model_types = [('action', model_action), ('ssl', model_ssl), ('supervised', model_supervised)]

    train_instances = [f'instance_{i}' for i in [1,2,3,4]]
    test_instance = 'instance_5'

    print(f"\n[compute_by_parameter] Starting embedding computation for {len(param_names)} parameters across {len(object_names)} objects\n")

    def plot_block_dict(data_dict, labels, title, out_file, ylabel: str):
        """Plot with dynamic ylim based on actual values."""
        color_map = {
            'action': 'green',
            'ssl': 'orange',
            'supervised': 'blue'
        }
        plt.figure(figsize=(12, 6))
        x = np.arange(len(labels))

        # Collect all values to compute dynamic ylim
        all_vals = []
        for m, vals in data_dict.items():
            all_vals.extend([v for v in vals if not np.isnan(v)])

        if all_vals:
            val_min = np.min(all_vals)
            val_max = np.max(all_vals)
            val_range = val_max - val_min if val_max > val_min else 0.001
            margin = val_range * 0.15
            ylim_min = val_min - margin
            ylim_max = val_max + margin
        else:
            ylim_min, ylim_max = 0.99, 1.0

        for m, vals in data_dict.items():
            vals_arr = np.array(vals, dtype=np.float64)
            color = color_map.get(m, 'black')
            plt.plot(x, vals_arr, marker='o', label=m, color=color, linewidth=2, markersize=8)
        if len(labels) > 0:
            plt.xticks(x, labels, rotation=90, fontsize=9)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.ylim(ylim_min, ylim_max)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out_file = Path(out_file)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_file, dpi=150)
        print(f"  [compute_by_parameter] Saved plot to {out_file}")
        plt.close()

    # Per parameter, compute embeddings across all objects
    for parameter_name in param_names:
        print(f"[compute_by_parameter] Processing parameter: {parameter_name}")
        train_data = {m: [] for m, _ in model_types}
        test_data = {m: [] for m, _ in model_types}

        for object_name in object_names:
            obj_param_pair = (object_name, parameter_name)
            indices_dict = param_obj_inst_indices.get(parameter_name, {}).get(object_name, {})

            if not indices_dict:
                # No data for this object/parameter combo
                for mtype, _ in model_types:
                    train_data[mtype].append(float('nan'))
                    test_data[mtype].append(float('nan'))
                continue

            for mtype, mpath in model_types:
                print(f"  [compute_by_parameter] Processing {object_name} / {mtype}")
                if not Path(mpath).exists():
                    print(f'    [Warning] checkpoint for {mtype} not found at {mpath}; results will be NaN')
                    train_data[mtype].append(float('nan'))
                    test_data[mtype].append(float('nan'))
                    continue

                model, preprocess = load_mv_model(str(mpath), device=str(device))
                ds = ImageParameterDataset(root=str(data_dir), transform=preprocess)

                # Train: average cosine over instances 1-4
                train_vals = []
                for inst_label in train_instances:
                    indices = indices_dict.get(inst_label, [])
                    if not indices:
                        continue
                    subset = __import__('torch').utils.data.Subset(ds, indices)
                    loader = __import__('torch').utils.data.DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
                    emb, params, _ = extract_embeddings_for_layer(model, loader, device, layer_name)
                    try:
                        v = adjacent_metric(emb, params, normalize_by_magnitude=normalize_by_magnitude)
                        train_vals.append(v)
                    except Exception as e:
                        print(f'    [Error] computing adjacent cosine for {object_name} {inst_label} {mtype}: {e}')
                        train_vals.append(float('nan'))

                train_avg = float(np.nanmean(np.array(train_vals, dtype=np.float64))) if train_vals else float('nan')
                train_data[mtype].append(train_avg)

                # Test: instance 5
                indices = indices_dict.get(test_instance, [])
                if indices:
                    subset = __import__('torch').utils.data.Subset(ds, indices)
                    loader = __import__('torch').utils.data.DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
                    emb, params, _ = extract_embeddings_for_layer(model, loader, device, layer_name)
                    try:
                        v = adjacent_metric(emb, params, normalize_by_magnitude=normalize_by_magnitude)
                        test_data[mtype].append(v)
                    except Exception as e:
                        print(f'    [Error] computing adjacent cosine for {object_name} test {mtype}: {e}')
                        test_data[mtype].append(float('nan'))
                else:
                    test_data[mtype].append(float('nan'))

        print(f"[compute_by_parameter] Finished processing {parameter_name}\n")

        # Generate plots for this parameter
        safe_param = parameter_name.replace(' ', '_').replace('/', '_')
        print(f"[compute_by_parameter] Plotting results for {parameter_name}...")

        if normalize_by_magnitude:
            ylabel = 'mean (1 - cos(z_t, z_{t+1})) / |Δa|'
        else:
            ylabel = 'mean cos(z_t, z_{t+1})'

        plot_block_dict(train_data, object_names, f'{parameter_name} - Train (instances 1-4)', f'{out_prefix}_train_{safe_param}.png', ylabel=ylabel)
        plot_block_dict(test_data, object_names, f'{parameter_name} - Test (instance 5)', f'{out_prefix}_test_{safe_param}.png', ylabel=ylabel)

    print(f"[compute_by_parameter] Completed successfully!\n")


def main():
    parser = argparse.ArgumentParser(description='Compute mean adjacent cosine per parameter across all objects')
    parser.add_argument('--data_dir', type=str, default=str(DEFAULT_DATA_DIR),
                        help=f"Dataset directory (default: {DEFAULT_DATA_DIR})")
    parser.add_argument('--out_prefix', type=str, default=str(DEFAULT_OUT_PREFIX),
                        help=f"Output filename prefix (default: {DEFAULT_OUT_PREFIX})")
    parser.add_argument('--model_ssl', type=str, default=str(DEFAULT_MODEL_SSL),
                        help=f"SSL model checkpoint path (default: {DEFAULT_MODEL_SSL})")
    parser.add_argument('--model_action', type=str, default=str(DEFAULT_MODEL_ACTION),
                        help=f"Action model checkpoint path (default: {DEFAULT_MODEL_ACTION})")
    parser.add_argument('--model_supervised', type=str, default=str(DEFAULT_MODEL_SUPERVISED),
                        help=f"Supervised model checkpoint path (default: {DEFAULT_MODEL_SUPERVISED})")
    parser.add_argument('--layer', type=str, default=DEFAULT_LAYER,
                        help=f"Layer name to extract embeddings from (default: {DEFAULT_LAYER})")
    parser.add_argument('--normalize-by-magnitude', action='store_true', default=DEFAULT_NORMALIZE_BY_MAGNITUDE,
                        help=f"If set, plot mean (1-cos)/|Δa| instead of raw cosine similarity (default: {DEFAULT_NORMALIZE_BY_MAGNITUDE})")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise SystemExit(f"Data dir does not exist: {data_dir}")

    print(f"Using data_dir: {data_dir}")
    print(f"Using out_prefix: {args.out_prefix}")
    print(f"Model SSL: {args.model_ssl}")
    print(f"Model Action: {args.model_action}")
    print(f"Model Supervised: {args.model_supervised}")
    print(f"Using layer: {args.layer}")
    print(f"Normalize by transformation magnitude: {args.normalize_by_magnitude}")

    csv_found = any(p.name == 'parameters.csv' for p in data_dir.rglob('parameters.csv'))
    if csv_found:
        print('[Info] Detected parameters.csv files; using ImageParameterDataset + model extract to compute embeddings')
        try:
            compute_by_parameter(data_dir, args.out_prefix, Path(args.model_ssl), Path(args.model_action), Path(args.model_supervised), layer_name=args.layer, normalize_by_magnitude=args.normalize_by_magnitude)
        except Exception as e:
            print(f"[Error] compute_by_parameter failed: {e}")
            raise
    else:
        raise SystemExit('No parameters.csv files found. This script requires an image dataset with CSV metadata.')


if __name__ == '__main__':
    main()

