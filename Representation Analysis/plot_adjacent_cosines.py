#!/usr/bin/env python3
"""
compute_adjacent_cosines.py

Usage (PowerShell):
python ./compute_adjacent_cosines.py --data_dir "C:/Users/silas/PycharmProjects/SimClr_MT/dataset_one_transformation" --out_prefix "results/adj"
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Tuple, List
from collections import defaultdict

# --- Default paths you can edit directly in this file ---
DEFAULT_DATA_DIR = Path(r"C:/Users/silas/PycharmProjects/SimClr_MT/dataset_one_transformation")
DEFAULT_OUT_PREFIX = Path(r"C:/Users/silas/PycharmProjects/SimClr_MT/results/adj/adj_cosine")
# Example model checkpoint paths (optional; kept for convenience)
DEFAULT_MODEL_SSL = Path(r"C:\Users\silas\PycharmProjects\SimClr_MT\model_files\resnet18\ssl\epoch_49.pt")
DEFAULT_MODEL_ACTION = Path(r"C:\Users\silas\PycharmProjects\SimClr_MT\model_files\resnet18\action\epoch_49.pt")
DEFAULT_MODEL_SUPERVISED = Path(r"C:\Users\silas\PycharmProjects\SimClr_MT\model_files\supervised_MAPS_resnet18_seed0_1.pt")
# --------------------------------------------------------
DEFAULT_LAYER = "layer4"
def find_instance_from_name(name: str) -> Optional[int]:
    patterns = [
        r'instance[_-]?(?:id)?([1-5])',
        r'inst[_-]?([1-5])',
        r'[_-]([1-5])[_\.-]',
        r'[_-]([1-5])$',
        r'\b([1-5])\b'
    ]
    for p in patterns:
        m = re.search(p, name, flags=re.IGNORECASE)
        if m:
            try:
                val = int(m.group(1))
                if 1 <= val <= 5:
                    return val
            except Exception:
                continue
    return None

def try_load_file(path: Path) -> Any:
    ext = path.suffix.lower()
    if ext in ('.pt', '.pth'):
        try:
            return torch.load(path, map_location='cpu')
        except Exception as e:
            print(f"torch.load failed for {path}: {e}")
    if ext == '.npy':
        return np.load(path, allow_pickle=True)
    if ext == '.npz':
        return dict(np.load(path, allow_pickle=True))
    if ext in ('.pkl', '.pickle'):
        with open(path, 'rb') as f:
            return pickle.load(f)
    # fallback: try torch.load
    try:
        return torch.load(path, map_location='cpu')
    except Exception:
        raise RuntimeError(f"Unknown or unsupported file type: {path}")

def tensor_like_to_numpy(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, (list, tuple)):
        return np.asarray(x)
    raise TypeError(f"Unsupported embedding type: {type(x)}")

def extract_embeddings(obj: Any) -> Dict[str, np.ndarray]:
    """
    Try to extract embeddings for model types. Returns dict mapping model_type->2D numpy array
    """
    out: Dict[str, np.ndarray] = {}

    # direct tensor/array -> assume generic 'ssl' embedding
    if isinstance(obj, (torch.Tensor, np.ndarray)):
        arr = tensor_like_to_numpy(obj)
        out['ssl'] = arr
        return out

    # dict-like
    if isinstance(obj, dict):
        keys = list(obj.keys())
        klower = {k.lower(): k for k in keys}
        patterns = {
            'action': ['action', 'action_emb', 'action_embedding', 'z_action', 'action_z'],
            'ssl': ['ssl', 'simclr', 'simclr_z', 'ssl_emb', 'z_ssl', 'z'],
            'supervised': ['supervised', 'sup', 'supervised_emb', 'z_sup', 'z_supervised']
        }
        for model_type, p_list in patterns.items():
            for p in p_list:
                if p in klower:
                    val = obj[klower[p]]
                    try:
                        out[model_type] = tensor_like_to_numpy(val)
                        break
                    except Exception:
                        continue
            if model_type in out:
                continue
            # substring match
            for k in keys:
                low = k.lower()
                for p in p_list:
                    if p in low:
                        try:
                            out[model_type] = tensor_like_to_numpy(obj[k])
                            break
                        except Exception:
                            continue
                if model_type in out:
                    break

        # If nothing recognized, try to collect 2D arrays and give them generated keys
        if not out:
            for k in keys:
                try:
                    arr = tensor_like_to_numpy(obj[k])
                except Exception:
                    continue
                if arr is not None and arr.ndim == 2:
                    out[f'unknown_{k}'] = arr
            return out

    # object with attributes
    for attr in ('embeddings', 'embedding', 'z', 'z_all'):
        if hasattr(obj, attr):
            val = getattr(obj, attr)
            if val is not None:
                return extract_embeddings(val)

    raise RuntimeError('Could not extract embeddings from object (unknown structure)')

def mean_adjacent_cosine(arr: np.ndarray) -> float:
    if arr is None:
        return float('nan')
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f'Expected 2D array (seq_len, dim). Got shape {arr.shape}')
    L = arr.shape[0]
    if L < 2:
        return float('nan')
    norms = np.linalg.norm(arr, axis=1)
    eps = 1e-8
    norms = np.maximum(norms, eps)
    arr_n = arr / norms[:, None]
    c = np.sum(arr_n[:-1] * arr_n[1:], axis=1)
    return float(np.mean(c))

def collect_data(data_dir: Path) -> Tuple[Dict[int, List[Tuple[Path, Dict[str, np.ndarray]]]], List[str]]:
    mapping = defaultdict(list)
    seen_keys = set()
    file_count = 0
    print(f"[collect_data] Walking through {data_dir} recursively...")
    for p in data_dir.rglob('*'):
        if not p.is_file():
            continue
        file_count += 1
        if file_count % 50 == 0:
            print(f"  [collect_data] Processed {file_count} files so far...")
        try:
            inst = find_instance_from_name(str(p) + ' ' + str(p.parent))
            if inst is None:
                inst = find_instance_from_name(p.parent.name)
            if inst is None:
                # skip files without instance token
                continue
            loaded = try_load_file(p)
            emb = extract_embeddings(loaded)
            for k in emb.keys():
                seen_keys.add(k)
            mapping[inst].append((p, emb))
        except Exception as e:
            print(f'Warning: skipping {p} due to error: {e}')
            continue
    print(f"[collect_data] Finished: processed {file_count} files total, loaded {sum(len(v) for v in mapping.values())} embedding objects")
    return mapping, sorted(list(seen_keys))

def plot_results(results_by_instance: Dict[int, List[Tuple[Path, Dict[str, np.ndarray]]]], out_prefix: str = 'adj_cosine'):
    model_types = set()
    for items in results_by_instance.values():
        for _, d in items:
            model_types.update(d.keys())
    model_types = sorted(model_types)
    # Prefer a consistent display order for plots/summary.
    preferred_order = ['supervised', 'ssl', 'action']
    model_types = [m for m in preferred_order if m in model_types] + [m for m in model_types if m not in preferred_order]

    display_labels = {
        'action': 'Action',
        'ssl': 'SSL',
        'supervised': 'Supervised'
    }

    print(f"[plot_results] Found model types: {model_types}")

    train_instances = [1,2,3,4]
    train_labels = []
    train_data = {m: [] for m in model_types}
    print(f"[plot_results] Computing mean adjacent cosines for training instances 1-4...")
    for inst in train_instances:
        items = results_by_instance.get(inst, [])
        if items:
            print(f"  [plot_results] Instance {inst}: processing {len(items)} objects")
        for p, emb in items:
            train_labels.append(f"{inst}:{p.name}")
            for m in model_types:
                if m in emb:
                    try:
                        v = mean_adjacent_cosine(emb[m])
                    except Exception as e:
                        print(f"Error computing cosine for {p} model {m}: {e}")
                        v = float('nan')
                else:
                    v = float('nan')
                train_data[m].append(v)

    test_labels = []
    test_data = {m: [] for m in model_types}
    print(f"[plot_results] Computing mean adjacent cosines for test instance 5...")
    for p, emb in results_by_instance.get(5, []):
        if len(test_labels) == 0:
            print(f"  [plot_results] Instance 5: processing embeddings")
        test_labels.append(p.name)
        for m in model_types:
            if m in emb:
                try:
                    v = mean_adjacent_cosine(emb[m])
                except Exception as e:
                    print(f"Error computing cosine for {p} model {m}: {e}")
                    v = float('nan')
            else:
                v = float('nan')
            test_data[m].append(v)

    def plot_block(data_dict, labels, title, out_file):
        # Color mapping for model types
        color_map = {
            'action': 'green',
            'ssl': 'orange',
            'supervised': 'blue'
        }
        plt.figure(figsize=(10, 6))
        x = np.arange(len(labels))
        
        # Collect all values to compute dynamic ylim
        all_vals = []
        for m, vals in data_dict.items():
            all_vals.extend([v for v in vals if not np.isnan(v)])
        
        if all_vals:
            val_min = np.min(all_vals)
            val_max = np.max(all_vals)
            val_range = val_max - val_min if val_max > val_min else 0.001
            margin = val_range * 0.15  # 15% margin
            ylim_min = val_min - margin
            ylim_max = val_max + margin
        else:
            ylim_min, ylim_max = 0.99, 1.0
        
        for m in model_types:
            vals = data_dict[m]
            vals_arr = np.array(vals, dtype=np.float64)
            color = color_map.get(m, 'black')
            plt.plot(x, vals_arr, marker='o', label=display_labels.get(m, m), color=color, linewidth=2, markersize=8)
        if len(labels) > 0:
            plt.xticks(x, labels, rotation=90, fontsize=8)
        plt.ylabel('mean cos(z_t, z_{t+1})')
        plt.title(title)
        plt.ylim(ylim_min, ylim_max)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out_file = Path(out_file)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_file, dpi=150)
        print(f"Saved plot to {out_file}")
        plt.close()

    if any(len(v)>0 for v in train_data.values()):
        print(f"[plot_results] Generating training plot...")
        plot_block(train_data, train_labels, 'Train (instances 1-4) mean adjacent cosine per object', f'{out_prefix}_train_1-4.png')
    else:
        print('No train files found for instances 1-4')

    if len(test_labels) > 0:
        print(f"[plot_results] Generating test plot...")
        plot_block(test_data, test_labels, 'Test (instance 5) mean adjacent cosine per object', f'{out_prefix}_test_5.png')
    else:
        print('No test files found for instance 5')

    print('\nSummary (mean over objects, ignoring NaNs):')
    for m in model_types:
         tvals = np.array(train_data[m], dtype=np.float64)
         kvals = np.array(test_data[m], dtype=np.float64)
         tmean = np.nanmean(tvals) if tvals.size > 0 else float('nan')
         kmean = np.nanmean(kvals) if kvals.size > 0 else float('nan')
         print(f"{display_labels.get(m, m)}: train_mean={tmean:.4f}   test_mean={kmean:.4f}")


def compute_using_models(data_dir: Path, out_prefix: str, model_ssl: Path, model_action: Path, model_supervised: Path, layer_name: str = 'layer4_last_relu', batch_size: int = 128, num_workers: int = 0):
    """Use ImageParameterDataset and repo model loaders to compute mean adjacent cosines per object.
    Produces same style of train/test plots as plot_results.
    """
    print(f"\n[compute_using_models] Starting preprocessing and dataset loading...")
    # Lazy import check
    if 'ImageParameterDataset' not in globals() or 'extract_embeddings_for_layer' not in globals() or 'load_mv_model' not in globals():
        try:
            from plot_layer_pca_by_parameter import ImageParameterDataset, extract_embeddings_for_layer
            from pretrained.load_mvimgnet_model import load_mv_model
            print(f"[compute_using_models] Successfully imported repo utilities")
        except Exception as e:
            raise RuntimeError('Required repo utilities not available: ensure plot_layer_pca_by_parameter and pretrained.load_mvimgnet_model are importable')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[compute_using_models] Device: {device}")

    # find a model to obtain a preprocess
    any_model_path = model_ssl if Path(model_ssl).exists() else model_action if Path(model_action).exists() else model_supervised
    print(f"[compute_using_models] Loading initial model from: {any_model_path}")
    model_any, preprocess_any = load_mv_model(str(any_model_path), device=str(device))
    print(f"[compute_using_models] Loaded initial model; building dataset...")
    dataset = ImageParameterDataset(root=str(data_dir), transform=preprocess_any)
    print(f"[compute_using_models] Dataset loaded: {len(dataset)} total samples")
    print(f"[compute_using_models] Instance labels: {dataset.instance_labels}")

    # Build mapping from object -> instance_label -> indices
    obj_inst_indices: dict[str, dict[str, list[int]]] = {}
    # Also map (object, parameter) -> instance -> indices for separate plots
    obj_param_inst_indices: dict[tuple[str, str], dict[str, list[int]]] = {}
    for idx, rel in enumerate(dataset.relative_paths):
        parts = Path(rel).parts
        object_name = parts[0] if len(parts) > 1 else dataset.root.name
        # Try to discern parameter from path structure
        # For hierarchical: object/parameter/instance/images
        parameter_name = parts[1] if len(parts) > 2 else 'default'
        instance_label = None
        for lab in dataset.instance_labels:
            if lab in rel:
                instance_label = lab
                break
        if instance_label is None:
            m = re.search(r'instance[_-]?(\d)', rel, flags=re.IGNORECASE)
            instance_label = f'instance_{m.group(1)}' if m else 'unknown'
        obj_inst_indices.setdefault(object_name, {}).setdefault(instance_label, []).append(idx)
        obj_param_inst_indices.setdefault((object_name, parameter_name), {}).setdefault(instance_label, []).append(idx)

    print(f"[compute_using_models] Built index mapping for {len(obj_inst_indices)} objects")
    for obj, inst_dict in sorted(obj_inst_indices.items()):
        print(f"  - {obj}: instances {sorted(inst_dict.keys())} (total samples: {sum(len(v) for v in inst_dict.values())})")

    print(f"[compute_using_models] Built per-object-parameter mapping for {len(obj_param_inst_indices)} combinations:")
    for (obj, param), inst_dict in sorted(obj_param_inst_indices.items()):
        print(f"  - {obj} / {param}: instances {sorted(inst_dict.keys())} (total samples: {sum(len(v) for v in inst_dict.values())})")

    object_names = sorted(obj_inst_indices.keys())
    obj_param_pairs = sorted(obj_param_inst_indices.keys())
    model_types = [('supervised', model_supervised), ('ssl', model_ssl), ('action', model_action)]

    train_instances = [f'instance_{i}' for i in [1,2,3,4]]
    test_instance = 'instance_5'

    print(f"\n[compute_using_models] Starting embedding computation for {len(obj_param_pairs)} object-parameter combinations and {len(model_types)} models\n")

    def plot_block_dict(data_dict, labels, title, out_file):
        # Color mapping for model types
        color_map = {
            'action': 'green',
            'ssl': 'orange',
            'supervised': 'blue'
        }
        display_labels = {
            'action': 'Action',
            'ssl': 'SSL',
            'supervised': 'Supervised'
        }
        plt.figure(figsize=(10, 6))
        x = np.arange(len(labels))
        
        # Collect all values to compute dynamic ylim
        all_vals = []
        for m, vals in data_dict.items():
            all_vals.extend([v for v in vals if not np.isnan(v)])
        
        if all_vals:
            val_min = np.min(all_vals)
            val_max = np.max(all_vals)
            val_range = val_max - val_min if val_max > val_min else 0.001
            margin = val_range * 0.15  # 15% margin
            ylim_min = val_min - margin
            ylim_max = val_max + margin
        else:
            ylim_min, ylim_max = 0.99, 1.0
        
        for m in model_types:
            vals = data_dict[m]
            vals_arr = np.array(vals, dtype=np.float64)
            color = color_map.get(m, 'black')
            plt.plot(x, vals_arr, marker='o', label=display_labels.get(m, m), color=color, linewidth=2, markersize=8)
        if len(labels) > 0:
            plt.xticks(x, labels, rotation=90, fontsize=8)
        plt.ylabel('mean cos(z_t, z_{t+1})')
        plt.title(title)
        plt.ylim(ylim_min, ylim_max)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out_file = Path(out_file)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_file, dpi=150)
        print(f"  [compute_using_models] Saved plot to {out_file}")
        plt.close()

    # Per (object, parameter) pair, compute embeddings
    for object_name in object_names:
        params_for_object = sorted(set(p for (o, p) in obj_param_pairs if o == object_name))
        for parameter_name in params_for_object:
            print(f"[compute_using_models] Processing object: {object_name}, parameter: {parameter_name}")
            train_data = {m: [] for m, _ in model_types}
            test_data = {m: [] for m, _ in model_types}
            cur_pair = (object_name, parameter_name)

            for mtype, mpath in model_types:
                print(f"  [compute_using_models] Processing model: {mtype}")
                if not Path(mpath).exists():
                    print(f'[Warning] checkpoint for {mtype} not found at {mpath}; results will be NaN')
                    train_data[mtype].append(float('nan'))
                    continue
                print(f"    [compute_using_models] Loading checkpoint: {mpath}")
                model, preprocess = load_mv_model(str(mpath), device=str(device))
                ds = ImageParameterDataset(root=str(data_dir), transform=preprocess)

                vals = []
                for inst_label in train_instances:
                    indices = obj_param_inst_indices.get(cur_pair, {}).get(inst_label, [])
                    if not indices:
                        print(f"      [compute_using_models] Skipping {inst_label}: no samples found")
                        continue
                    print(f"      [compute_using_models] Processing train {inst_label}: {len(indices)} samples")
                    subset = __import__('torch').utils.data.Subset(ds, indices)
                    loader = __import__('torch').utils.data.DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
                    print(f"        [compute_using_models] Extracting embeddings from layer '{layer_name}' ({len(loader)} batches)...")
                    emb, _, _ = extract_embeddings_for_layer(model, loader, device, layer_name)
                    print(f"        [compute_using_models] Extracted embeddings shape: {emb.shape}")
                    try:
                        v = mean_adjacent_cosine(emb)
                        print(f"        [compute_using_models] Mean adjacent cosine: {v:.6f}")
                    except Exception as e:
                        print(f'[Error] computing adjacent cosine for {object_name} {inst_label} {mtype}: {e}')
                        v = float('nan')
                    vals.append(v)
                train_data[mtype].append(float(np.nanmean(np.array(vals, dtype=np.float64))) if vals else float('nan'))
                print(f"    [compute_using_models] Train avg for {mtype}: {train_data[mtype][-1]:.6f}")

            for mtype, mpath in model_types:
                if not Path(mpath).exists():
                    test_data[mtype].append(float('nan'))
                    continue
                print(f"    [compute_using_models] Processing test model: {mtype}")
                model, preprocess = load_mv_model(str(mpath), device=str(device))
                ds = ImageParameterDataset(root=str(data_dir), transform=preprocess)
                indices = obj_param_inst_indices.get(cur_pair, {}).get(test_instance, [])
                if not indices:
                    print(f"      [compute_using_models] Skipping {test_instance}: no samples found")
                    test_data[mtype].append(float('nan'))
                    continue
                print(f"      [compute_using_models] Processing test {test_instance}: {len(indices)} samples")
                subset = __import__('torch').utils.data.Subset(ds, indices)
                loader = __import__('torch').utils.data.DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
                print(f"        [compute_using_models] Extracting embeddings from layer '{layer_name}' ({len(loader)} batches)...")
                emb, _, _ = extract_embeddings_for_layer(model, loader, device, layer_name)
                print(f"        [compute_using_models] Extracted embeddings shape: {emb.shape}")
                try:
                    v = mean_adjacent_cosine(emb)
                    print(f"        [compute_using_models] Mean adjacent cosine: {v:.6f}")
                except Exception as e:
                    print(f'[Error] computing adjacent cosine for {object_name} test {mtype}: {e}')
                    v = float('nan')
                test_data[mtype].append(v)
                print(f"    [compute_using_models] Test result for {mtype}: {test_data[mtype][-1]:.6f}")

            print(f"[compute_using_models] Finished processing {object_name} / {parameter_name}\n")

            # Generate plots for this (object, parameter) combination
            safe_object = object_name.replace(' ', '_').replace('/', '_')
            safe_param = parameter_name.replace(' ', '_').replace('/', '_')
            plot_suffix = f"{safe_object}_{safe_param}"
            print(f"[compute_using_models] Plotting results for {object_name} / {parameter_name}...")


            plot_block_dict(train_data, ['train_avg'], f'{object_name} / {parameter_name} - Train (instances 1-4)', f'{out_prefix}_train_{plot_suffix}.png')
            plot_block_dict(test_data, ['test_avg'], f'{object_name} / {parameter_name} - Test (instance 5)', f'{out_prefix}_test_{plot_suffix}.png')

    print(f"[compute_using_models] Completed successfully!\n")

def main():
    parser = argparse.ArgumentParser(description='Compute mean adjacent cosine per object')
    # By default the script will use the constants above so you don't need to pass args
    parser.add_argument('--data_dir', type=str, default=str(DEFAULT_DATA_DIR),
                        help=f"Dataset directory (default: {DEFAULT_DATA_DIR})")
    parser.add_argument('--out_prefix', type=str, default=str(DEFAULT_OUT_PREFIX),
                        help=f"Output filename prefix (default: {DEFAULT_OUT_PREFIX})")
    # Optional model checkpoint paths (kept for convenience / future integration)
    parser.add_argument('--model_ssl', type=str, default=str(DEFAULT_MODEL_SSL),
                        help=f"SSL model checkpoint path (default: {DEFAULT_MODEL_SSL})")
    parser.add_argument('--model_action', type=str, default=str(DEFAULT_MODEL_ACTION),
                        help=f"Action model checkpoint path (default: {DEFAULT_MODEL_ACTION})")
    parser.add_argument('--model_supervised', type=str, default=str(DEFAULT_MODEL_SUPERVISED),
                        help=f"Supervised model checkpoint path (default: {DEFAULT_MODEL_SUPERVISED})")
    parser.add_argument('--layer', type=str, default=DEFAULT_LAYER,
                        help=f"Layer name to extract embeddings from (default: {DEFAULT_LAYER})")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise SystemExit(f"Data dir does not exist: {data_dir}")

    # Print chosen (or default) model/checkpoint paths so they're easy to change inside the file
    print(f"Using data_dir: {data_dir}")
    print(f"Using out_prefix: {args.out_prefix}")
    print(f"Model SSL: {args.model_ssl}")
    print(f"Model Action: {args.model_action}")
    print(f"Model Supervised: {args.model_supervised}")
    print(f"Using layer: {args.layer}")
    # If this dataset appears to be an image-based dataset with CSVs, use repo loading to compute embeddings
    csv_found = any(p.name == 'parameters.csv' for p in data_dir.rglob('parameters.csv'))
    if csv_found:
        print('[Info] Detected parameters.csv files; attempting to use ImageParameterDataset + model extract to compute embeddings')
        try:
            compute_using_models(data_dir, args.out_prefix, Path(args.model_ssl), Path(args.model_action), Path(args.model_supervised), layer_name=args.layer)
            return
        except Exception as e:
            print(f"[Warning] compute_using_models failed: {e}; falling back to file-based loading")

    # Fallback: load embedding files directly from the data directory
    print('\n[main] Using file-based embedding loading fallback...')
    print('[main] Searching for embedding files (.pt/.pth/.npy/.npz/.pkl) under data_dir recursively...')
    mapping, model_keys = collect_data(data_dir)
    if not mapping:
        raise SystemExit('No files loaded. Check naming conventions and supported file types (.pt/.pth/.npy/.npz/.pkl).')

    print('Model keys found across files:', model_keys)
    for inst in sorted(mapping.keys()):
        print(f'Instance {inst}: {len(mapping[inst])} files')

    print('\n[main] Computing plots from loaded embeddings...')
    plot_results(mapping, out_prefix=args.out_prefix)
    print('[main] Completed successfully!\n')

if __name__ == '__main__':
    main()