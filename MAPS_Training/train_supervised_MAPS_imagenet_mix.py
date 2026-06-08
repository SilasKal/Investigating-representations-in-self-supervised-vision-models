import argparse
import math
import os
import re
import tarfile
from io import BytesIO
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from tqdm import tqdm
from PIL import Image

from torchvision import datasets, transforms
from torchvision.models import ResNet18_Weights, ResNet50_Weights

from utils.datasets import ImageNetPerClassTarDataset, MAPSInstanceSplitDataset


# ----------------------------
# Collate: strip tv_tensors
# ----------------------------
def _to_plain_tensor(x):
    if isinstance(x, torch.Tensor):
        # Materialize into fresh contiguous base-tensor storage so default_collate
        # never tries to resize a non-resizable backing storage.
        if type(x) is not torch.Tensor:
            x = x.as_subclass(torch.Tensor)
        return x.detach().contiguous().clone()
    return x


def _map_tree(obj):
    if isinstance(obj, np.ndarray):
        # Avoid non-resizable tensor views created from numpy-backed storage.
        return torch.as_tensor(obj).contiguous().clone()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (list, tuple)):
        return type(obj)(_map_tree(o) for o in obj)
    if isinstance(obj, dict):
        return {k: _map_tree(v) for k, v in obj.items()}
    return _to_plain_tensor(obj)


def _safe_collate(batch):
    elem = batch[0]

    if isinstance(elem, torch.Tensor):
        tensors = [b.detach().contiguous().clone() for b in batch]
        shapes = [tuple(t.shape) for t in tensors]
        if any(s != shapes[0] for s in shapes[1:]):
            if all(t.ndim == 1 for t in tensors):
                max_len = max(t.size(0) for t in tensors)
                padded = []
                for t in tensors:
                    if t.size(0) == max_len:
                        padded.append(t)
                        continue
                    out = torch.zeros((max_len,), dtype=t.dtype)
                    out[: t.size(0)] = t
                    padded.append(out)
                tensors = padded
            else:
                raise RuntimeError(f"Cannot collate tensors with different shapes: {shapes}")
        return torch.stack(tensors, dim=0)

    if isinstance(elem, np.ndarray):
        tensors = [torch.as_tensor(b).contiguous().clone() for b in batch]
        return torch.stack(tensors, dim=0)

    if isinstance(elem, np.generic):
        return torch.tensor([b.item() for b in batch])

    if isinstance(elem, (float, int)):
        return torch.tensor(batch)

    if isinstance(elem, str):
        return batch

    if isinstance(elem, dict):
        return {k: _safe_collate([d[k] for d in batch]) for k in elem}

    if isinstance(elem, tuple):
        return tuple(_safe_collate(items) for items in zip(*batch))

    if isinstance(elem, list):
        return [_safe_collate(items) for items in zip(*batch)]

    return batch


def collate_strip_tvtensors(batch):
    batch = [_map_tree(b) for b in batch]
    return _safe_collate(batch)


# ----------------------------
# Dataset wrappers/helpers
# ----------------------------
class ImageNetValSubsetDataset(Dataset):
    """
    Build a class-filtered ImageNet validation/test set from tar files.

    Expected layout:
      imagenet_val_root/
        n01440764.tar  (contains images for class n01440764)
        n01443537.tar  (contains images for class n01443537)
        ...
    """

    def __init__(self, root: str, wnids: Sequence[str], transform=None, action_dim: int = 16):
        self.root = root
        self.transform = transform
        self.action_dim = int(action_dim)
        self.wnids = list(wnids)
        self.wnid_to_label = {w: i for i, w in enumerate(self.wnids)}
        self.tar_handles = {}  # Cache for open tar files
        self.samples: List[Tuple[str, str, int]] = []  # (tar_path, member_name, label)

        # Build samples list from tar files
        for wnid in self.wnids:
            tar_path = os.path.join(root, f"{wnid}.tar")
            if os.path.exists(tar_path):
                try:
                    with tarfile.open(tar_path, "r") as tar:
                        for member in tar.getmembers():
                            if member.isfile() and self._is_image_file(member.name):
                                label = self.wnid_to_label[wnid]
                                self.samples.append((tar_path, member.name, label))
                except Exception as e:
                    print(f"Warning: Failed to read tar file {tar_path}: {e}")

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No validation images found for requested WNIDs under: {root}. "
                "Check --imagenet_val_root and ensure tar files exist (e.g., n01440764.tar, etc.)."
            )

    @staticmethod
    def _is_image_file(filename: str) -> bool:
        """Check if filename is an image file."""
        img_extensions = {'.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp'}
        return os.path.splitext(filename)[1].lower() in img_extensions

    def _get_tar_handle(self, tar_path: str):
        """Get or create a cached tar file handle."""
        if tar_path not in self.tar_handles:
            self.tar_handles[tar_path] = tarfile.open(tar_path, "r")
        return self.tar_handles[tar_path]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        tar_path, member_name, label = self.samples[idx]

        # Open tar and extract image
        tar = self._get_tar_handle(tar_path)
        member = tar.getmember(member_name)
        f = tar.extractfile(member)
        img = Image.open(BytesIO(f.read())).convert('RGB')

        action = torch.zeros((self.action_dim,), dtype=torch.float32)
        if self.transform is not None:
            img0 = self.transform(img)
            img1 = img0
        else:
            tt = transforms.ToTensor()
            img0 = tt(img)
            img1 = img0

        return (img0, img1, action), int(label)

    def __del__(self):
        """Close all cached tar handles."""
        for tar in self.tar_handles.values():
            try:
                tar.close()
            except:
                pass


def unpack_batch(batch):
    x = batch[0]
    y = batch[1]
    if isinstance(x, (list, tuple)) and len(x) == 3:
        img0, img1, action = x
        return img0, img1, action, y
    return x, None, None, y


def _labels_from_dataset(dataset: Dataset) -> np.ndarray:
    labels = np.empty(len(dataset), dtype=np.int64)
    for i in range(len(dataset)):
        labels[i] = int(dataset[i][1])
    return labels


def _sample_from_pool(rng: np.random.RandomState, pool: Sequence[int], n: int) -> List[int]:
    if n <= 0:
        return []
    if len(pool) == 0:
        raise ValueError("Tried to sample from an empty pool")
    replace = n > len(pool)
    return rng.choice(np.asarray(pool), size=n, replace=replace).tolist()


def _labels_from_imagenet_train(dataset: Dataset) -> np.ndarray:
    # Avoid image decoding when labels are already present in the indexed samples.
    if hasattr(dataset, "samples"):
        return np.asarray([int(s[2]) for s in dataset.samples], dtype=np.int64)
    return _labels_from_dataset(dataset)


def _inst_sort_key(name: str):
    m = re.search(r"(\d+)$", str(name))
    return (int(m.group(1)) if m else 10**9, str(name))


def _select_one_instance_per_object(dataset: MAPSInstanceSplitDataset, instance_num: int) -> Subset:
    if instance_num <= 0:
        raise ValueError(f"instance_num must be >= 1, got {instance_num}")

    by_obj: Dict[str, List[str]] = {}
    for obj, inst, _ in dataset.index:
        by_obj.setdefault(str(obj), [])
        by_obj[str(obj)].append(str(inst))

    selected_inst_by_obj: Dict[str, str] = {}
    for obj, insts in by_obj.items():
        uniq = sorted(set(insts), key=_inst_sort_key)
        chosen = None
        for inst in uniq:
            m = re.search(r"(\d+)$", inst)
            if m and int(m.group(1)) == instance_num:
                chosen = inst
                break
        if chosen is None:
            pos = instance_num - 1
            if 0 <= pos < len(uniq):
                chosen = uniq[pos]
            else:
                raise ValueError(
                    f"Object '{obj}' has {len(uniq)} instances {uniq}, cannot pick instance {instance_num}."
                )
        selected_inst_by_obj[obj] = chosen

    keep_indices = [
        i for i, (obj, inst, _local_i) in enumerate(dataset.index)
        if selected_inst_by_obj.get(str(obj)) == str(inst)
    ]
    if not keep_indices:
        raise RuntimeError(f"No MAPS samples kept for instance {instance_num}.")
    return Subset(dataset, keep_indices)


def build_instance_balanced_mix_indices(
    maps_dataset: Dataset,
    imagenet_labels: np.ndarray,
    maps_ratio: float,
    seed: int,
):
    """
    Build training indices that maintain constant dataset size while varying the MAPS/ImageNet ratio.

    Total training size is fixed at len(imagenet_labels) regardless of maps_ratio.
    This ensures fair comparison: differences in performance are due to MAPS quality, not data quantity.

    Args:
        maps_dataset: MAPS training dataset
        imagenet_labels: Labels from ImageNet training dataset
        maps_ratio: Fraction of total training data from MAPS (0.0 = ImageNet only, 1.0 = MAPS only)
        seed: Random seed for reproducibility

    Returns:
        List of indices into the concatenated [maps_train + imagenet_train] dataset
    """
    if not (0.0 <= maps_ratio <= 1.0):
        raise ValueError(f"maps_train_ratio must be in [0,1], got {maps_ratio}")

    maps_n = len(maps_dataset)
    imagenet_n = len(imagenet_labels)
    rng = np.random.RandomState(seed)

    # Special case: pure MAPS experiment should remain MAPS-only, even if MAPS has
    # fewer samples than ImageNet in this filtered setup.
    if maps_ratio >= 1.0 - 1e-12:
        if maps_n <= 0:
            raise ValueError("MAPS train set is empty, cannot build MAPS-only indices.")
        maps_indices = rng.choice(maps_n, size=maps_n, replace=False)
        return maps_indices.tolist()

    total_n = int(imagenet_n)
    if total_n <= 0:
        raise ValueError("ImageNet train set is empty, cannot build mixed training indices.")

    n_maps_target = int(round(maps_ratio * total_n))
    n_imagenet_target = total_n - n_maps_target

    n_maps_sample = min(n_maps_target, maps_n)
    n_imagenet_sample = min(n_imagenet_target, imagenet_n)

    # Fill shortfall from whichever source still has unique samples available.
    remaining = total_n - (n_maps_sample + n_imagenet_sample)
    if remaining > 0:
        maps_spare = maps_n - n_maps_sample
        add_maps = min(remaining, maps_spare)
        n_maps_sample += add_maps
        remaining -= add_maps
    if remaining > 0:
        imagenet_spare = imagenet_n - n_imagenet_sample
        add_img = min(remaining, imagenet_spare)
        n_imagenet_sample += add_img
        remaining -= add_img
    if remaining > 0:
        raise ValueError(
            f"Not enough unique samples for mix_total_train={total_n}: "
            f"maps_available={maps_n}, imagenet_available={imagenet_n}"
        )

    mixed_indices: List[int] = []

    # Sample from MAPS without replacement
    if n_maps_sample > 0:
        maps_indices = rng.choice(maps_n, size=n_maps_sample, replace=False)
        mixed_indices.extend(maps_indices.tolist())

    # Sample from ImageNet without replacement
    if n_imagenet_sample > 0:
        imagenet_indices = rng.choice(imagenet_n, size=n_imagenet_sample, replace=False)
        # ImageNet indices are offset by maps_n in the concatenated dataset
        mixed_indices.extend([maps_n + i for i in imagenet_indices.tolist()])

    if len(mixed_indices) == 0:
        raise RuntimeError("Mixed train index is empty. Check ratio and dataset contents.")

    rng.shuffle(mixed_indices)
    return mixed_indices


# ----------------------------
# Model
# ----------------------------
def build_supervised_resnet(model_name: str, num_classes: int, pretrained_backbone: bool = True):
    import torchvision

    if model_name == "resnet50":
        print(f"[mix] Building ResNet50 with pretrained_backbone={pretrained_backbone}")
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained_backbone else None
        model = torchvision.models.resnet50(weights=weights)
    elif model_name == "resnet18":
        print(f"[mix] Building ResNet18 with pretrained_backbone={pretrained_backbone}")
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained_backbone else None
        model = torchvision.models.resnet18(weights=weights)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    in_dim = model.fc.in_features
    model.fc = nn.Linear(in_dim, num_classes)
    return model


# ----------------------------
# Train/eval
# ----------------------------
@torch.no_grad()
def evaluate(model, loader, device, desc: str):
    model.eval()
    correct = 0.0
    total = 0
    for batch in tqdm(loader, desc=desc):
        img0, _, _, y = unpack_batch(batch)
        x = img0.to(device, non_blocking=True)
        y = y.long().to(device, non_blocking=True)
        logits = model(x)
        correct += float((logits.argmax(1) == y).sum().item())
        total += x.size(0)
    return correct / max(total, 1)


def train_supervised_mix(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[mix] Using device: {device}")
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset_args = argparse.Namespace(
        data_root=args.maps_root,
        unijit=False,
        seed=args.seed,
        log_dir="save",
        maps_img_subdir=args.maps_img_subdir,
        maps_logdist_scale=args.maps_logdist_scale,
        maps_pair_mode=args.maps_pair_mode,
        maps_test_instance_index=args.maps_test_instance_num - 1,
    )

    maps_train = MAPSInstanceSplitDataset(
        args=dataset_args,
        run_name="supervised_maps_imagenet_mix",
        split="train",
        transform=train_tf,
        contrastive=False,
        logger=False,
        fabric=None,
    )
    maps_test = MAPSInstanceSplitDataset(
        args=dataset_args,
        run_name="supervised_maps_imagenet_mix",
        split="test",
        transform=eval_tf,
        contrastive=False,
        logger=False,
        fabric=None,
    )

    maps_n_classes = maps_train.n_classes

    maps_train = _select_one_instance_per_object(maps_train, args.maps_train_instance_num)
    maps_test = _select_one_instance_per_object(maps_test, args.maps_test_instance_num)

    # Match ImageNet synthetic action size to MAPS action size to keep collation stable.
    maps_action = maps_train[0][0][2]
    maps_action_dim = int(maps_action.numel()) if isinstance(maps_action, torch.Tensor) else 16

    wnids = [w.strip() for w in args.imagenet_wnids.split(",") if w.strip()]
    if len(wnids) == 0:
        raise ValueError("--imagenet_wnids cannot be empty")

    if len(wnids) != maps_n_classes:
        raise ValueError(
            "Number of ImageNet WNIDs must match number of MAPS classes. "
            f"Got {len(wnids)} WNIDs vs {maps_n_classes} MAPS classes."
        )

    imagenet_train = ImageNetPerClassTarDataset(
        root=args.imagenet_train_root,
        wnids=wnids,
        transform=train_tf,
        contrastive=False,
        action_dim=maps_action_dim,
        index_cache_path=args.imagenet_index_cache,
    )

    imagenet_test = ImageNetValSubsetDataset(
        root=args.imagenet_val_root,
        wnids=wnids,
        transform=eval_tf,
        action_dim=maps_action_dim,
    )

    n_classes = maps_n_classes
    if imagenet_train.n_classes != n_classes:
        raise ValueError("Class count mismatch between MAPS and ImageNet train datasets")

    imagenet_labels = _labels_from_imagenet_train(imagenet_train)

    mixed_train = ConcatDataset([maps_train, imagenet_train])
    mixed_indices = build_instance_balanced_mix_indices(
        maps_dataset=maps_train,
        imagenet_labels=imagenet_labels,
        maps_ratio=args.maps_train_ratio,
        seed=args.seed,
    )
    mixed_train_subset = Subset(mixed_train, mixed_indices)

    n_maps_selected = int(np.sum(np.asarray(mixed_indices) < len(maps_train)))
    n_img_selected = len(mixed_indices) - n_maps_selected

    train_loader = DataLoader(
        mixed_train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_strip_tvtensors,
    )
    maps_test_loader = DataLoader(
        maps_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_strip_tvtensors,
    )
    imagenet_test_loader = DataLoader(
        imagenet_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_strip_tvtensors,
    )

    model = build_supervised_resnet(
        model_name=args.model,
        num_classes=n_classes,
        pretrained_backbone=not args.no_pretrained,
    ).to(device)

    opt = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.wd,
        nesterov=True,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs) if args.cosine else None
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    best_score = -math.inf
    best_maps = 0.0
    best_imagenet = 0.0

    print(
        "[mix] "
        f"train_maps={len(maps_train)} train_imagenet={len(imagenet_train)} "
        f"target_total={len(imagenet_train)} "
        f"maps_ratio={args.maps_train_ratio:.3f} imagenet_ratio={1.0 - args.maps_train_ratio:.3f} "
        f"selected_maps={n_maps_selected} selected_imagenet={n_img_selected} "
        f"selected_total={len(mixed_train_subset)} classes={n_classes}"
    )

    for epoch in range(args.epochs):
        model.train()
        tot_loss = 0.0
        tot_acc = 0.0
        n = 0

        for batch in tqdm(train_loader, desc=f"train {epoch}"):
            img0, _, _, y = unpack_batch(batch)
            x = img0.to(device, non_blocking=True)
            y = y.long().to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                logits = model(x)
                loss = F.cross_entropy(logits, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            bs = x.size(0)
            tot_loss += float(loss.item()) * bs
            tot_acc += float((logits.argmax(1) == y).sum().item())
            n += bs

        if sched is not None:
            sched.step()

        train_loss = tot_loss / max(n, 1)
        train_acc = tot_acc / max(n, 1)

        maps_test_acc = evaluate(model, maps_test_loader, device, desc=f"maps_test {epoch}")
        imagenet_test_acc = evaluate(model, imagenet_test_loader, device, desc=f"imgnet_test {epoch}")

        if args.ckpt_metric == "maps":
            score = maps_test_acc
        elif args.ckpt_metric == "imagenet":
            score = imagenet_test_acc
        else:
            score = 0.5 * (maps_test_acc + imagenet_test_acc)

        print(
            f"[mix] epoch={epoch} lr={opt.param_groups[0]['lr']:.3e} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"maps_test_acc={maps_test_acc:.4f} imagenet_test_acc={imagenet_test_acc:.4f} "
            f"score({args.ckpt_metric})={score:.4f}"
        )

        if score > best_score:
            best_score = score
            best_maps = maps_test_acc
            best_imagenet = imagenet_test_acc
            if args.save_path:
                ckpt = {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "best_score": best_score,
                    "best_maps_test_acc": best_maps,
                    "best_imagenet_test_acc": best_imagenet,
                    "args": vars(args),
                    "wnids": wnids,
                }
                torch.save(ckpt, args.save_path)
                print(f"[mix] saved best to {args.save_path}")

    print(
        f"[mix] done best_score={best_score:.4f} "
        f"best_maps_test_acc={best_maps:.4f} best_imagenet_test_acc={best_imagenet:.4f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # MAPS
    parser.add_argument("--maps_root", type=str, required=True)
    parser.add_argument("--maps_img_subdir", type=str, default="images")
    parser.add_argument("--maps_logdist_scale", type=float, default=1.5)
    parser.add_argument("--maps_pair_mode", type=str, default="next", choices=["next", "random_next"])
    parser.add_argument("--maps_train_instance_num", type=int, default=1)
    parser.add_argument("--maps_test_instance_num", type=int, default=5)

    # ImageNet
    parser.add_argument("--imagenet_train_root", type=str, required=True)
    parser.add_argument("--imagenet_val_root", type=str, required=True)
    parser.add_argument("--imagenet_index_cache", type=str, required=True)
    parser.add_argument("--imagenet_wnids", type=str, required=True, help="Comma-separated WNIDs")

    # Mix setup
    parser.add_argument(
        "--maps_train_ratio",
        type=float,
        default=0.5,
        help="Fraction in [0,1] of training samples from MAPS; for ratios < 1, total size is fixed to ImageNet train size, while 1.0 is MAPS-only.",
    )

    # Model/train
    parser.add_argument("--model", type=str, default="resnet50", choices=["resnet18", "resnet50"])
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--cosine", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=0)

    # Checkpointing
    parser.add_argument("--save_path", type=str, default="sup_maps_imagenet_mix_best.pt")
    parser.add_argument("--ckpt_metric", type=str, default="mean", choices=["maps", "imagenet", "mean"])

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_supervised_mix(args)

