import argparse
import os
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from collections import defaultdict

from utils.datasets import MAPSInstanceSplitDataset

from torch.utils.data._utils.collate import default_collate

from torchvision.models import ResNet50_Weights
from torchvision.models import ResNet18_Weights
# ----------------------------
# Collate: strip tv_tensors
# ----------------------------
def _to_plain_tensor(x):
    if isinstance(x, torch.Tensor) and type(x) is not torch.Tensor:
        return x.as_subclass(torch.Tensor)
    return x

def _map_tree(obj):
    if isinstance(obj, (list, tuple)):
        return type(obj)(_map_tree(o) for o in obj)
    if isinstance(obj, dict):
        return {k: _map_tree(v) for k, v in obj.items()}
    return _to_plain_tensor(obj)

def collate_strip_tvtensors(batch):
    batch = [_map_tree(b) for b in batch]
    return default_collate(batch)


# ----------------------------
# dataset helpers
# ----------------------------
def unpack_batch(batch):
    # supports ((img0,img1,action), label) or (img0,label)
    x = batch[0]
    y = batch[1]
    if isinstance(x, (list, tuple)) and len(x) == 3:
        img0, img1, action = x
        return img0, img1, action, y
    return x, None, None, y

def get_all_labels_fast(dataset):
    for attr in ["targets", "labels", "y"]:
        if hasattr(dataset, attr):
            return [int(v) for v in getattr(dataset, attr)]
    return [int(dataset[i][1]) for i in range(len(dataset))]

def stratified_split(dataset, train_ratio=0.8, seed=0, labels=None):
    rng = np.random.RandomState(seed)
    if labels is None:
        labels = get_all_labels_fast(dataset)

    label_to_indices = defaultdict(list)
    for idx, y in enumerate(labels):
        label_to_indices[int(y)].append(idx)

    train_idx, test_idx = [], []
    for y, idxs in label_to_indices.items():
        idxs = np.array(idxs)
        rng.shuffle(idxs)
        n_train = int(len(idxs) * train_ratio)
        train_idx.extend(idxs[:n_train].tolist())
        test_idx.extend(idxs[n_train:].tolist())

    return Subset(dataset, train_idx), Subset(dataset, test_idx)


# ----------------------------
# model: standard supervised resnet
# ----------------------------
def build_supervised_resnet(model_name: str, num_classes: int):
    import torchvision
    if model_name == "resnet50":
        # print('[build_supervised_resnet] Using pretrained weights for ResNet-50')
        # m = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        print('[build_supervised_resnet] Using not pretrained weights for ResNet-50')
        m = torchvision.models.resnet50(weights=None)
    elif model_name == "resnet18":
        # print('[build_supervised_resnet] Using pretrained weights for ResNet-18')
        # m = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        print('[build_supervised_resnet] Using not pretrained weights for ResNet-18')
        m = torchvision.models.resnet18(weights=None)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    in_dim = m.fc.in_features
    m.fc = nn.Linear(in_dim, num_classes)
    return m


# ----------------------------
# training
# ----------------------------
def supervised_train_1gpu(
    args,
    train_set,
    test_set,
    n_classes,
):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"[split] train={len(train_set)} test={len(test_set)} classes={n_classes}")

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_strip_tvtensors,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_strip_tvtensors,
    )

    # model
    model = build_supervised_resnet(args.model, n_classes).to(device)

    # opt/sched
    opt = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.wd,
        nesterov=True,
    )
    if args.cosine:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    else:
        sched = None

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    best_acc = 0.0
    for epoch in range(args.epochs):
        # ---- train ----
        model.train()
        tot_loss, tot_acc, n = 0.0, 0.0, 0

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

        # ---- eval ----
        model.eval()
        correct, m = 0.0, 0
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"test  {epoch}"):
                img0, _, _, y = unpack_batch(batch)
                x = img0.to(device, non_blocking=True)
                y = y.long().to(device, non_blocking=True)

                logits = model(x)
                correct += float((logits.argmax(1) == y).sum().item())
                m += x.size(0)

        train_loss = tot_loss / max(n, 1)
        train_acc = tot_acc / max(n, 1)
        test_acc = correct / max(m, 1)

        print(f"[sup] epoch={epoch} lr={opt.param_groups[0]['lr']:.3e} "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} test_acc={test_acc:.4f}")

        if args.save_path and args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            root, ext = os.path.splitext(args.save_path)
            periodic_path = f"{root}_epoch_{epoch + 1:04d}{ext or '.pt'}"
            ckpt_periodic = {
                "epoch": epoch,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "best_acc": best_acc,
                "args": vars(args),
            }
            torch.save(ckpt_periodic, periodic_path)
            print(f"[sup] saved periodic checkpoint to {periodic_path}")

        if test_acc > best_acc:
            best_acc = test_acc
            if args.save_path:
                ckpt = {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "best_acc": best_acc,
                    "args": vars(args),
                }
                torch.save(ckpt, args.save_path)
                print(f"[sup] saved best to {args.save_path} (acc={best_acc:.4f})")

    print(f"[sup] best_test_acc={best_acc:.4f}")
    return model


# ----------------------------
# main
# ----------------------------
if __name__ == "__main__":
    from torchvision import transforms

    p = argparse.ArgumentParser()

    # data
    p.add_argument("--maps_root", type=str, required=True,
                   help="Root directory of the MAPS dataset")

    # model/train
    p.add_argument("--model", type=str, default="resnet50", choices=["resnet18", "resnet50"])
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=8)

    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--cosine", action="store_true")
    p.add_argument("--amp", action="store_true")

    # MAPS dataset options
    p.add_argument("--maps_img_subdir", type=str, default="images")
    p.add_argument("--maps_logdist_scale", type=float, default=1.5)
    p.add_argument("--maps_pair_mode", type=str, default="next",
                   choices=["next", "random_next"])
    p.add_argument("--maps_test_instance_index", type=int, default=-1)

    # saving
    p.add_argument("--save_path", type=str, default="sup_maps_best.pt")
    p.add_argument("--save_every", type=int, default=10,
                   help="Save a periodic checkpoint every N epochs (<=0 disables periodic saving)")

    args = p.parse_args()

    # Train transform
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Test transform (no random augmentation)
    test_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Build a minimal args namespace compatible with SimpleDataset.__init__
    # SimpleDataset expects: args.data_root, args.unijit, args.seed, args.log_dir, etc.
    dataset_args = argparse.Namespace(
        data_root=args.maps_root,
        unijit=False,
        seed=0,
        log_dir="save",
        maps_img_subdir=args.maps_img_subdir,
        maps_logdist_scale=args.maps_logdist_scale,
        maps_pair_mode=args.maps_pair_mode,
        maps_test_instance_index=args.maps_test_instance_index,
    )

    # Create train and test datasets using the instance-based split
    train_dataset = MAPSInstanceSplitDataset(
        args=dataset_args,
        run_name="supervised_maps",
        split="train",
        transform=train_tf,
        contrastive=False,
        logger=False,
        fabric=None,
    )

    test_dataset = MAPSInstanceSplitDataset(
        args=dataset_args,
        run_name="supervised_maps",
        split="test",
        transform=test_tf,
        contrastive=False,
        logger=False,
        fabric=None,
    )

    n_classes = train_dataset.n_classes

    supervised_train_1gpu(
        args=args,
        train_set=train_dataset,
        test_set=test_dataset,
        n_classes=n_classes,
    )
