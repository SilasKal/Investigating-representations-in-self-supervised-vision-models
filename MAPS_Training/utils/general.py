#!/usr/bin/python
# _____________________________________________________________________________

# ----------------
# import libraries
# ----------------

# standard libraries
# -----
import os
import random
from copy import deepcopy

import numpy as np
import torch

import lightning as L
from lightning.fabric.strategies import DDPStrategy
from tqdm import tqdm


@torch.no_grad()
def get_representations(args, net, data_loader, t, get_pair=False   ):
    """
    Get all representations of the dataset given the network and the data loader
    params:
        args: arguments
        net: the network to be used (torch.nn.Module)
        data_loader: data loader of the dataset (DataLoader)
    return:
        tuple of data with the first one being image representations. Other data depends on the dataset.
    """
    net.eval()
    gathered_data = [[], [], [], [], [], [], [], [], [], []]
    strt_idx = 0 if not get_pair else 1

    for data in tqdm(data_loader):
        rep = net(t(data[0][0]))
        gathered_data[0].append(rep)
        if get_pair:
            gathered_data[1].append(net(t(data[0][1])))
        for i in range(1, len(data)):
            gathered_data[i+strt_idx].append(data[i])

        if "supervised" in args.modules:
            gathered_data[len(data)+strt_idx].append(net.sup_projector(rep))

        if args.name == "test3":
            break
    tensor_data = [torch.cat(data, dim=0) for data in gathered_data if data]
    return tensor_data


def prepare_device(args):
    accelerator = getattr(args, "device", "cpu")
    devices = int(getattr(args, "num_devices", 1))
    print(f'torch.cuda.is_available(): {torch.cuda.is_available()}')
    # Fallback if CUDA requested but not available
    if accelerator == "cuda" and not torch.cuda.is_available():
        accelerator = "cpu"
        devices = 1

    # IMPORTANT: only use DDP when running multi-device
    if devices > 1:
        strategy = DDPStrategy(broadcast_buffers=False)
    else:
        strategy = "auto"   # no distributed init

    fabric = L.Fabric(
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        num_nodes=1
    )

    # Only launch distributed when multi-device
    if devices > 1:
        fabric.launch()
    print(fabric)
    print(accelerator)

    return fabric



def run_forward(args, x, net):
    x1, x2 = x.split(x.shape[0] // 2)
    rep1 = net(x1)
    rep2 = net(x2)
    return torch.cat((rep1, rep2), dim=0)



def init_target_net(net, net_target):
    # initialize target network
    for param_online, param_target in zip(net.parameters(), net_target.parameters()):
        param_target.data.copy_(param_online.data)  # initialize
        param_target.requires_grad = False  # not update by gradient


def get_dataset_kwargs(d):
    d_new = deepcopy(d)
    del d_new["class"]
    return d_new


def is_target_needed(args):
    if args.main_loss in ['BYOL'] or "byol" in args.modules:
        return True
    return False


def update_target_net(net, net_target, tau):
    for param, target_param in zip(net.parameters(), net_target.parameters()):
        target_param.data.copy_((1 - tau) * param.data + tau * target_param.data)


# def save_model(fabric, net, log_dir, epoch, optimizer=None, scheduler=None):
#     path = os.path.join(log_dir, 'models')
#     if fabric.global_rank == 0:
#         if not os.path.exists(path):
#             os.mkdir(path)
#     obj = {}
#     obj["model"] = net
#     if optimizer is not None:
#         obj["optimizer"] = optimizer.state_dict()
#     if scheduler is not None:
#         obj["scheduler"] = scheduler.state_dict()
#     fabric.save(os.path.join(path, f'epoch_{epoch}.pt'), obj)

def save_model(fabric, net, log_dir, epoch, optimizer=None, scheduler=None):
    path = os.path.join(log_dir, "models")

    # Only one rank creates dir + writes files
    if fabric.global_rank == 0:
        os.makedirs(path, exist_ok=True)

    # Make sure everyone sees the directory before saving
    fabric.barrier()

    # Only rank 0 writes the checkpoint (prevents corruption)
    if fabric.global_rank != 0:
        return

    # Remove Fabric/DDP wrappers to get the raw model
    # Compatible with older Lightning versions that lack fabric.unwrap()
    model = net
    if hasattr(fabric, 'unwrap'):
        model = fabric.unwrap(net)
    elif hasattr(net, 'module'):
        model = net.module  # unwrap DDP wrapper
    obj = {
        "epoch": int(epoch),
        "model": model.state_dict(),
    }
    if optimizer is not None:
        obj["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        obj["scheduler"] = scheduler.state_dict()

    ckpt_path = os.path.join(path, f"epoch_{epoch}.pt")
    tmp_path = ckpt_path + ".tmp"

    # Atomic write: write tmp then rename
    with open(tmp_path, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, ckpt_path)


import os
import torch

def load_model(fabric, net, args, optimizer=None, scheduler=None, strict=True):
    path = args.path_load_model
    if os.path.isdir(path):
        path = os.path.join(path, f"epoch_{args.epoch_load_model}.pt")
    else:
        if not path.endswith(".pt"):
            path = path + ".pt"

    # ---- load checkpoint (Fabric or plain torch) ----
    if fabric is None:
        checkpoint = torch.load(path, map_location="cpu")
        global_rank = 0
    else:
        checkpoint = fabric.load(path)
        global_rank = fabric.global_rank

    state = checkpoint["model"]

    # 1) drop entire heads that are dataset-dependent
    DROP_PREFIXES = (
        "action_head.",        # action input dim changed (8 -> 6)
        "sup_lin_projector",   # supervised head class count changed
    )
    state = {k: v for k, v in state.items() if not k.startswith(DROP_PREFIXES)}

    # 2) drop any remaining shape-mismatched keys
    model_sd = net.state_dict()
    filtered = {}
    for k, v in state.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            filtered[k] = v

    missing, unexpected = net.load_state_dict(filtered, strict=False)

    if global_rank == 0:
        print(f"[load] path={path}")
        print(f"[load] loaded={len(filtered)}")
        print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")

    # optional: restore optimizer/scheduler if compatible
    if optimizer is not None and "optimizer" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
        except Exception as e:
            if global_rank == 0:
                print(f"[load] optimizer state not loaded: {e}")

    if getattr(args, "cosine_decay", False) and scheduler is not None and "scheduler" in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint["scheduler"])
        except Exception as e:
            if global_rank == 0:
                print(f"[load] scheduler state not loaded: {e}")



#@torch.no_grad()
def normalize(x):
    return (x - x.mean(0, keepdim=True)) / (x.var(dim=0, keepdim=True) + 1e-5).sqrt()

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
        return False
    else:
        raise Exception('Boolean value expected.')

def str2table(v):
    return v.split(',')

