import csv
import io
import math
import os
import time
import random

import scipy
import h5py
import pandas as pd
import numpy as np
import torch
from torch.linalg import lstsq

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


from utils.general import str2bool, str2table, get_representations
from utils.logger import EpochLogger




class SimpleDataset(Dataset):

    def __init__(self, args, run_name, split='train', transform=None, target_transform=None, contrastive=True, logger=True, fabric=None, eval=False, **kwargs):
        self.args = args
        self.contrastive = contrastive
        self.transform = transform
        self.target_transform = target_transform
        self.split = split
        self.fabric=fabric
        self.run_name = run_name
        self.eval_mode = eval
        if split == "test" and logger and fabric.global_rank == 0:
            self.epoch_logger = EpochLogger(output_dir=os.path.join(args.log_dir, run_name),
                                            exp_name="Seed-" + str(args.seed), output_fname='progress.txt')
            self.epoch_logger.save_config(args)
        if self.args.unijit:
            s = self.args.jitter_strength
            jit = transforms.RandomApply([transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)], p=self.args.jitter)
            grayscale = transforms.RandomGrayscale(p=0.2)
            self.jit = transforms.Compose([jit, grayscale])

        self.time = time.time()

    @classmethod
    def get_args(cls, parser):
        return parser

    @torch.no_grad()
    def eval(self, net, dataloader_train_eval, dataloader_test, epoch=0, modules=[], tv=None, **kwargs):
        if self.fabric.global_rank == 0:
            test_time = time.time()

        data_test = get_representations(self.args, net, dataloader_test, tv)
        data_test = self.fabric.all_gather(data_test)

        if self.fabric.global_rank == 0:
            self.epoch_logger.log_tabular("epoch", epoch)
            self.epoch_logger.log_tabular("all_time", (time.time() - self.time) / self.args.test_every)
            self.epoch_logger.log_tabular("test_time", (time.time() - test_time) / self.args.test_every)

            f_l_test = {}
            if "0" in self.args.eval_labels:
                f_l_test["0"] = {"features": data_test[0], "labels": data_test[1], "supervised": data_test[-1]}

            for m in modules:
                for k, v in m.eval(net, f_l_test).items():
                    self.epoch_logger.log_tabular(k, v)
            self.time = time.time()
            self.epoch_logger.dump_tabular()
        self.fabric.barrier()

    def get_log_dir(self):
        return os.path.join(self.args.log_dir, self.run_name)

class Toys4kDataset(SimpleDataset):
    def __init__(self, *args, view=None, category=None, object=None, **kwargs):
        super().__init__(*args,  **kwargs)
        self.hdf5_file = h5py.File(os.path.join(self.args.data_root, "data.h5"), 'r')
        annotations_file = "dataset.parquet" if self.split == "train" else "dataset_test.parquet"
        self.img_labels = pd.read_parquet(os.path.join(self.args.data_root, annotations_file))


        self.num_views = 180
        self.max_views = 180
        self.view = None
        self.img_labels.columns = self.img_labels.columns.astype(str)

        if self.args.finetune_labels != -1 and self.split == "train":
            self.img_labels = self.img_labels.groupby("2").apply(lambda x: x.sample(self.args.finetune_labels)).reset_index(drop=True)

        if self.eval_mode:
            self.img_labels = self.img_labels.groupby("2").apply(lambda x: x.sample(1)).reset_index(drop=True)


        result = [int(r.split("_")[-3]) for r in self.img_labels["0"]]
        self.img_labels["12"] = result
        if view is not None:
            self.img_labels = self.img_labels.loc[self.img_labels.iloc[:, 12] == view]
            self.img_labels.reset_index(inplace=True)

        if category is not None:
            self.img_labels = self.img_labels.loc[self.img_labels.loc[:, "5"] == category]
            self.img_labels.reset_index(inplace=True)

        elif object is not None:
            self.img_labels = self.img_labels.loc[self.img_labels.loc[:, "2"] == object]
            self.img_labels.reset_index(inplace=True)


        self.replace = "binoc" in self.args.data_root
        self.num_columns = len(self.img_labels.columns)
        self.log_backgrounds = not ("back5_" in self.args.data_root or "back7_" in self.args.data_root)


        i, j, k = 0, 0, 0
        self.obj_to_int = {}
        self.cat_to_int = {}
        self.category_list = []
        self.object_list = []
        for index, _ in self.img_labels.groupby("2").count().iterrows():
            self.obj_to_int[index] = i
            self.object_list.append(index)
            i += 1
        for index, _ in self.img_labels.groupby("5").count().iterrows():
            self.cat_to_int[index] = j
            self.category_list.append(index)
            j += 1
        self.back_to_int = {}
        for index, _ in self.img_labels.groupby("3").count().iterrows():
            self.back_to_int[index] = k
            k += 1
        self.max_backgrounds = len(self.back_to_int)
        self.n_classes = len(self.cat_to_int)
        self.n_objs = len(self.obj_to_int)

    @classmethod
    def get_args(cls, parser):
        return parser

    def __len__(self):
        return len(self.img_labels)

    def get_actions(self, idx, r, idx2):
        if self.args.sampling_mode == "randomwalk":
            return torch.tensor([r, r], dtype=torch.float32)


        true_rotation = r * 360 / self.max_views
        rad_rotation = math.pi * true_rotation / 180
        return torch.tensor([math.sin(rad_rotation), math.cos(rad_rotation)])



    def get_image(self, idx):
        if self.args.hdf5:
            h5_index = self.img_labels.loc[idx, "h5_index"]
            return Image.open(io.BytesIO(self.hdf5_file[self.split][h5_index])),

        img_path = os.path.join(self.args.data_root, self.img_labels.loc[idx, "2"], self.img_labels.loc[idx, "0"])
        if self.replace:
            img_path = img_path.replace(".png", "_vc.png")
        return Image.open(img_path),

    def get_other_image(self, idx):
        splitted = self.img_labels.loc[idx, "0"].split("_")
        original_pos = int(splitted[-3])
        if self.args.sampling_mode == "randomwalk+":
            rotation = 1
        elif self.args.sampling_mode == "randomwalk":
            rotation = (1 if random.random() < 0.5 else -1)
        elif self.args.sampling_mode == "opposite":
            rotation = self.num_views/2
        elif self.args.sampling_mode == "uniform":
            rotation = random.randint(-self.num_views, self.num_views)
        nv = original_pos + rotation
        if nv > self.max_views:
            pos = nv - self.max_views
        elif nv <= 0:
            pos = nv + self.max_views
        else:
            pos = nv

        new_idx = idx + pos - original_pos
        action = self.get_actions(idx, rotation, idx + pos - original_pos)
        return *self.get_image(new_idx), action


    def __getitem__(self, idx):
        image, = self.get_image(idx)
        label = self.cat_to_int[self.img_labels.loc[idx, "5"]]
        label_back = self.back_to_int[self.img_labels.loc[idx, "3"]]
        label_obj = self.obj_to_int[self.img_labels.loc[idx, "2"]]
        # label_view = int(self.img_labels.iloc[idx, 0].split("_")[-3])
        label_view = int(self.img_labels.loc[idx, "12"])
        if self.target_transform:
            label = self.target_transform(label)
        state = torch.get_rng_state()
        if self.args.unijit and self.contrastive:
            image = self.transform(image)
            img_pair, a = self.get_other_image(idx)
            img_pair = self.transform(img_pair)
            state = torch.get_rng_state()
            image_t = self.jit(image)
            if random.random() <= self.args.punijit:
                torch.set_rng_state(state)
            img_pair_t = self.jit(img_pair)


            return (image_t, img_pair_t, a), label,label_back, label_obj

        if self.args.unijit:
            image_t = self.transform(image)
            img_pair_t = self.transform(image)
            state = torch.get_rng_state()
            image_t = self.jit(image_t)
            torch.set_rng_state(state)
            img_pair_t = self.jit(img_pair_t)
            return (image_t, img_pair_t, torch.zeros(2, )), label, label_back, label_obj, label_view

        if self.transform:
            image_t = self.transform(image)

        if self.transform and self.contrastive:
            img_pair, a = self.get_other_image(idx)#if random.random() < self.args.p_time else (image, torch.zeros((2,)))
            img_pair_t = self.transform(img_pair)

        if not self.contrastive:
            return (image_t, image_t, torch.zeros(2,)), label, label_back, label_obj, label_view


        return (image_t, img_pair_t, a), label, label_back, label_obj, label_view


    @torch.no_grad()
    def eval(self, net, dataloader_train_eval, dataloader_test, epoch=0, scheduler=None, modules=[], dataset_train=None, tv=None, **kwargs):
        test_time = time.time()

        #Compute features
        # data = get_representations(self.args, net, dataloader_train_eval)
        # if "2" in self.args.eval_labels:
        #     f_obj_train, f_obj_test = data[0][dataset_train.obj_split_train], data[0][dataset_train.obj_split_test]
        #     labels_obj_train, labels_obj_test = data[3][dataset_train.obj_split_train], data[3][dataset_train.obj_split_test]
        #     f_l_test["2"] = {"features": f_obj_test, "labels": labels_obj_test}

        data = get_representations(self.args, net, dataloader_train_eval, tv)
        features_train_eval, labels_train_eval = data[0], data[1]
        lstsq_model = lstsq(features_train_eval, torch.nn.functional.one_hot(labels_train_eval, self.n_classes).type(torch.float32))


        data_test = get_representations(self.args, net, dataloader_test, tv)
        f_l_test={}
        if "0" in self.args.eval_labels:
            f_l_test["0"] = {"features": data_test[0], "labels": data_test[1], "supervised": data_test[-1]}

            mask_side = data_test[4] == self.max_views
            f_l_test["_side"] = {"features": data_test[0][mask_side], "labels": data_test[1][mask_side]}

            mask_face = data_test[4] == 135
            f_l_test["_face"] = {"features": data_test[0][mask_face], "labels": data_test[1][mask_face]}

            mask_quarter = data_test[4] == 157
            f_l_test["_quarter"] = {"features": data_test[0][mask_quarter], "labels": data_test[1][mask_quarter]}

        pred = data_test[0] @ lstsq_model.solution
        mean_acc = (pred.argmax(-1) == data_test[1]).to(torch.float).mean()

        self.epoch_logger.log_tabular("test_acc1", mean_acc.item())
        self.epoch_logger.log_tabular("sparse", torch.count_nonzero(data_test[0], dim=1).to(torch.float32).mean(dim=0).item())
        self.epoch_logger.log_tabular("epoch", epoch)
        self.epoch_logger.log_tabular("all_time", (time.time()-self.time)/self.args.test_every)
        self.epoch_logger.log_tabular("test_time", (time.time()-test_time)/self.args.test_every)
        for m in modules:
            for k, v in m.eval(net, f_l_test).items():
                self.epoch_logger.log_tabular(k, v)


        self.time = time.time()
        self.epoch_logger.dump_tabular()

class CO3D(SimpleDataset):


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from utils.data_types import CO3D_CATEGORIES
        assert self.args.hdf5, "only hdf5 available"
        self.hdf5_file = h5py.File(os.path.join(self.args.data_root, "data.h5"), 'r')
        self.dataset = pd.read_parquet(os.path.join(self.args.data_root, "dataset.parquet" if self.split!="test" else "dataset_test.parquet"))
        self.t = 0
        self.cat_to_int = {}
        self.category_list = []
        j=0

        if self.args.finetune_labels != -1 and self.split == "train":
            self.dataset = self.dataset.groupby("object").apply(lambda x: x.sample(self.args.finetune_labels)).reset_index(drop=True)

        if self.split == "train":
            dataset2 = pd.read_parquet(os.path.join(self.args.data_root,"dataset_test.parquet"))
            dd2 = dataset2.groupby("object").first().to_dict()
            dd1 = self.dataset.groupby("object").first().to_dict()
            for k, r in dd2["path"].items():
                assert k not in dd1["path"], "Test and train dataset overlap"

        for index, _ in self.dataset.groupby("category").count().iterrows():
            self.cat_to_int[index] = j
            self.category_list.append(index)

            j += 1
        # self.n_classes = len(self.category_list) if self.args.mode != "finetune" else 51
        self.n_classes = 51
        self.action_headers = np.array(["a"+str(1+i) for i in range(14)])
        self.action_quatheaders = np.array(["a"+str(1+i) for i in range(9)])
        self.action_foclength_trans = np.array(["a"+str(1+i) for i in range(9,14)])
        if self.args.co3d_quaternion:
            self.action_mean = torch.tensor([0.0057,-0.0274,0.168, 0.545]+[-1.16e+06, 2.71e+05, 4.83e+05,-0.000415,-0.000257])
            self.action_std = torch.tensor([0.0841, 0.404, 0.499, 0.505]+[ 3.38e+09, 7.88e+08,1.4e+09, 9.26, 4.94])
        else:
            self.action_std = torch.tensor([9.66991054e-01, 7.18770433e-01, 5.58006881e-01, 7.14118006e-01,6.62875426e-01, 4.25014783e-01, 5.67008818e-01, 4.29122972e-01,4.37485732e-01, 3.02687459e+09, 7.04659724e+08, 1.25568361e+09,1.03948843e+01, 5.53417153e+00])
            self.action_mean = torch.tensor([-0.00023,-0.0001,0.00003, 0.00004,0.00010,-0.00037,0.00001, 0.00026, 0.00018, 2326900, -541714, -965291, -0.00240, -0.00142])
        print(len(self.dataset), self.split, self.n_classes)



    def __len__(self):
        return len(self.dataset)

    @classmethod
    def get_args(cls, parser):
        parser.add_argument("--co3d_normalize", type=str2bool,default=True)
        parser.add_argument("--co3d_quaternion", type=str2bool,default=True)
        return parser


    def open_image(self, category, obj, index, get_size=False):
        if get_size:
            npar = self.hdf5_file.get(category).get(obj)
            return Image.open(io.BytesIO(npar[index])), len(npar)
        return Image.open(io.BytesIO(self.hdf5_file.get(category).get(obj)[index]))


    def get_action(self, idx, new_idx):
        if self.args.co3d_quaternion:
            q1 = torch.tensor(self.dataset.loc[idx, self.action_quatheaders].values.astype(np.float32)).view(3,3)
            q2 = torch.tensor(self.dataset.loc[new_idx, self.action_quatheaders].values.astype(np.float32)).view(3,3)
            qtrans = torch.matmul(torch.transpose(q1, 1, 0), q2)
            r = scipy.spatial.transform.Rotation.from_matrix(qtrans)
            qtrans = torch.tensor(r.as_quat(), dtype=torch.float32)

            diff_translength = torch.tensor(self.dataset.loc[new_idx, self.action_foclength_trans].values.astype(np.float32) - self.dataset.loc[idx, self.action_foclength_trans].values.astype(np.float32))
            action = torch.cat((qtrans, diff_translength), dim=0)
            return action

        a0 = torch.tensor(self.dataset.loc[idx, self.action_headers].values.astype(np.float32),dtype=torch.float32)
        a1 = torch.tensor(self.dataset.loc[new_idx, self.action_headers].values.astype(np.float32),dtype=torch.float32)
        action = a0 - a1
        return action

    def __getitem__(self, idx):
        category, obj, frame_index = self.dataset.loc[idx, "category"],self.dataset.loc[idx, "object"],self.dataset.loc[idx, "index"]

        image, size = self.open_image(category, obj, frame_index, get_size=True)
        label = self.cat_to_int[category]
        if self.transform:
            image_t = self.transform(image)

        if not self.contrastive:
            return (image_t, self.transform(image) if self.split == "train" else image_t, torch.zeros((14,))), label

        if self.args.sampling_mode == "uniform":
            new_frame_index = random.randint(0, size-1)
        elif self.args.sampling_mode == "randomwalk+":
            new_frame_index = frame_index+1 if frame_index < size-1 else frame_index - 1
        else:
            new_frame_index = max(0, min(size-1, (frame_index-1 if random.random() < 0.5 else frame_index+1)))

        new_idx = idx + new_frame_index - frame_index
        assert str(obj) == str(self.dataset.loc[new_idx, "object"])
        image_pair = self.open_image(category, obj, self.dataset.loc[new_idx, "index"])

        action = self.get_action(idx, new_idx)
        if self.args.co3d_normalize:
            action = (action - self.action_mean)/self.action_std

        if self.transform:
            image_pair = self.transform(image_pair)
        return (image_t, image_pair, action), label


class MVImgNet(SimpleDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.category_mapping = {}
        category_list = []
        for index, v in pd.read_csv("utils/mvimgnet_category.txt", header=None).iterrows():
            self.category_mapping[v[0]] = index
            category_list.append(v[0])
        self.n_classes = len(self.category_mapping)

        if self.args.hdf5 and self.args.hdf5_mode == "partition":
            if not os.path.exists(os.path.join(self.args.data_root, "data_all.h5")):
                raise Exception("You should build the merged data hdf5 file")
            self.hdf5_file = h5py.File(os.path.join(self.args.data_root, "data_all.h5"), "r")
            self.dataset = pd.read_parquet(os.path.join(self.args.data_root, f"dataset_{self.split}_all3.parquet"))
        elif self.args.hdf5:
            self.hdf5_file = h5py.File(os.path.join(self.args.data_root, "data2.h5"), "r")
            self.dataset = pd.read_parquet(os.path.join(self.args.data_root, "dataset_" + self.split + "2.parquet"))
        else:
            self.dataset = pd.read_parquet(os.path.join(self.args.data_root, "dataset_" + self.split + "2.parquet"))



        if self.args.imgnet_subset and self.split == "train":
            uniqs = self.dataset["object"].unique()
            pathobj = os.path.join(self.args.data_root, f"subset_{self.args.seed}_{self.args.imgnet_subset}")
            if self.fabric.global_rank == 0:
                if not os.path.exists(pathobj):
                    sampled_objects = np.random.choice(uniqs, int(len(uniqs)*self.args.imgnet_subset))
                    f = open(pathobj, "w")
                    csv.writer(f).writerow(sampled_objects)
                    f.close()

            self.fabric.barrier()
            f = open(pathobj, "r")
            sampled_objects = next(iter(csv.reader(f)))
            f.close()
            self.dataset = self.dataset.query('object in @sampled_objects').reset_index(drop=True)

        if self.args.finetune_labels != -1 and self.split == "train":
            self.dataset = self.dataset.groupby("object").apply(lambda x: x.sample(self.args.finetune_labels)).reset_index(drop=True)

        self.action_headers = np.array(["q0","q1","q2","q3","t0","t1","t2"])
        self.action_mean = torch.tensor([0.759, -0.000354, -0.00682, -0.00723 , 0.00314, 0.00787, -0.0171, 0])
        self.action_std = torch.tensor([0.431, 0.048, 0.358, 0.328, 3.8, 1.25, 1.13, 1])

        print(len(self.dataset))

    def __len__(self):
        return len(self.dataset)

    def open_image(self, category, obj, index, path, idx):
        if self.args.hdf5_mode == "partition":
            partition = self.dataset.loc[idx, "partition"]
            p = self.hdf5_file.get(partition)
            c = p.get(category)
            o = c.get(obj)
            return Image.open(io.BytesIO(o[index]))


        c = self.hdf5_file.get(category)
        o = c.get(obj)
        return Image.open(io.BytesIO(o[index]))


    @classmethod
    def get_args(cls, parser):
        parser.add_argument("--hdf5_mode", type=str, default="normal")
        parser.add_argument("--action_clamp", type=str2bool, default=True)
        parser.add_argument("--imgnet_subset", type=float, default=0)
        return parser

    def quaternion_multiply(self, quaternion1, quaternion0):
        w0, x0, y0, z0 = quaternion0
        w1, x1, y1, z1 = quaternion1
        return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                         x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                         -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                         x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)

    def get_action(self, idx, new_idx, image):
        a0 = self.dataset.loc[idx, self.action_headers].values.astype(np.float32)
        a1 = self.dataset.loc[new_idx, self.action_headers].values.astype(np.float32)

        q0, q1 = np.concatenate((a0[1:4],a0[0:1]), axis=0), np.concatenate((a1[1:4],a1[0:1]), axis=0)
        resq = self.quaternion_multiply((q0[3:4],-q0[0:1],-q0[1:2],-q0[2:3]), (q1[3:4],q1[0:1],q1[1:2],q1[2:3])).squeeze()
        imsizeM = np.array([float((image.size[0] > image.size[1]))])
        if not self.args.action_clamp:
            action = torch.tensor(np.concatenate((resq, a1[4:] - a0[4:], imsizeM), axis=0), dtype=torch.float32)
        else:
            action = torch.tensor(np.concatenate((resq, np.clip(a1[4:] - a0[4:], -50, 50), imsizeM), axis=0), dtype=torch.float32)

        action = (action - self.action_mean)/self.action_std
        return action

    def __getitem__(self, idx):
        # idx=530
        category, obj, frame_index = self.dataset.loc[idx, "category"],self.dataset.loc[idx, "object"],self.dataset.loc[idx, "frame"]
        size = self.dataset.loc[idx, "length"]
        image = self.open_image(str(category), str(obj),frame_index, self.dataset.loc[idx, "path"], idx)

        label = np.int64(self.category_mapping[category])

        if self.transform:
            image_t = self.transform(image)

        ft = self.args.mode in ["finetune","finetune_all"]
        if not self.contrastive or ft:
            return (image_t, self.transform(image) if self.split == "train" and not ft else image_t, torch.zeros((8,))), label

        if self.args.sampling_mode == "uniform":
            new_frame_index = random.randint(0, size-1)
        elif self.args.sampling_mode == "randomwalk+":
            new_frame_index = frame_index+1 if frame_index < size-1 else frame_index - 1
        else:
            new_frame_index = max(0, min(size-1, (frame_index-1 if random.random() < 0.5 else frame_index+1)))

        new_idx = idx + new_frame_index - frame_index

        assert str(obj) == str(self.dataset.loc[new_idx, "object"]), f"{str(obj)}, {str(self.dataset.loc[new_idx, 'object'])}, {idx}, {new_idx}, {frame_index}, {new_frame_index}, {size}"
        image_pair = self.open_image(str(category), str(obj), self.dataset.loc[new_idx, "frame"],self.dataset.loc[new_idx, "path"], new_idx)

        action = self.get_action(idx, new_idx, image)

        if self.transform:
            image_pair = self.transform(image_pair)
        return (image_t, image_pair, action), label

import os
import re
import math
import random
from typing import Dict, List, Tuple

import pandas as pd
import torch
from PIL import Image

def _wrap_to_pi(x: float) -> float:
    # maps to (-pi, pi]
    return (x + math.pi) % (2 * math.pi) - math.pi

def _extract_frame_id(fname: str) -> int:
    """
    Extract an integer frame index from filenames like:
      0000.png, 0001.jpg, frame_000123.png, etc.
    Falls back to 0 if nothing found (shouldn't happen if your naming is consistent).
    """
    base = os.path.splitext(os.path.basename(fname))[0]
    m = re.search(r"(\d+)$", base)
    return int(m.group(1)) if m else 0


class MAPSDataset(SimpleDataset):
    """
    Expected folder structure:
      MAPS_10k/
        001_goldfish/
          images/
            0000.png
            0001.png
            ...
          parameters.csv
        002_somecat/
          images/
          parameters.csv
        ...

    parameters.csv must contain columns:
      - image   (filename, e.g. "0000.png")
      - camera.azimuth
      - camera.distance
      - camera.elevation
      - camera.roll

    Returns (when contrastive=True):
      (img_t, img_t1, action), label
    where label is an integer category id derived from folder name prefix.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.root = self.args.data_root  # point this to .../MAPS_10k
        self.img_subdir = getattr(self.args, "maps_img_subdir", "images")
        self.logdist_scale = float(getattr(self.args, "maps_logdist_scale", 1.5))
        self.pair_mode = getattr(self.args, "maps_pair_mode", "next")  # "next" or "random_next"
        
        # Import here to avoid circular imports
        from utils.augmentations import get_action_indices_base
        self.action_indices = get_action_indices_base(self.args)

        # Discover category folders (e.g., "001_goldfish")
        all_cat_dirs = sorted(
            [d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d))]
        )

        if len(all_cat_dirs) == 0:
            raise FileNotFoundError(f"No category folders found under data_root={self.root}")

        # test train split
        split_ratio = float(getattr(self.args, "maps_train_split", 0.9))
        cat_dirs = all_cat_dirs  # keep ALL categories in both splits

        # --- NEW: contiguous label mapping (0..K-1) ---
        # Parse the numeric prefix (e.g., "404_dog" -> 404)
        cat_prefix: Dict[str, int] = {}
        for d in all_cat_dirs:
            m = re.match(r"^(\d+)_", d)
            if m:
                cat_prefix[d] = int(m.group(1))
            else:
                # fallback: if no prefix, assign -1 and handle below
                cat_prefix[d] = -1

        # Create a stable contiguous mapping based on sorted unique prefixes
        # (This ensures labels are in [0, K-1] and consistent across runs)
        uniq_prefixes = sorted(set(cat_prefix.values()))
        # if you had any -1 fallbacks, keep them too (they'll just become a valid class)
        prefix_to_contig = {p: i for i, p in enumerate(uniq_prefixes)}

        # Folder -> contiguous class index
        self.cat_name_to_label: Dict[str, int] = {d: prefix_to_contig[p] for d, p in cat_prefix.items()}

        # Optional: store original imagenet id per folder (useful for debugging/logging)
        self.cat_name_to_imagenet_id: Dict[str, int] = cat_prefix.copy()

        # Also store the mapping back (contig -> imagenet id)
        self.contig_to_imagenet_id = {prefix_to_contig[p]: p for p in uniq_prefixes}

        # Number of classes
        self.n_classes = len(uniq_prefixes)

        # with all 1000 classes
        # Build label mapping based on numeric prefix before underscore
        # self.cat_name_to_label: Dict[str, int] = {}
        # for d in all_cat_dirs:
        #     m = re.match(r"^(\d+)_", d)
        #     if m:
        #         self.cat_name_to_label[d] = int(m.group(1))
        #     else:
        #         # fallback if format differs
        #         self.cat_name_to_label[d] = len(self.cat_name_to_label)

        # Load per-category parameter tables and build a global index list
        # Each entry: (cat_dir, local_idx) where local_idx indexes into sorted rows for that category
        self.tables: Dict[str, pd.DataFrame] = {}
        self.image_dirs: Dict[str, str] = {}
        self.index: List[Tuple[str, int]] = []

        required_cols = ["image", "camera.azimuth", "camera.distance", "camera.elevation", "camera.roll"]

        for cat in cat_dirs:
            cat_path = os.path.join(self.root, cat)
            csv_path = os.path.join(cat_path, "parameters.csv")
            img_dir = os.path.join(cat_path, self.img_subdir)

            df = pd.read_csv(csv_path)

            # required columns check (unchanged)
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                raise ValueError(
                    f"CSV {csv_path} missing columns: {missing}\nFound: {list(df.columns)}"
                )

            # ensure correct temporal ordering
            df = df.copy()
            df["_frame_id"] = df["image"].astype(str).map(_extract_frame_id)
            df = df.sort_values("_frame_id").reset_index(drop=True)

            # -------- NEW SPLIT LOGIC --------
            n = len(df)
            split_idx = int(n * split_ratio)

            if self.split == "train":
                df = df.iloc[:split_idx].reset_index(drop=True)
            elif self.split == "test":
                df = df.iloc[split_idx:].reset_index(drop=True)
            # else: keep full df

            # --------------------------------

            self.tables[cat] = df
            self.image_dirs[cat] = img_dir

            for local_i in range(len(df)):
                self.index.append((cat, local_i))

        # Number of classes for linear eval modules etc.
        self.n_classes = len(all_cat_dirs)

        if len(self.index) == 0:
            raise RuntimeError("MAPSDataset index is empty (no rows found).")

    @classmethod
    def get_args(cls, parser):
        # optional knobs; safe defaults
        parser.add_argument("--maps_img_subdir", type=str, default="images")
        parser.add_argument("--maps_logdist_scale", type=float, default=1.5)
        parser.add_argument("--maps_pair_mode", type=str, default="next", choices=["next", "random_next"])
        parser.add_argument("--maps_train_split", type=float, default=0.9)
        return parser

    def __len__(self):
        return len(self.index)

    def _open_image(self, cat: str, fname: str) -> Image.Image:
        path = os.path.join(self.image_dirs[cat], fname)
        return Image.open(path).convert("RGB")

    def _camera_action_from_rows(self, row0: pd.Series, row1: pd.Series) -> torch.Tensor:
        az0, az1 = float(row0["camera.azimuth"]), float(row1["camera.azimuth"])
        el0, el1 = float(row0["camera.elevation"]), float(row1["camera.elevation"])
        rl0, rl1 = float(row0["camera.roll"]), float(row1["camera.roll"])
        d0, d1 = float(row0["camera.distance"]), float(row1["camera.distance"])

        daz = _wrap_to_pi(az1 - az0)
        drl = _wrap_to_pi(rl1 - rl0)

        # elevation: linear scaling
        delv = (el1 - el0) / math.pi

        # distance: log-ratio scaling
        d0 = max(d0, 1e-6)
        d1 = max(d1, 1e-6)
        dlog = math.log(d1 / d0) / self.logdist_scale

        full_action = torch.tensor(
            [
                math.sin(daz), math.cos(daz),
                math.sin(drl), math.cos(drl),
                delv,
                dlog
            ],
            dtype=torch.float32
        )
        
        # Filter action based on selected indices
        if self.action_indices is not None:
            full_action = full_action[self.action_indices]
        
        return full_action

    def __getitem__(self, idx: int):
        cat, local_i = self.index[idx]
        df = self.tables[cat]
        label = self.cat_name_to_label[cat]

        row0 = df.iloc[local_i]
        img0 = self._open_image(cat, str(row0["image"]))

        # Choose paired index within same category sequence
        if self.pair_mode == "random_next":
            # either next or previous with 50/50; stays in-bounds
            if local_i == 0:
                local_j = 1
            elif local_i == len(df) - 1:
                local_j = local_i - 1
            else:
                local_j = local_i + 1 if random.random() < 0.5 else local_i - 1
        else:
            # "next" (default): i -> i+1, last pairs to previous
            local_j = local_i + 1 if local_i < len(df) - 1 else local_i - 1

        row1 = df.iloc[local_j]
        img1 = self._open_image(cat, str(row1["image"]))

        action = self._camera_action_from_rows(row0, row1)

        # Apply transforms (match repo conventions)
        if self.transform:
            img0_t = self.transform(img0)
            img1_t = self.transform(img1)
        else:
            # fallback: convert to tensor if no transform is provided
            # (repo usually passes a transform, so this is rarely used)
            from torchvision.transforms import ToTensor
            tt = ToTensor()
            img0_t = tt(img0)
            img1_t = tt(img1)

        if not self.contrastive:
            # keep shape consistent with other datasets in the repo
            return (img0_t, img0_t, torch.zeros_like(action)), label

        return (img0_t, img1_t, action), label

import os
import re
import math
import random
from typing import Dict, List, Tuple

import pandas as pd
import torch
from PIL import Image


class MAPSInstanceSplitDataset(SimpleDataset):
    """
    MAPS variant where each object has multiple instance subfolders.

    Folder structure:
      root/
        object_name/
          object_name_001/
            images/
            parameters.csv
          object_name_002/
            images/
            parameters.csv
          ...
          object_name_005/
            images/
            parameters.csv

    Split rule (per object):
      - train: use first N-1 instances (default 4 of 5)
      - test:  use last instance (default 1 of 5)

    Pairing:
      pairing is done WITHIN the same instance sequence (never across instances),
      using maps_pair_mode: "next" or "random_next".

    Returns:
      (img0_t, img1_t, action), label
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.root = self.args.data_root
        self.img_subdir = getattr(self.args, "maps_img_subdir", "images")
        self.logdist_scale = float(getattr(self.args, "maps_logdist_scale", 1.5))
        self.pair_mode = getattr(self.args, "maps_pair_mode", "next")  # next / random_next
        
        # Import here to avoid circular imports
        from utils.augmentations import get_action_indices_base
        self.action_indices = get_action_indices_base(self.args)

        # which instance index to hold out as test (default: last)
        # -1 means "last instance after sorting"
        self.test_instance_index = int(getattr(self.args, "maps_test_instance_index", -1))

        # If you ever have >5 or <5 instances, this is safer than hardcoding "4/1"
        # train uses all except the held-out instance
        # test uses only the held-out instance
        # (If you want to force 4/1 even with more instances, tell me and I’ll adapt.)

        # Discover top-level object dirs (classes)
        object_dirs = sorted([
            d for d in os.listdir(self.root)
            if os.path.isdir(os.path.join(self.root, d))
        ])
        if not object_dirs:
            raise FileNotFoundError(f"No object folders found under data_root={self.root}")

        # Map object -> contiguous label [0..C-1] (works with your 10-class head setup)
        self.obj_to_label: Dict[str, int] = {obj: i for i, obj in enumerate(object_dirs)}
        self.n_classes = len(object_dirs)

        # Storage per (object, instance)
        # tables[(obj, inst)] = df
        self.tables: Dict[Tuple[str, str], pd.DataFrame] = {}
        self.image_dirs: Dict[Tuple[str, str], str] = {}
        # global index list entries: (obj, inst, local_i)
        self.index: List[Tuple[str, str, int]] = []

        required_cols = ["image", "camera.azimuth", "camera.distance", "camera.elevation", "camera.roll", "background.hue",
                         "background.noise", "background.saturation", "background.value", "light.azimuth", "light.elevation",
                         "light.power"]

        # Iterate objects and their instance subfolders
        for obj in object_dirs:
            obj_path = os.path.join(self.root, obj)

            instance_dirs = sorted([
                d for d in os.listdir(obj_path)
                if os.path.isdir(os.path.join(obj_path, d))
            ])

            # Keep only dirs that look like "<obj>_001" etc. (but still tolerant)
            # If your instance dirs always match this, great; if not, it still works.
            # Example pattern: acoustic-guitar_001
            # We'll sort by trailing digits if present.
            def inst_sort_key(name: str):
                m = re.search(r"(\d+)$", name)
                return (int(m.group(1)) if m else 10**9, name)

            instance_dirs = sorted(instance_dirs, key=inst_sort_key)

            if not instance_dirs:
                continue

            # pick held-out instance
            ti = self.test_instance_index
            if ti < 0:
                ti = len(instance_dirs) - 1
            if not (0 <= ti < len(instance_dirs)):
                raise ValueError(
                    f"maps_test_instance_index={self.test_instance_index} out of range for object '{obj}' "
                    f"with {len(instance_dirs)} instances."
                )

            test_inst = instance_dirs[ti]
            train_insts = [d for i, d in enumerate(instance_dirs) if i != ti]

            if self.split == "train":
                used_instances = train_insts
            elif self.split == "test":
                used_instances = [test_inst]
            else:
                used_instances = instance_dirs  # e.g. eval/val modes if you ever use them

            for inst in used_instances:
                inst_path = os.path.join(obj_path, inst)
                csv_path = os.path.join(inst_path, "parameters.csv")
                img_dir = os.path.join(inst_path, self.img_subdir)

                if not os.path.exists(csv_path):
                    raise FileNotFoundError(f"Missing parameters.csv: {csv_path}")
                if not os.path.isdir(img_dir):
                    raise FileNotFoundError(f"Missing images folder: {img_dir}")

                df = pd.read_csv(csv_path)
                missing = [c for c in required_cols if c not in df.columns]
                if missing:
                    raise ValueError(
                        f"CSV {csv_path} missing columns: {missing}\nFound: {list(df.columns)}"
                    )

                # enforce temporal ordering inside the instance
                df = df.copy()
                df["_frame_id"] = df["image"].astype(str).map(_extract_frame_id)
                df = df.sort_values("_frame_id").reset_index(drop=True)

                key = (obj, inst)
                self.tables[key] = df
                self.image_dirs[key] = img_dir

                for local_i in range(len(df)):
                    self.index.append((obj, inst, local_i))

        if len(self.index) == 0:
            raise RuntimeError("MAPSInstanceSplitDataset index is empty. Check folder structure / CSVs.")

    @classmethod
    def get_args(cls, parser):
        def _add_arg_once(parser, option: str, **kwargs):
            # argparse stores already-added options in parser._option_string_actions
            if option in getattr(parser, "_option_string_actions", {}):
                return
            parser.add_argument(option, **kwargs)
        _add_arg_once(parser, "--maps_img_subdir", type=str, default="images")
        _add_arg_once(parser, "--maps_logdist_scale", type=float, default=1.5)
        _add_arg_once(parser, "--maps_pair_mode", type=str, default="next",
                      choices=["next", "random_next"])
        _add_arg_once(parser, "--maps_test_instance_index", type=int, default=-1)
        return parser

    def __len__(self):
        return len(self.index)

    def _open_image(self, key: Tuple[str, str], fname: str) -> Image.Image:
        path = os.path.join(self.image_dirs[key], fname)
        return Image.open(path).convert("RGB")

    def _action_from_rows(self, row0: pd.Series, row1: pd.Series) -> torch.Tensor:
        # --- read values (safe) ---
        def _wrap_to_pi(x: float) -> float:
            return (x + math.pi) % (2 * math.pi) - math.pi

        def _safe_float(v, default=0.0) -> float:
            # handles NaN / missing gracefully
            try:
                x = float(v)
                if math.isnan(x) or math.isinf(x):
                    return default
                return x
            except Exception:
                return default

        def _log_ratio(d1: float, d0: float, scale: float) -> float:
            d0 = max(d0, 1e-6)
            d1 = max(d1, 1e-6)
            return math.log(d1 / d0) / scale
        cam_az0 = _safe_float(row0["camera.azimuth"])
        cam_az1 = _safe_float(row1["camera.azimuth"])
        cam_rl0 = _safe_float(row0["camera.roll"])
        cam_rl1 = _safe_float(row1["camera.roll"])
        cam_el0 = _safe_float(row0["camera.elevation"])
        cam_el1 = _safe_float(row1["camera.elevation"])
        cam_d0 = _safe_float(row0["camera.distance"], default=1.0)
        cam_d1 = _safe_float(row1["camera.distance"], default=1.0)

        bg_h0 = _safe_float(row0["background.hue"])
        bg_h1 = _safe_float(row1["background.hue"])
        bg_n0 = _safe_float(row0["background.noise"])
        bg_n1 = _safe_float(row1["background.noise"])
        bg_s0 = _safe_float(row0["background.saturation"])
        bg_s1 = _safe_float(row1["background.saturation"])
        bg_v0 = _safe_float(row0["background.value"])
        bg_v1 = _safe_float(row1["background.value"])

        li_az0 = _safe_float(row0["light.azimuth"])
        li_az1 = _safe_float(row1["light.azimuth"])
        li_el0 = _safe_float(row0["light.elevation"])
        li_el1 = _safe_float(row1["light.elevation"])
        li_p0 = _safe_float(row0["light.power"])
        li_p1 = _safe_float(row1["light.power"])

        # --- meaningful deltas ---
        # angles -> sin/cos of wrapped delta
        d_cam_az = _wrap_to_pi(cam_az1 - cam_az0)
        d_cam_rl = _wrap_to_pi(cam_rl1 - cam_rl0)
        d_bg_h = _wrap_to_pi(bg_h1 - bg_h0)
        d_li_az = _wrap_to_pi(li_az1 - li_az0)

        # elevations -> scaled delta
        d_cam_el = (cam_el1 - cam_el0) / math.pi
        d_li_el = (li_el1 - li_el0) / math.pi

        # distance -> log ratio
        d_cam_d = _log_ratio(cam_d1, cam_d0, self.logdist_scale)

        # other scalars -> simple delta
        d_bg_n = bg_n1 - bg_n0
        d_bg_s = bg_s1 - bg_s0
        d_bg_v = bg_v1 - bg_v0
        d_li_p = li_p1 - li_p0

        # final action (15 dims)
        full_action = torch.tensor(
            [
                math.sin(d_cam_az), math.cos(d_cam_az),
                math.sin(d_cam_rl), math.cos(d_cam_rl),
                d_cam_el,
                d_cam_d,
                math.sin(d_bg_h), math.cos(d_bg_h),
                d_bg_n, d_bg_s, d_bg_v,
                math.sin(d_li_az), math.cos(d_li_az),
                d_li_el,
                d_li_p,
            ],
            dtype=torch.float32
        )
        
        # Filter action based on selected indices
        if self.action_indices is not None:
            full_action = full_action[self.action_indices]
        
        return full_action

    def __getitem__(self, idx: int):
        obj, inst, local_i = self.index[idx]
        key = (obj, inst)
        df = self.tables[key]

        label = self.obj_to_label[obj]

        row0 = df.iloc[local_i]
        img0 = self._open_image(key, str(row0["image"]))

        # pair index within SAME instance
        if self.pair_mode == "random_next":
            if local_i == 0:
                local_j = 1
            elif local_i == len(df) - 1:
                local_j = local_i - 1
            else:
                local_j = local_i + 1 if random.random() < 0.5 else local_i - 1
        else:
            local_j = local_i + 1 if local_i < len(df) - 1 else local_i - 1

        row1 = df.iloc[local_j]
        img1 = self._open_image(key, str(row1["image"]))
        action = self._action_from_rows(row0, row1)

        if self.transform:
            img0_t = self.transform(img0)
            img1_t = self.transform(img1)
        else:
            from torchvision.transforms import ToTensor
            tt = ToTensor()
            img0_t = tt(img0)
            img1_t = tt(img1)

        if not self.contrastive:
            return (img0_t, img0_t, torch.zeros_like(action)), label

        return (img0_t, img1_t, action), label

import os
from typing import List, Dict

def find_wnid_tars_flat(root: str, wnids: List[str]) -> Dict[str, str]:
    """
    Layout:
      root/
        n01440764.tar
        n01443537.tar
        ...

    Returns wnid -> tar_path
    """
    out = {}
    missing = []
    for wnid in wnids:
        p = os.path.join(root, f"{wnid}.tar")
        if not os.path.isfile(p):
            missing.append(wnid)
        else:
            out[wnid] = p

    if missing:
        raise FileNotFoundError(
            f"Missing tar files for WNIDs: {missing}\n"
            f"Expected under: {root}\n"
            f"Example: {os.path.join(root, missing[0] + '.tar') if missing else ''}"
        )
    return out

import io
import os
import tarfile
from typing import List, Tuple, Optional

import torch
from torch.utils.data import Dataset
from PIL import Image


class ImageNetPerClassTarDataset(Dataset):
    """
    ImageNet train layout:
      root/ILSVRC2012_img_train/
        n01440764.tar
        n01443537.tar
        ...

    Each tar contains images for that class.

    This dataset loads ONLY the provided WNIDs and remaps labels to [0..K-1].

    Returns:
      (img0_t, img1_t, action), label
    """

    IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    def __init__(
        self,
        root: str,
        wnids: List[str],
        transform=None,
        contrastive: bool = True,
        action_dim: int = 16,
        index_cache_path: Optional[str] = None,
    ):
        super().__init__()
        self.root = root
        self.transform = transform
        self.contrastive = contrastive
        self.action_dim = int(action_dim)

        # de-dupe, stable order
        self.wnids = list(dict.fromkeys(wnids))
        self.wnid_to_label = {w: i for i, w in enumerate(self.wnids)}
        self.n_classes = len(self.wnids)

        # resolve tar paths
        self.wnid_to_tar = find_wnid_tars_flat(root, self.wnids)

        # load / build index: (tar_path, member_name, label)
        if index_cache_path is not None and os.path.isfile(index_cache_path):
            import pickle
            with open(index_cache_path, "rb") as f:
                cache = pickle.load(f)
            if cache.get("wnids") != self.wnids:
                raise ValueError("Index cache WNIDs do not match current WNIDs selection.")
            self.samples = cache["samples"]
        else:
            self.samples = self._build_index()
            if index_cache_path is not None:
                import pickle
                os.makedirs(os.path.dirname(index_cache_path), exist_ok=True)
                with open(index_cache_path, "wb") as f:
                    pickle.dump({"wnids": self.wnids, "samples": self.samples}, f)

        if len(self.samples) == 0:
            raise RuntimeError("No images found inside the selected WNID tar files.")

    @classmethod
    def get_args(cls, parser):
        # add args only once (prevents "conflicting option string" if multiple datasets share flags)
        def _add_arg_once(name, **kwargs):
            if name in getattr(parser, "_option_string_actions", {}):
                return
            parser.add_argument(name, **kwargs)

        _add_arg_once("--imagenet_root", type=str, default="", help="Path to ImageNet per-class tar folder")
        _add_arg_once("--imagenet_wnids", type=str, default="", help="Comma-separated list of WNIDs")
        _add_arg_once("--imagenet_train_ratio", type=float, default=0.8)
        _add_arg_once("--imagenet_split_seed", type=int, default=0)
        _add_arg_once("--imagenet_index_cache", type=str, default="", help="Cache dir for tar indices")

        return parser

    def _build_index(self) -> List[Tuple[str, str, int]]:
        samples: List[Tuple[str, str, int]] = []
        for wnid in self.wnids:
            tar_path = self.wnid_to_tar[wnid]
            label = self.wnid_to_label[wnid]

            with tarfile.open(tar_path, "r:*") as tf:
                for m in tf.getmembers():
                    if not m.isfile():
                        continue
                    name = m.name
                    if name.lower().endswith(self.IMG_EXTS):
                        samples.append((tar_path, name, label))
        return samples

    def __len__(self):
        return len(self.samples)

    def _load_image_from_tar(self, tar_path: str, member_name: str) -> Image.Image:
        with tarfile.open(tar_path, "r:*") as tf:
            f = tf.extractfile(member_name)
            if f is None:
                raise FileNotFoundError(f"Missing member in tar: {tar_path} :: {member_name}")
            data = f.read()
        return Image.open(io.BytesIO(data)).convert("RGB")

    def __getitem__(self, idx: int):
        tar_path, member_name, label = self.samples[idx]
        img = self._load_image_from_tar(tar_path, member_name)

        action = torch.zeros((self.action_dim,), dtype=torch.float32)

        if self.transform is not None:
            img0 = self.transform(img)
            img1 = self.transform(img) if self.contrastive else img0
        else:
            from torchvision.transforms import ToTensor
            tt = ToTensor()
            img0 = tt(img)
            img1 = img0

        return (img0, img1, action), int(label)
