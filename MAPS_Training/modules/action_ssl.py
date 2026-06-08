import torch

from modules.loss_module import LossModule
from networks.heads import MLPHead
from utils.augmentations import get_action_size
from utils.constants import LOSS, SIMILARITY_FUNCTIONS
from utils.general import normalize, is_target_needed, str2bool
import numpy as np

class ActionSSL(LossModule):
    def __init__(self,args, fabric, net=None, net_target=None, **kwargs):
        self.args = args
        self.fabric=fabric
        net.add_module("action_projector", MLPHead(args, 2*net.num_output, args.action_dim, args.action_feature_dim, self.args.action_layers))
        net.register_buffer("action_proj_output", torch.empty((args.batch_size, args.feature_dim)), persistent=False)

        # full action size as provided by dataset/registry
        action_full_size = get_action_size(args)

        # new arg: which subspace of the action vector the action head receives
        # choices: 'all' (default), 'camera', 'background', 'light'
        action_subspace = getattr(self.args, "action_subspace", "all")
        self.action_subspace = action_subspace

        # mapping for known action-size layouts (MAPSInstanceSplitDataset = 15 dims)
        if action_subspace == "all":
            action_size = action_full_size
            self._action_indices = None
        else:
            # build index mapping for the known 15-dim MAPS action
            if action_full_size == 15:
                mapping = {
                    "camera": list(range(0,6)),        # 6 dims
                    "background": list(range(6,11)),   # 5 dims
                    "light": list(range(11,15)),       # 4 dims
                }
            elif action_full_size == 6:
                # legacy MAPS (6-dim camera-only) -> only camera supported
                mapping = {"camera": list(range(0,6))}
            else:
                raise ValueError(f"Unsupported action_full_size={action_full_size} for action_subspace selection")

            if action_subspace not in mapping:
                raise ValueError(f"Requested action_subspace='{action_subspace}' not available for action size {action_full_size}")

            self._action_indices = mapping[action_subspace]
            action_size = len(self._action_indices)

        # create action head expecting the reduced action size
        net.add_module("action_head", torch.nn.Sequential(
            MLPHead(args, action_size, self.args.hidden_dim,  args.action_feature_dim),torch.nn.BatchNorm1d(args.action_feature_dim, affine=False)))
        # self.loss = LOSS[args.main_loss](args, SIMILARITY_FUNCTIONS[args.similarity])
        self.loss = LOSS[self.args.main_loss](args, SIMILARITY_FUNCTIONS[args.similarity], fabric, temperature=self.args.action_temperature, lambda_vicreg=self.args.lambda_vicreg_a, mu_vicreg=self.args.mu_vicreg_a, v_vicreg=self.args.v_vicreg_a)

        if is_target_needed(args):
            net.add_module("action_predictor",MLPHead(args,args.action_feature_dim, args.action_dim, args.action_feature_dim,args.hidden_layers))
            net.add_module("action_head_predictor",MLPHead(args,args.action_feature_dim, args.action_dim, args.action_feature_dim,args.hidden_layers))

        self.loss_store = self.fabric.to_device(torch.zeros((1,)))
        # action_mean/std now sized according to the reduced action head input
        self.action_mean = self.fabric.to_device(torch.zeros((action_size,)))
        self.action_std = self.fabric.to_device(torch.zeros((action_size,)))
        self.loss_cpt = 1e-5

        print(
            f"[ActionSSL] action_subspace={self.action_subspace} action_full_size={action_full_size} action_size={self.action_mean.shape[0]} _action_indices={self._action_indices}",
            flush=True)

    def apply(self, net, rep=None, net_target=None, rep_target=None, action=None, data=None, **kwargs):
        # Slice action to the configured subspace if needed
        if self._action_indices is None:
            reduced_action = action
        else:
            # action shape: (B, full_action_size)
            inds = self._action_indices
            # gather maintains dtype/device
            reduced_action = action[:, inds]
        print(f"[ActionSSL.apply] reduced_action.shape={tuple(reduced_action.shape)} device={reduced_action.device}",
              flush=True)

        # Compute action representation
        action_rep = net.action_head(reduced_action)

        # Compute action prediction
        output = torch.cat(rep.split(rep.shape[0] // 2), dim=1)
        net.action_proj_output = net.action_projector(output)
        if self.args.main_loss in ['BYOL']:
            rep_target = torch.cat(rep_target.split(rep.shape[0] // 2), dim=1)
            y1 = net_target.action_projector(torch.cat(rep_target.split(rep.shape[0] // 2), dim=1)).detach()
            # pass reduced action also to target head
            y2 = net_target.action_head(reduced_action).detach()

            y11 = net.action_predictor(net.action_proj_output)
            y22 = net.action_head_predictor(action_rep)

            loss_mean = self.loss(torch.cat((y1,y2),dim=0), torch.cat((y22,y11),dim=0)).mean()
        else:
            # Compute loss
            loss_mean = self.args.action_weight * self.loss(action_rep, net.action_proj_output).mean()

        # Compute loss stats using the reduced_action
        self.loss_store += loss_mean.detach()
        self.action_mean += reduced_action.mean(dim=0)
        self.action_std += torch.pow(reduced_action,2).mean(dim=0)

        self.loss_cpt += 1
        return loss_mean

    @torch.no_grad()
    def eval(self, net, *args):
        dict = {"action_loss": self.loss_store.item()/self.loss_cpt}
        for a in range(self.action_mean.shape[0]):
            dict[f"a{a}_mean"] = self.action_mean[a].cpu().item()/self.loss_cpt
            dict[f"a{a}_std"] = np.sqrt(self.action_std[a].cpu().item()/self.loss_cpt - dict[f"a{a}_mean"]**2)
        self.action_mean.zero_()
        self.action_std.zero_()
        self.loss_store[:]=0
        self.loss_cpt = 1e-5
        return dict

    @classmethod
    def get_args(cls, parser):
        parser.add_argument('--lambda_vicreg_a', default=25, type=float)
        parser.add_argument('--mu_vicreg_a', default=25, type=float)
        parser.add_argument('--v_vicreg_a', default=1, type=float)
        parser.add_argument('--action_temperature', default=0.1, type=float)
        parser.add_argument('--action_dim', default=1024, type=int)
        parser.add_argument('--action_feature_dim', default=128, type=int)
        parser.add_argument('--action_layers', default=1, type=int)
        # new: choose which part of the action vector the action_head receives
        parser.add_argument('--action_subspace', default='all', type=str, choices=['all','camera','background','light'],
                            help="Which subspace of the full action vector is fed to the action head (default 'all')")
        return parser
