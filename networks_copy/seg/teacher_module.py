from copy import deepcopy

import numpy as np
import torch
from torch.nn import Module
from torch.nn.modules.dropout import _DropoutNd
from data.data_segmentation import weight_binary_ratio
from networks_copy.seg.model_superhuman import UNet_PNI

def get_module(module):
    """Get `nn.ModuleDict` to fit the `MMDistributedDataParallel` interface.

    Args:
        module (MMDistributedDataParallel | nn.ModuleDict): The input
            module that needs processing.

    Returns:
        nn.ModuleDict: The ModuleDict of multiple networks.
    """
    if isinstance(module,  torch.nn.parallel.DistributedDataParallel):
        return module.module

    return module




class EMATeacher(Module):

    def __init__(self, device, args):
        super(EMATeacher, self).__init__()

        self.alpha = args.seg_train["alpha"] # 0.999, 0.9

        self.pseudo_threshold = args.seg_train["pseudo_threshold"] # # 0.968
        # self.psweight_ignore_top = cfg['pseudo_weight_ignore_top']
        # self.psweight_ignore_bottom = cfg['pseudo_weight_ignore_bottom']
        self.auto_mask_tgt = args.seg_train["auto_mask_tgt"]
        self.mask_loss = args.seg_train["mask_loss"]

        # self.ema_model = define_instance(args, "seg_def").to(device)
        self.ema_model=UNet_PNI(in_planes=1,
                            out_planes=3,
                            filters=[28, 36, 48, 64, 80],
                            upsample_mode="bilinear",
                            decode_ratio=1,
                            merge_mode="add",
                            pad_mode="zero",
                            bn_mode="async",
                            relu_mode="elu",
                            init_mode="kaiming_normal").to(device)

        self.debug = False
        self.debug_output = {}

    def get_ema_model(self):
        return get_module(self.ema_model)

    def _init_ema_weights(self, model):
        for param in self.get_ema_model().parameters():
            param.detach_()
        mp = list(model.parameters())
        mcp = list(self.get_ema_model().parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()

    def _update_ema(self, model, iter):
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
        for ema_param, param in zip(self.get_ema_model().parameters(),
                                    model.parameters()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]

    def update_debug_state(self):
        self.get_ema_model().automatic_debug = False
        self.get_ema_model().debug = self.debug


    def update_weights(self, model, iter):
        # Init/update ema model
        if iter == 0:
            self._init_ema_weights(model)

        if iter > 0:
            self._update_ema(model, iter)

    def __call__(self, target_img,show_feature=False, if_thres=False, if_sigmoid=True):
        self.update_debug_state()

        # Generate pseudo-label
        for m in self.get_ema_model().modules():
            if isinstance(m, _DropoutNd):
                m.training = False
                
        outputs=self.ema_model(target_img)

        device = outputs.device

        if if_thres: 
            thres = self.pseudo_threshold
            outputs = torch.clamp(outputs, min=0.000001, max=0.999999)
            outputs[outputs <= thres] = 0
            outputs[outputs > thres] = 1

            if self.auto_mask_tgt:
                pseudo_weight = weight_binary_ratio(outputs.detach().cpu().numpy())
                pseudo_weight = torch.from_numpy(pseudo_weight).to(device)
            else:
                pseudo_weight = outputs.clone().detach()
                pseudo_weight[pseudo_weight==0] = self.mask_loss

            return outputs, pseudo_weight
        else:
            return outputs