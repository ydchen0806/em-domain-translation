# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import logging
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np

from monai.networks.utils import copy_model_state
from monai.utils import RankFilter

from torch.amp import GradScaler, autocast  # 
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import PolynomialLR


from .diff_model_setting import initialize_distributed, load_config, setup_logging
from .utils import binarize_labels, define_instance, prepare_maisi_controlnet_json_dataloader, setup_ddp
from .sample import ldm_conditional_sample_translated_image, ldm_conditional_sample_translated_image_lb

# from torch.utils.data import DataLoader
from monai.data import DataLoader
from data.dataset_lb import CondTargetDataset_lb,CondTargetDataset_sampling
from data.provider_valid import ValidDataset_seg
from scripts.diff_model_infer_self import visual_saver, visual_saver_with_cond

from data.data_segmentation import weight_binary_ratio

import waterz
from utils.show import show_affs, show_affs_whole
from utils.fragment import watershed

from skimage.metrics import adapted_rand_error as adapted_rand_ref
from skimage.metrics import variation_of_information as voi_ref
from sklearn.metrics import f1_score
from networks_copy.seg.model_superhuman import UNet_PNI


# from utils.lmc import mc_baseline # 不确定
from utils.utils import get_current_consistency_weight


from loss.loss import WeightedMSE, WeightedBCE,MSELoss, BCELoss
from networks_copy.seg.teacher_module import EMATeacher


import warnings

warnings.filterwarnings("ignore")


def l2_normalize(x):
    return F.normalize(x, p=2, dim=-1)



class AddMask3D:
    def __init__(self, mask_block_size=(4,4,4)):
        """
        mask_block_size: tuple of 3 ints (bz, bh, bw) 对应 Z,H,W 的块大小
        """
        assert isinstance(mask_block_size, (tuple, list)) and len(mask_block_size) == 3, \
            "mask_block_size 必须是长度为3的 tuple 或 list"
        self.mask_block_size = mask_block_size

    def __call__(self, batch_data, mask_ratio=0.5):
        from scipy.ndimage import zoom
        """
        batch_data: torch.Tensor [B, 1, Z, H, W]
        mask_ratio: 掩码率，0~1之间
        """
        batch_size, c, z, h, w = batch_data.shape
        assert c == 1, "目前只支持单通道数据"

        bz, bh, bw = self.mask_block_size
        masked_batch = batch_data.clone()

        for b in range(batch_size):
            # 生成粗略 mask，分别按 Z/H/W 的块数计算
            mshape = (
                max(1, round(z / bz)),
                max(1, round(h / bh)),
                max(1, round(w / bw))
            )
            mask_small = np.random.rand(*mshape)
            mask_small = (mask_small > mask_ratio).astype(np.float32)

            # 放大到原图尺寸
            zoom_factors = (z / mshape[0], h / mshape[1], w / mshape[2])
            mask = zoom(mask_small, zoom_factors, order=0)

            # 转成 torch Tensor
            mask_tensor = torch.from_numpy(mask).float().to(batch_data.device)
            mask_tensor = mask_tensor.unsqueeze(0)  # [1, Z, H, W]

            # 应用掩码
            masked_batch[b] = masked_batch[b] * mask_tensor

        return masked_batch

def feature_alignment(fea_src, fea_tgt, gt_src, threshold=0.7):
    var_loss = torch.tensor(0, dtype=fea_src.dtype, device=fea_src.device)
    align_loss = torch.tensor(0, dtype=fea_src.dtype, device=fea_src.device)
    dist_losses = []

    fea_src = F.normalize(fea_src, p=2, dim=1)  # L2 norm
    fea_tgt = F.normalize(fea_tgt, p=2, dim=1)  # L2 norm
    feature_dim = fea_src.shape[1]

    # import random
    # random_number = random.randint(1000, 9999)
    
    assert fea_src.shape[0] == gt_src.shape[0]

    # discriminative loss used for source domain
    num_classes = 2 # two classes

    # (b, c, d, h, w)
    fea_src = fea_src.permute(0, 2, 3, 4, 1)
    num_channels = gt_src.shape[1] # 3

    for channel in range(num_channels):
        gt_src_channel = gt_src[:, channel, :, :, :]
        channel_var_loss = torch.tensor(0, dtype=fea_src.dtype, device=fea_src.device)
        channel_align_loss = torch.tensor(0, dtype=fea_src.dtype, device=fea_src.device)
        channel_dist_loss = 0.0

        # discriminative loss used for source domain
        proto = []

        for k in range(num_classes):
            mask_k = gt_src_channel == k
            emb_k = fea_src[mask_k, :]
            num = emb_k.shape[0]
            if num == 0:
                continue

            proto_k = l2_normalize(torch.mean(emb_k, dim=0))
            proto.append(proto_k)

            channel_var_loss += (1 - torch.mm(emb_k, proto_k.reshape(feature_dim, 1))).pow(2).mean() / num_classes

        proto = torch.stack(proto)
        if proto.shape[0] == 2:
            channel_dist_loss = torch.sum(proto[0] * proto[1]).pow(2)

            # alignment
            b, c, d, h, w = fea_tgt.shape
            fea_tgt_emb = fea_tgt.permute(0, 2, 3, 4, 1)
            fea_tgt_emb = torch.reshape(fea_tgt_emb, (-1, feature_dim))
            fea_tgt_emb = l2_normalize(fea_tgt_emb)
            proto = l2_normalize(proto)
            seg_tgt = torch.einsum('nd,md->nm', fea_tgt_emb, proto)

            for k in range(proto.shape[0]):
                seg_tgt_k = seg_tgt[:, k]
                mask_k = seg_tgt_k > threshold
                emb_k = seg_tgt_k[mask_k]
                num = emb_k.shape[0]
                if num == 0:
                    continue

                channel_align_loss += (1 - emb_k.pow(2).mean()) / num_classes

            # seg_tgt = torch.reshape(seg_tgt, (b, d, h, w, -1))
            # seg_tgt = seg_tgt.permute(0, 4, 1, 2, 3)
            del seg_tgt
            # [b, 2, d, h, w]
        else:
            channel_dist_loss = 0.0
            channel_align_loss = 0.0
            seg_tgt = None
        ## tgt_like_proto_pred还是难度太高了
        # feat_i = seg_tgt.detach().cpu().numpy() # [b, 2, d, h, w]
        # save_path = "./outputs/exp_tmp/"

        # file_name=os.path.join(save_path, str(random_number)+'_'+str(channel)+'.png')
        
        # plt.figure()

        # ax = plt.subplot(1, 4, 1)
        # plt.imshow(gt_src.detach().cpu().numpy()[0][channel,0])
        # plt.axis('off') 

        # ax = plt.subplot(1, 4, 2)
        # plt.imshow(feat_i[0][0,0])
        # plt.axis('off') 

        # ax = plt.subplot(1, 4, 3)
        # plt.imshow(feat_i[0][1,0])
        # plt.axis('off') 

        # seg_tgt_s = torch.argmax(seg_tgt, dim=1) # [b, d, h, w]
        # ax = plt.subplot(1, 4, 4)
        # plt.imshow(seg_tgt_s.detach().cpu().numpy()[0,0])
        # plt.axis('off') 

        # plt.savefig(file_name,dpi=300)

        var_loss += channel_var_loss
        align_loss += channel_align_loss
        dist_losses.append(channel_dist_loss)

    var_loss /= num_channels
    align_loss /= num_channels
    dist_loss = sum(dist_losses) / num_channels

    return var_loss, dist_loss, align_loss, None


def get_scheduler(optimizer, total_steps, warmup_iters, power, min_lr_ratio=1e-6):
    
    # 这里的 current_step 是由 LambdaLR 自动传入的内部变量
    def lr_lambda(current_step):
        # 1. Linear Warmup 阶段
        if current_step < warmup_iters:
            # 返回的是初始学习率的倍数
            return max(min_lr_ratio, current_step / warmup_iters)
        
        # 2. Poly Decay 阶段
        # 确保不会出现负数或除以零
        if current_step >= total_steps:
            return 0.0
            
        decay_steps = total_steps - warmup_iters
        current_decay_step = current_step - warmup_iters
        
        # 公式: (1 - curr_decay / total_decay)^power
        return (1.0 - current_decay_step / decay_steps) ** power

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

def process_tensor_to_uint8(tensor):
    """
    将torch tensor转换为0-255的uint8 numpy数组，逻辑与原代码完全一致
    Args:
        tensor: 待处理的torch tensor（维度需包含[0,0]索引）
    Returns:
        uint8类型的numpy数组，值范围0-255
    """
    arr = tensor[0,0].cpu().detach().numpy()
    arr = arr * 255.0
    arr = np.clip(arr, 0, 255)
    arr = arr.astype(np.uint8)
    return arr


def display_saver(tensorboard_writer,epoch,
                  src,src_pred,lbs,
                  sample,sample_pred,
                  tgt_img,tgt_pred,
                  tgt_aug,tgt_aug_pred):
    # print('src:', src.shape)
    # print('src_pred:', src_pred.shape)
    # print('lbs:', lbs.shape)
    # print('sample:', sample.shape)
    # print('sample_pred:', sample_pred.shape)
    # print('tgt_img:', tgt_img.shape)
    # print('tgt_pred:', tgt_pred.shape)
    # print('tgt_aug:', tgt_aug.shape)
    # print('tgt_aug_pred:', tgt_aug_pred.shape)
    
    src = process_tensor_to_uint8(src)
    src_pred = process_tensor_to_uint8(src_pred)
    lbs = process_tensor_to_uint8(lbs)
    sample = process_tensor_to_uint8(sample)
    sample_pred = process_tensor_to_uint8(sample_pred)
    tgt_img = process_tensor_to_uint8(tgt_img)
    tgt_pred = process_tensor_to_uint8(tgt_pred)
    tgt_aug = process_tensor_to_uint8(tgt_aug)
    tgt_aug_pred = process_tensor_to_uint8(tgt_aug_pred)
    

    k=0
    tensorboard_writer.add_image(
        "src/image",
        torch.from_numpy(src[k]).unsqueeze(0),
        epoch,
    )
    tensorboard_writer.add_image(
        "src/image_pred",
        torch.from_numpy(src_pred[k]).unsqueeze(0),
        epoch,
    )
    tensorboard_writer.add_image(
        "src/label",
        torch.from_numpy(lbs[k]).unsqueeze(0),
        epoch,
    )
    tensorboard_writer.add_image(
        "src/sample",
        torch.from_numpy(sample[k]).unsqueeze(0),
        epoch,
    )
    tensorboard_writer.add_image(
        "src/sample_pred",
        torch.from_numpy(sample_pred[k]).unsqueeze(0),
        epoch,
    )
    tensorboard_writer.add_image(
        "tgt/image",
        torch.from_numpy(tgt_img[k]).unsqueeze(0),
        epoch,
    )
    tensorboard_writer.add_image(
        "tgt/image_pred",
        torch.from_numpy(tgt_pred[k]).unsqueeze(0),
        epoch,
    )
    tensorboard_writer.add_image(
        "tgt/aug",
        torch.from_numpy(tgt_aug[k]).unsqueeze(0),
        epoch,
    )
    tensorboard_writer.add_image(
        "tgt/aug_pred",
        torch.from_numpy(tgt_aug_pred[k]).unsqueeze(0),
        epoch,
    )
    

def apply_autoencoder(
        autoencoder,
        inputs, # [b, 1, 160, 160, z]  -> [b, 4, 40, 40, z]
        use_vqgan=False):
    if use_vqgan:
        z = autoencoder.encode(inputs.permute(0,1,4,3,2),
                                               quantize=False, include_embeddings=True)
        # print('min:',autoencoder.codebook.embeddings.min())
        # print('max:',autoencoder.codebook.embeddings.max())
        # z = ((z - autoencoder.codebook.embeddings.min()) /
        #              (autoencoder.codebook.embeddings.max() -
        #               autoencoder.codebook.embeddings.min())).detach()
        z = ((z - autoencoder.codebook.embeddings.min()) /
                     (autoencoder.codebook.embeddings.max() -
                      autoencoder.codebook.embeddings.min())) * 2.0 - 1.0

        return z.permute(0,1,4,3,2)
    else:
        return autoencoder.encode_stage_2_inputs(inputs)
    
def main():
    parser = argparse.ArgumentParser(description="maisi.controlnet.training")
    parser.add_argument(
        "-e",
        "--environment-file",
        default="./configs/environment_maisi_seg_train.json",
        help="environment json file that stores environment path",
    )
    parser.add_argument(
        "-c",
        "--config-file",
        default="./configs/config_maisi.json",
        help="config json file that stores network hyper-parameters",
    )
    parser.add_argument(
        "-t",
        "--training-config",
        default="./configs/config_maisi_seg_train.json",
        help="config json file that stores training hyper-parameters",
    )
    parser.add_argument("-g", "--gpus", default=1, type=int, help="number of gpus per node")
    args = parser.parse_args()

    # Step 0: configuration
    logger = logging.getLogger("maisi.seg.training")
    # logger = setup_logging("training")
    # whether to use distributed data parallel
    use_ddp = args.gpus > 1
    if use_ddp:
        rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        # 【注意改回来！！！】
        device = setup_ddp(rank, world_size)
        logger.addFilter(RankFilter())
    else:
        rank = 0
        world_size = 1
        device = torch.device(f"cuda:{rank}")

    torch.cuda.set_device(device)
    if rank == 0:
        logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
        logger.info(f"World_size: {world_size}")

    with open(args.environment_file, "r") as env_file:
        env_dict = json.load(env_file)
    with open(args.config_file, "r") as config_file:
        config_dict = json.load(config_file)
    with open(args.training_config, "r") as training_config_file:
        training_config_dict = json.load(training_config_file)

    for k, v in env_dict.items():
        setattr(args, k, v)
    for k, v in config_dict.items():
        setattr(args, k, v)
    for k, v in training_config_dict.items():
        setattr(args, k, v)

    # initialize tensorboard writer
    if rank == 0:
        tensorboard_path = args.log_path
        # Path(tensorboard_path).mkdir(parents=True, exist_ok=True)
        tensorboard_writer = SummaryWriter(tensorboard_path)
        f_valid_txt = open(os.path.join(args.log_path, 'valid.txt'), 'a')

        # if not os.path.exists(args.cache_path):
        #     os.makedirs(args.cache_path)
        # if not os.path.exists(args.valid_path):
        #     os.makedirs(args.valid_path)
    # Step 1: set data loader
    if args.seg_train["AtoC"]:
        src_dir='./local_data/AC3_AC4/'
        src_dataset='AC4_inputs.h5'
        tgt_dir = './local_data/CREMI'
        tgt_dataset='cremiC_inputs_interp.h5'
    else:
        src_dir = './local_data/CREMI'
        src_dataset='cremiC_inputs_interp.h5'
        tgt_dir='./local_data/AC3_AC4/'
        tgt_dataset='AC4_inputs.h5'

    if args.seg_train["online_sampling"] or args.seg_train["no_sampling"]: # 【可能需要调整aug参数】
    # DACS: # MIC: tgt_aug=False   # 200
        dataset_train = CondTargetDataset_lb(src_dir=src_dir, src_dataset=src_dataset,\
                                            tgt_dir=tgt_dir, tgt_dataset=tgt_dataset, \
                                                crop_size=args.seg_train["crop_size"],length=50,if_xyz=False,strong_aug=True,src_aug=True)
        # dataset_train = CondTargetDataset_sampling(src_dir=src_dir, src_dataset=src_dataset,\
        #                                   tgt_dir=tgt_dir, tgt_dataset=tgt_dataset, \
        #                                     sampling_dir=args.seg_train["sampling_dir"], crop_size=args.seg_train["crop_size"],
        #                                     length=200,if_xyz=False,strong_aug=True,src_aug=True,pos_bias=True)
    else: # 【这里是把padding去掉了，直接crop一样的大小】【注意flow的生成有误，这里pos bias了，其他方法没有】
        dataset_train = CondTargetDataset_sampling(src_dir=src_dir, src_dataset=src_dataset,\
                                          tgt_dir=tgt_dir, tgt_dataset=tgt_dataset, \
                                            sampling_dir=args.seg_train["sampling_dir"], crop_size=args.seg_train["crop_size"],
                                            length=200,if_xyz=False,strong_aug=True,src_aug=True,pos_bias=True)
    train_loader = DataLoader(dataset_train, batch_size=args.seg_train["batch_size"], \
                              num_workers=4, shuffle=True, drop_last=True)
    valid_provider = ValidDataset_seg(root_dir=tgt_dir,train_dataset=tgt_dataset,
                                      crop_size=args.seg_train["crop_size"],if_train=False)
    # Step 2: define diffusion model and controlnet

    # if args.seg_train["online_sampling"]:
    #     # autoencoder
    #     if args.controlnet_train["use_vqgan"]:
    #         autoencoder = define_instance(args, "vqgan_def").to(device)
    #         autoencoder = autoencoder.load_from_checkpoint(args.trained_autoencoder_path,
    #                     embedding_dim=args.latent_channels,
    #                     n_codes=args.vqgan_def["n_codes"],
    #                     n_hiddens=args.vqgan_def["n_hiddens"],
    #                     downsample=args.vqgan_def["downsample"],
    #                     image_channels=args.image_channels).to(device)
    #     else:
    #         autoencoder = define_instance(args, "autoencoder_def").to(device)
    #         try:
    #             checkpoint_autoencoder = torch.load(args.trained_autoencoder_path, weights_only=True)
    #             new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint_autoencoder.items()}
    #             autoencoder.load_state_dict(new_state_dict)
    #             logger.info(f"Pretrained AE {args.trained_autoencoder_path} loaded.")
    #         except FileNotFoundError:
    #             logger.error(f"The file {args.trained_autoencoder_path} does not exist!")
    #         except RuntimeError as e:
    #             logger.error(f"Error loading the model state dict: {e}")
    #         except Exception as e:
    #             logger.error(f"An unexpected error occurred: {e}")


    #     # diffusion Model【注意加载的部分为什么有些没有完整加载？】
    #     unet = define_instance(args, "diffusion_unet_def").to(device)
    #     # load trained diffusion model
    #     if args.trained_diffusion_path is not None:
    #         if not os.path.exists(args.trained_diffusion_path):
    #             raise ValueError("Please download the trained diffusion unet checkpoint.")
    #         diffusion_model_ckpt = torch.load(args.trained_diffusion_path, map_location=device)
    #         unet.load_state_dict(diffusion_model_ckpt["unet_state_dict"])
    #         # load scale factor from diffusion model checkpoint
    #         scale_factor = diffusion_model_ckpt["scale_factor"]
    #         if rank == 0:
    #             logger.info(f"Load trained diffusion model from {args.trained_diffusion_path}.")
    #             logger.info(f"loaded scale_factor from diffusion model ckpt -> {scale_factor}.")
    #     else:
    #         scale_factor = 1.0
    #         if rank == 0:
    #             logger.info("trained diffusion model is not loaded.")
    #             logger.info(f"set scale_factor -> {scale_factor}.")

    #     # ControlNet
    #     controlnet = define_instance(args, "controlnet_def").to(device)
    #     # copy_model_state(controlnet, unet.state_dict())
    #     # "./models/maisi/model_maisi_controlnet-20datasets-e20wl100fold0bc_noi_dia_fsize_current_alternative.pt"
    #     if args.trained_controlnet_path is not None:
    #         if not os.path.exists(args.trained_controlnet_path):
    #             raise ValueError("Please download the trained ControlNet checkpoint.")
    #         controlnet_ckpt = torch.load(args.trained_controlnet_path, map_location=device)
    #         # copy_model_state(controlnet, controlnet_ckpt["controlnet_state_dict"])
    #         if rank == 0:
    #             logger.info(f"load trained controlnet model from {args.trained_controlnet_path}")
    #     else:
    #         if rank == 0:
    #             logger.info("train controlnet model from scratch.")

    #     # freeze the parameters 
    #     for p in unet.parameters():
    #         p.requires_grad = False
    #     for p in autoencoder.parameters(): 
    #         p.requires_grad = False

    #     # 【思考，感觉一边训controlnet一边训seg不太可行】
    #     if not args.seg_train["with_controlnet_train"]:
    #         for p in controlnet.parameters(): 
    #             p.requires_grad = False
    
    # superhuman
    # seg_model = define_instance(args, "seg_def").to(device)
    seg_model = UNet_PNI(in_planes=1,
                            out_planes=3,
                            filters=[28, 36, 48, 64, 80],
                            upsample_mode="bilinear",
                            decode_ratio=1,
                            merge_mode="add",
                            pad_mode="zero",
                            bn_mode="async",
                            relu_mode="elu",
                            init_mode="kaiming_normal").to(device)
    # load trained diffusion model
    if args.trained_segmentation_path is not None:
        if not os.path.exists(args.trained_segmentation_path):
            raise ValueError("Please download the trained superhuman checkpoint.")
        model_ckpt = torch.load(args.trained_segmentation_path, map_location=device)
        seg_model.load_state_dict(model_ckpt["seg_state_dict"])
        # 【注意seg_state_dict 和current_epoch】
        current_epoch = model_ckpt["epoch"]
        if rank == 0:
            logger.info(f"Load trained superhuman from {args.trained_segmentation_path}.")
    else:
        current_epoch=0
        if rank == 0:
            logger.info("trained superhuman is not loaded.")

    if args.seg_train["use_ema"]:
        print('using EMA teacher...')
        teacher = EMATeacher(device, args)
        teacher._init_ema_weights(seg_model)
    
    

    

    if use_ddp:
        if args.seg_train["with_controlnet_train"] and args.seg_train["online_sampling"]:
            controlnet = DDP(controlnet, device_ids=[device], output_device=rank, find_unused_parameters=True)
        seg_model = DDP(seg_model, device_ids=[device], output_device=rank,broadcast_buffers=False, find_unused_parameters=True)
        
    
    
    # Step 3: training config
        # 【可能要删,之后，loss】
    if args.seg_train["with_controlnet_train"]:
        weighted_loss = args.seg_train["weighted_loss"]
        weighted_loss_label = args.seg_train["weighted_loss_label"]
        optimizer = torch.optim.AdamW(params=controlnet.parameters(), lr=args.seg_train["lr"])
    
    optimizer = torch.optim.AdamW(seg_model.parameters(), lr=args.seg_train["lr"], betas=(0.9, 0.999),
                                     eps=1e-8,  # 1e-8为默认 # 1e-6 FP16
                                     weight_decay=0.01,  # 0.01为默认
                                     amsgrad=False # True
                                     )
    # total_steps = (args.seg_train["n_epochs"] * len(train_loader.dataset)) / args.seg_train["batch_size"]
    total_steps = ((args.seg_train["n_epochs"] - current_epoch) * len(train_loader.dataset)) / args.seg_train["batch_size"]
    logger.info(f"total number of training steps: {total_steps}.")

    # lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
    #         optimizer, 
    #         factor=1.0,  # 衰减系数=1 → 学习率始终为初始值
    #         total_iters=total_steps  # 预热步数=0，无预热
    #     )
    lr_scheduler = get_scheduler(optimizer, total_steps = total_steps, warmup_iters=10000, power=args.seg_train["power"], min_lr_ratio=0.0)
    # lr_scheduler = PolynomialLR(optimizer, total_iters=total_steps, power=args.seg_train["power"])

    

    # Step 4: training
    n_epochs = args.seg_train["n_epochs"]
    amp=True
    if amp:
        scaler = GradScaler("cuda")
    total_step = 0
    best_loss = 1e4

    # if weighted_loss > 1.0:
    #     logger.info(f"apply weighted loss = {weighted_loss} on labels: {weighted_loss_label}")

    # 【复杂，暂不考虑同时训练controlnet】
    seg_model.train()
    if args.seg_train["online_sampling"]:
        unet.eval()
        autoencoder.eval()
        controlnet.eval()

        # 设置推理生成参数
        output_size = args.controlnet_infer["autoencoder_sliding_window_infer_size"]
        num_downsample_level = max(
            1,
            (
                len(args.diffusion_unet_def["num_channels"])
                if isinstance(args.diffusion_unet_def["num_channels"], list)
                else len(args.diffusion_unet_def["attention_levels"])
            ),
        )
        divisor = 2 ** (num_downsample_level - 2)
        # logger.info(f"num_downsample_level -> {num_downsample_level}, divisor -> {divisor}.")
        # 4, 4
        top_region_index_tensor, bottom_region_index_tensor, spacing_tensor = None, None, None
        latent_shape = (args.latent_channels, output_size[0] // 4, output_size[1] // 4, output_size[2] // 4)

        use_vqgan = args.seg_train["use_vqgan"]

        noise_scheduler = define_instance(args, "noise_scheduler")

    # loss
    if args.seg_train["loss_func"] == 'MSELoss':
        criterion = MSELoss()
    elif args.seg_train["loss_func"] == 'BCELoss':
        criterion = BCELoss()
    elif args.seg_train["loss_func"] == 'WeightedBCELoss':
        criterion = WeightedBCE()
    elif args.seg_train["loss_func"] == 'WeightedMSELoss':
        criterion = WeightedMSE()
    else:
        raise AttributeError("NO this criterion")

    
    prev_time = time.time()
    # for epoch in range(n_epochs):
    # 【因为这里ddp代码会在epoch后同步，所以没有改成seg的iter的模式】
    for epoch in range(current_epoch, n_epochs):
        epoch_loss_ = 0
        for step, batch in enumerate(train_loader):
            src_img = batch["image"].to(device) # src img
            # 目前的是[inputs, conds]
            tgt_img = batch["cond"].to(device) # target img
            tgt_img_aug = batch["tgt"].to(device) # target img aug
            if args.seg_train.get("run_baseline", False) and args.seg_train["method"]=="mic":
                mask_aug = AddMask3D(mask_block_size=(2,10,10))
                tgt_img_aug = mask_aug(tgt_img_aug, mask_ratio=0.7)

            lbs = batch["lb"].to(device) # src lb
            weightmap = batch["weight"].to(device) # weightmap
            
            if args.seg_train["online_sampling"]:

                with torch.amp.autocast("cuda"):
                    inputs = apply_autoencoder(autoencoder, src_img.permute(0,1,4,3,2), use_vqgan=use_vqgan)
                    conds = apply_autoencoder(autoencoder, tgt_img.permute(0,1,4,3,2), use_vqgan=use_vqgan)
                    if not use_vqgan:
                        inputs = inputs * scale_factor
                    # inputs = autoencoder.encode_stage_2_inputs(src_img.permute(0,1,4,3,2))
                    # inputs = inputs* scale_factor
                    # conds = autoencoder.encode_stage_2_inputs(tgt_img.permute(0,1,4,3,2))

                # 260403
                # torch.Size([B, 1, 18, 160, 160])
                # 输出是[B, 3, 18, 160, 160]
                
                synthetic_images = ldm_conditional_sample_translated_image_lb(
                    autoencoder,
                    unet,
                    controlnet,
                    noise_scheduler,
                    scale_factor,
                    device,
                    conds, # 这是压缩后的tgt img
                    lbs.permute(0,1,4,3,2), # src lb 没有压缩
                    top_region_index_tensor,
                    bottom_region_index_tensor,
                    spacing_tensor,
                    latent_shape=latent_shape,
                    output_size=output_size,
                    noise_factor=1.0,
                    num_inference_steps=args.controlnet_infer["num_inference_steps"],
                    # reduce it when GPU memory is limited
                    autoencoder_sliding_window_infer_size=output_size,
                )
            elif not args.seg_train["no_sampling"]:
                synthetic_images = batch["sample"].to(device).permute(0,1,4,3,2)
                
            # if args.seg_train["special_no_maisi_lr"]:
            #     current_lr = args.seg_train["lr"]
            
            # synthetic_images = synthetic_images.permute(0,1,4,3,2).float()
            optimizer.zero_grad(set_to_none=True)

            # 调整权重，随epoch
            if args.seg_train["weight_rampup"] :
                rampup_epoch=args.seg_train["rampup_epochs"]
                weight_syn = get_current_consistency_weight(epoch, consistency=args.seg_train["weight_syn"], consistency_rampup=rampup_epoch)
                weight_tgt = get_current_consistency_weight(epoch, consistency=args.seg_train["weight_tgt"], consistency_rampup=rampup_epoch)
                weight_fa = get_current_consistency_weight(epoch, consistency=args.seg_train["weight_fa"], consistency_rampup=rampup_epoch)
                # weight_proto_pred = get_current_consistency_weight(epoch, consistency=args.seg_train["weight_proto_pred"], consistency_rampup=rampup_epoch)
                weight_consist_fea = get_current_consistency_weight(epoch, consistency=args.seg_train["weight_consist_fea"], consistency_rampup=rampup_epoch)
            else:
                weight_syn = args.seg_train["weight_syn"]
                weight_tgt = args.seg_train["weight_tgt"]
                weight_fa = args.seg_train["weight_fa"]
                # weight_proto_pred = args.seg_train["weight_proto_pred"]
                weight_consist_fea = args.seg_train["weight_consist_fea"]
            # src  torch.Size([1, 1, 32, 160, 160]) torch.float32
            # synthetic_images: torch.Size([1, 1, 160, 160, 32]) float16...
            
            
            with autocast("cuda", enabled=amp): # 混合精度, 有时把float16转成float32
                if args.seg_train["feature_align"]:
                    src_pred, src_feat = seg_model(src_img, show_feature=True)
                else:
                    src_pred = seg_model(src_img)

                # print('src pred:', src_pred.shape)

                if not args.seg_train["no_sampling"]:
                    if args.seg_train["feature_align"]:
                        syn_pred, syn_feat = seg_model(synthetic_images.permute(0,1,4,3,2),show_feature=True)
                    else:
                        syn_pred = seg_model(synthetic_images.permute(0,1,4,3,2))
                if weight_tgt > 0:
                    if args.seg_train["feature_align"]:
                        tgt_pred_aug, tgt_aug_feat = seg_model(tgt_img_aug, show_feature=True)
                    else:
                        tgt_pred_aug= seg_model(tgt_img_aug)

                with torch.no_grad():   # 
                    if args.seg_train["use_ema"]: 
                        # pred是伪标签，会根据是否auto_mask_tgt决定自动计算还是用mask loss
                        tgt_pred, pseudo_weight = teacher(tgt_img,if_thres=True,show_feature=False)
                    else:
                        if weight_tgt > 0:
                            if args.seg_train["feature_align"]:
                                tgt_pred,tgt_feat = seg_model(tgt_img,show_feature=True)
                            else:
                                tgt_pred= seg_model(tgt_img)
                                tgt_pred = tgt_pred.detach()
                            # tgt_pred = torch.clamp(tgt_pred, min=0.000001, max=0.999999)

                            # # tgt_pred_thres = tgt_pred.clone()
                            # tgt_pred[tgt_pred<=0.5] = 0
                            # tgt_pred[tgt_pred>0.5] = 1
                            # if args.seg_train["mask_loss"] > 1:
                            #     pseudo_weight = tgt_pred.clone().detach()
                            #     pseudo_weight[pseudo_weight==0] = args.seg_train["mask_loss"]
                                if args.seg_train["mask_loss"] > 1: # 注意是相反的，0的才是真正的边界，1是全白背景
                                    pseudo_weight = torch.clamp(tgt_pred, min=0.000001, max=0.999999)
                                    pseudo_weight[pseudo_weight <= 0.5] = args.seg_train["mask_loss"]
                                    pseudo_weight[pseudo_weight > 0.5] = 1

            
            # torch.Size([1, 1, 32, 160, 160])
            # 1,3,4,at version 4; expected version 3
            # 3,4 at version 3; expected version 2 .分不分开都一样
            # 1,2 at version 3; expected version 2 ,分开。loss没写也一样
            # 1,src_loss,2 也是一样
            # 1,2,3,4 去掉 with   是version 5; expected version 2 instead.
                    # version 5; expected version 4
                # 原因是nn.parallel.DistributedDataParallel 
                    # broadcast_buffers=True导致参数被覆盖修改
                    # broadcast_buffers=False
            ##############################
            # source loss
            src_loss = criterion(src_pred.float(), lbs.float(), weightmap.float())
            
            # print('src_loss:',src_loss)
            if not args.seg_train["no_sampling"]:
                src_syn_loss = criterion(syn_pred.float(), lbs.float(), weightmap.float())* weight_syn
            else:
                src_syn_loss = torch.tensor(0.0).to(device)
            loss = src_loss + src_syn_loss
            # loss = src_loss
            # loss = src_syn_loss

            # target loss # ,weightmap_tgt.float()
            if args.seg_train.get("run_baseline", False) and args.seg_train["method"]=="dacs":
                M = (lbs > 0).any(dim=1, keepdim=True).float()  # [B, 1, Z, H, W]

                x_mix = M * src_img + (1 - M) * tgt_img_aug
                y_mix = M * lbs + (1 - M) * tgt_pred

                weight_mix = M + (1 - M) * pseudo_weight
                mix_pred = seg_model(x_mix)
                mix_loss = args.seg_train.get("weight_mix", 1.0) * (F.mse_loss(mix_pred, y_mix, reduction='none') * weight_mix).mean()
                # print('mix_loss:',mix_loss)
                loss += mix_loss
                tgt_loss = torch.tensor(0.0).to(device)

            elif weight_tgt > 0:
                if args.seg_train["mask_loss"] > 1:
                    tgt_loss = criterion(tgt_pred_aug.float(), tgt_pred.float(),pseudo_weight.float())* weight_tgt
                else:
                    tgt_loss = criterion(tgt_pred_aug.float(), tgt_pred.float())* weight_tgt
            else:
                tgt_loss = torch.tensor(0.0).to(device)
            loss += tgt_loss

            # 【rampup前面0.0073,0.0082】
            # print('weight_tgt:', weight_tgt) # weight_syn: 0.0006737946999085467
            # print('src_loss:', src_loss) # 1.5718
            # print('src_syn_loss:', src_syn_loss) # 0.0011
            # print('tgt_loss:', tgt_loss) # 这个总是0, inf????
#             src_loss: tensor(1.5492, device='cuda:0', grad_fn=<DivBackward0>)
# src_syn_loss: tensor(0.0011, device='cuda:0', grad_fn=<MulBackward0>)
# tgt_loss: tensor(0., device='cuda:0', grad_fn=<MulBackward0>)
            with torch.amp.autocast("cuda"):
                if args.seg_train["feature_align"]:
                    loss_var, loss_dist, loss_aglin, _ = feature_alignment(src_feat, tgt_aug_feat, lbs, threshold=0.5)
                    loss_fa = (loss_var + loss_dist + loss_aglin) * weight_fa
                    
                    if not args.seg_train["no_sampling"]:
                        loss_var, loss_dist, loss_aglin, _ = feature_alignment(syn_feat, tgt_aug_feat, lbs, threshold=0.5)
                        loss_fa += (loss_var + loss_dist + loss_aglin) * weight_fa * weight_syn
                        loss_fa/= (1.0 + weight_syn)
                    '''
                    loss_var: 0.4742598533630371
                    loss_dist: 0.49135932326316833
                    loss_aglin: 0.6683384776115417
                    
                    '''
                    loss += loss_fa
                else:
                    loss_fa= torch.tensor(0.0).to(device)
            
            if args.seg_train["consist_fea_loss"]: # 最好不要用，实际loss很大很大
                loss_consist_fea = criterion(tgt_aug_feat.float(), tgt_feat.float())* weight_consist_fea
                loss += loss_consist_fea
            else:
                loss_consist_fea = torch.tensor(0.0).to(device)


            if amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            lr_scheduler.step()
            total_step += 1

            if args.seg_train["use_ema"]:
                teacher.update_weights(seg_model, total_step)

            if rank == 0:
                # write train loss for each batch into tensorboard
                tensorboard_writer.add_scalar(
                    "train/train_loss_iter", loss.detach().cpu().item(), total_step
                )
                tensorboard_writer.add_scalar(
                    "train/train_src_loss_iter", src_loss.detach().cpu().item(), total_step
                )
                tensorboard_writer.add_scalar(
                    "train/train_src_syn_loss_iter", src_syn_loss.detach().cpu().item(), total_step
                )
                tensorboard_writer.add_scalar(
                    "train/train_tgt_loss_iter", tgt_loss.detach().cpu().item(), total_step
                )
                tensorboard_writer.add_scalar(
                    "train/feat_align_loss_iter", loss_fa.detach().cpu().item(), total_step
                )
                tensorboard_writer.add_scalar(
                    "train/tgt_feat_loss_iter", loss_consist_fea.detach().cpu().item(), total_step
                )
                batches_done = step + 1
                batches_left = len(train_loader) - batches_done
                time_left = timedelta(seconds=batches_left * (time.time() - prev_time))
                prev_time = time.time()
                logger.info(
                    "\r[Epoch %d/%d] [Batch %d/%d] [LR: %.8f] [loss: %.4f] ETA: %s "
                    % (
                        epoch + 1,
                        n_epochs,
                        step + 1,
                        len(train_loader),
                        # current_lr,
                        lr_scheduler.get_last_lr()[0],
                        loss.detach().cpu().item(),
                        time_left,
                    )
                )
            epoch_loss_ += loss.detach()

        epoch_loss = epoch_loss_ / (step + 1)
  

        if use_ddp:
            dist.barrier()
            dist.all_reduce(epoch_loss, op=torch.distributed.ReduceOp.AVG)
        
        val_interval = args.seg_train["val_interval"]
        visual_all_interval = args.seg_train["visual_all_interval"]
        # if epoch>=args.seg_train["rampup_epochs"]:
        #     val_interval /=2
        # if epoch >= 75-1:
        #     val_interval = 2
        # elif epoch >= 50-1:
        #     val_interval = 5

        if rank == 0:
            tensorboard_writer.add_scalar("train/train_loss_epoch", epoch_loss.cpu().item(), total_step)
            # save only on master GPU (rank 0)
            if args.seg_train["use_ema"]:
                model_state_dict = teacher.get_ema_model().state_dict()
            else:
                model_state_dict = seg_model.module.state_dict() if world_size > 1 else seg_model.state_dict()
            torch.save(
                {
                    "epoch": epoch + 1,
                    "loss": epoch_loss,
                    "seg_state_dict": model_state_dict,
                },
                f"{args.model_dir}/{args.exp_name}_current.pt",
            )


            


            if epoch_loss < best_loss:
                best_loss = epoch_loss
                logger.info(f"best loss -> {best_loss}.")
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "loss": epoch_loss,
                        "seg_state_dict": model_state_dict,
                    },
                    f"{args.model_dir}/{args.exp_name}_epoch{epoch + 1:04d}_loss{epoch_loss:.4f}.pt",
                )
            

            if not args.seg_train["no_sampling"] and weight_tgt>0 and (epoch == 0 or (epoch+1) % visual_all_interval == 0):
                display_saver(tensorboard_writer,epoch+1,
                  src_img,src_pred,lbs,
                  synthetic_images.permute(0,1,4,3,2),syn_pred,
                  tgt_img,tgt_pred,
                  tgt_img_aug,tgt_pred_aug)
                tensorboard_writer.flush()



            if (epoch+1) % val_interval == 0 : # or epoch == 0
                # 保存训练中间结果？算了太多
                # src_img, tgt_img, synthetic_images
                # pred_src, pred_tgt, pred_syn

                show_affs(epoch+1, src_img, src_pred[:, :3], lbs[:, :3], args.cache_path, model_type='superhuman')
                
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "loss": epoch_loss,
                        "seg_state_dict": model_state_dict,
                    },
                    f"{args.model_dir}/{args.exp_name}_epoch{epoch + 1:04d}_loss{epoch_loss:.4f}.pt",
                )


                # output_path = args.output_dir
                # # output_path=os.path.join(args.output_dir,'val/')
                # if not os.path.exists(output_path):
                #     os.makedirs(output_path)
                # visual_saver_with_cond(epoch+1, synthetic_images,lbs,src_img,tgt_img,valid_path, logger, tensorboard_writer)
                
                seg_model.eval()#验证
                dataloader = torch.utils.data.DataLoader(valid_provider, batch_size=1, num_workers=0,
                                                shuffle=False, drop_last=False, pin_memory=True) # data.provider_valid，验证，f1分数
                losses_valid = []
                for batch in dataloader:
                    inputs, target, weightmap = batch# dataloader[k]
                    inputs = inputs.cuda()
                    target = target.cuda()
                    weightmap = weightmap.cuda()
                    with torch.no_grad():
                        if args.seg_train["use_ema"]:
                            pred = teacher(inputs)
                        else:
                            pred = seg_model(inputs)
                    tmp_loss = criterion(pred, target, weightmap)
                    losses_valid.append(tmp_loss.item())
                    valid_provider.add_vol(np.squeeze(pred.data.cpu().numpy()))
                epoch_loss = sum(losses_valid) / len(losses_valid)
                out_affs = valid_provider.get_results() # (3, 50, 1024, 1024)
                gt_affs = valid_provider.get_gt_affs().copy() # gt_affs[0]
                gt_seg = valid_provider.get_gt_lb() # labels_origin[0]
                valid_provider.reset_output() # 重置
                out_affs = out_affs[:3]#?
                # gt_affs = gt_affs[:, :3]
                show_affs_whole(epoch+1, out_affs, gt_affs, args.valid_path)

                ##############
                # segmentation
                
                fragments = watershed(out_affs, 'maxima_distance')#分水岭产生fragments
                sf = 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256>>'
                seg_waterz = list(waterz.agglomerate(out_affs, [0.50], # 阈值
                                                    fragments=fragments,
                                                    scoring_function=sf,#aff_threshold_low,aff_threshold_high??
                                                    discretize_queue=256))[0]#离散化队列
                # 都是skimage.metrics调用的，要先gt。
                arand_waterz = adapted_rand_ref(gt_seg, seg_waterz, ignore_labels=(0))[0]#adapted_rand_error兰德误差1-2pr/(p+r)
                voi_split, voi_merge = voi_ref(gt_seg, seg_waterz, ignore_labels=(0))#voi信息差异指标，距离度量H(1|0),H(0|1)
                voi_sum_waterz = voi_split + voi_merge

                # baseline,utils/lmc.py
                # seg_lmc = mc_baseline(out_affs)#产生分割
                # arand_lmc = adapted_rand_ref(gt_seg, seg_lmc, ignore_labels=(0))[0]
                # voi_split, voi_merge = voi_ref(gt_seg, seg_lmc, ignore_labels=(0))
                # voi_sum_lmc = voi_split + voi_merge

                ##############

                # MSE均方误差
                whole_mse = np.sum(np.square(out_affs - gt_affs)) / np.size(gt_affs)
                out_affs = np.clip(out_affs, 0.000001, 0.999999)# 限定最大值和最小值，否则下面log会不行。numpy.clip(a, a_min, a_max, out=None)
                bce = -(gt_affs * np.log(out_affs) + (1 - gt_affs) * np.log(1 - out_affs))#二分类交叉熵
                whole_bce = np.sum(bce) / np.size(gt_affs)
                out_affs[out_affs <= 0.5] = 0
                out_affs[out_affs > 0.5] = 1
                # whole_f1 = 1 - f1_score(gt_affs.astype(np.uint8).flatten(), out_affs.astype(np.uint8).flatten())一个意思,为什么是1-
                whole_f1 = f1_score(1 - gt_affs.astype(np.uint8).flatten(), 1 - out_affs.astype(np.uint8).flatten())
                # print('epoch-%d, valid-loss=%.6f, MSE-loss=%.6f, BCE-loss=%.6f, F1-score=%.6f, VOI-waterz=%.6f, ARAND-waterz=%.6f, VOI-lmc=%.6f, ARAND-lmc=%.6f' % \
                #     (epoch+1, epoch_loss, whole_mse, whole_bce, whole_f1, voi_sum_waterz, arand_waterz, voi_sum_lmc, arand_lmc), flush=True)
                print('epoch-%d, valid-loss=%.6f, MSE-loss=%.6f, BCE-loss=%.6f, F1-score=%.6f, VOI-split=%.6f, VOI-merge=%.6f, VOI-waterz=%.6f, ARAND-waterz=%.6f' % \
                    (epoch+1, epoch_loss, whole_mse, whole_bce, whole_f1, voi_split, voi_merge, voi_sum_waterz, arand_waterz), flush=True)
                tensorboard_writer.add_scalar('valid/epoch_loss', epoch_loss, total_step)#所需要的数据保存在文件里面供可视化使用
                tensorboard_writer.add_scalar('valid/mse_loss', whole_mse, total_step)
                tensorboard_writer.add_scalar('valid/bce_loss', whole_bce, total_step)
                tensorboard_writer.add_scalar('valid/f1_score', whole_f1, total_step)
                tensorboard_writer.add_scalar('valid/voi_waterz', voi_sum_waterz, total_step)
                tensorboard_writer.add_scalar('valid/arand_waterz', arand_waterz, total_step)
                # tensorboard_writer.add_scalar('valid/voi_lmc', voi_sum_lmc, total_step)
                # tensorboard_writer.add_scalar('valid/arand_lmc', arand_lmc, total_step)
                # f_valid_txt.write('epoch-%d, valid-loss=%.6f, MSE-loss=%.6f, BCE-loss=%.6f, F1-score=%.6f, VOI-waterz=%.6f, ARAND-waterz=%.6f, VOI-lmc=%.6f, ARAND-lmc=%.6f' % \
                #                 (epoch+1, epoch_loss, whole_mse, whole_bce, whole_f1, voi_sum_waterz, arand_waterz, voi_sum_lmc, arand_lmc))
                f_valid_txt.write('epoch-%d, valid-loss=%.6f, MSE-loss=%.6f, BCE-loss=%.6f, F1-score=%.6f, VOI-split=%.6f, VOI-merge=%.6f, VOI-waterz=%.6f, ARAND-waterz=%.6f' % \
                                (epoch+1, epoch_loss, whole_mse, whole_bce, whole_f1, voi_split, voi_merge, voi_sum_waterz, arand_waterz))
                f_valid_txt.write('\n')
                f_valid_txt.flush()
                torch.cuda.empty_cache() # 释放缓存
        
        torch.cuda.empty_cache()
    if rank == 0:
        f_valid_txt.close()
    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
