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
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from .diff_model_setting import initialize_distributed, load_config, setup_logging
from .utils import binarize_labels, define_instance, prepare_maisi_controlnet_json_dataloader, setup_ddp
from .sample import ldm_conditional_sample_translated_image, ldm_conditional_sample_translated_image_lb,ldm_conditional_unsb

from monai.data import DataLoader
from data.dataset_lb import CondTargetDataset_lb
from scripts.diff_model_infer_self import visual_saver, visual_saver_with_cond
from networks_copy.unsb_model import PatchSampleF_3d
from networks_copy import networks
from networks_copy.patchnce import PatchNCELoss_v2

def patch_instance_norm_state_dict(state_dict, module, keys, i=0):
    """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
    key = keys[i]
    if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
        if module.__class__.__name__.startswith('InstanceNorm') and \
                (key == 'running_mean' or key == 'running_var'):
            if getattr(module, key) is None:
                state_dict.pop('.'.join(keys))
        if module.__class__.__name__.startswith('InstanceNorm') and \
            (key == 'num_batches_tracked'):
            state_dict.pop('.'.join(keys))
    else:
        patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

def apply_autoencoder(
        autoencoder,
        inputs, # [b, 1, 160, 160, z]  -> [b, 4, 40, 40, z]
        use_vqgan=False):
    if use_vqgan:
        z = autoencoder.encode(inputs.permute(0,1,4,3,2),
                                               quantize=False, include_embeddings=True)

        z = ((z - autoencoder.codebook.embeddings.min()) /
                     (autoencoder.codebook.embeddings.max() -
                      autoencoder.codebook.embeddings.min())) * 2.0 - 1.0

        return z.permute(0,1,4,3,2)
    else:
        return autoencoder.encode_stage_2_inputs(inputs)
    
def visual_debug(data, img_path, img_name):
    # data: [40, 40, 18]
    from PIL import Image
    import numpy as np

    data = data.cpu().detach().numpy()
    data = data * 255.0
    np.clip(data, 0, 255)
    data=data.astype(np.uint8) 

    data = np.transpose(data, (2, 1, 0)) 

    Image.fromarray(data[0]).save(os.path.join(img_path, img_name))

def main():
    parser = argparse.ArgumentParser(description="maisi.controlnet.training")
    parser.add_argument(
        "-e",
        "--environment-file",
        default="./configs/environment_maisi_controlnet_train.json",
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
        default="./configs/config_maisi_controlnet_train.json",
        help="config json file that stores training hyper-parameters",
    )
    parser.add_argument("-g", "--gpus", default=1, type=int, help="number of gpus per node")
    args = parser.parse_args()

    # Step 0: configuration
    logger = logging.getLogger("maisi.controlnet.training")
    # logger = setup_logging("training")
    # whether to use distributed data parallel
    use_ddp = args.gpus > 1
    if use_ddp:
        rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
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
        tensorboard_path = os.path.join(args.tfevent_path, args.exp_name)
        Path(tensorboard_path).mkdir(parents=True, exist_ok=True)
        tensorboard_writer = SummaryWriter(tensorboard_path)

    # Step 1: set data loader
    # train_loader, _ = prepare_maisi_controlnet_json_dataloader(
    #     json_data_list=args.json_data_list,
    #     data_base_dir=args.data_base_dir,
    #     rank=rank,
    #     world_size=world_size,
    #     batch_size=args.controlnet_train["batch_size"],
    #     cache_rate=args.controlnet_train["cache_rate"],
    #     fold=args.controlnet_train["fold"],
    # )
    dataset_train = CondTargetDataset_lb(src_dir='./local_data/CREMI', src_dataset='cremiC_inputs_interp.h5',\
                                          tgt_dir='./local_data/AC3_AC4/', tgt_dataset='AC4_inputs.h5', length=100,src_aug=False,strong_aug=False)
    train_loader = DataLoader(dataset_train, batch_size=args.controlnet_train["batch_size"], \
                              num_workers=4, shuffle=True, drop_last=True)

    # Step 2: define diffusion model and controlnet

    ### 【加载autoencoder】
    if args.controlnet_train["use_vqgan"]:
        autoencoder = define_instance(args, "vqgan_def").to(device)
        autoencoder = autoencoder.load_from_checkpoint(args.trained_autoencoder_path,
                    embedding_dim=args.latent_channels,
                    n_codes=args.vqgan_def["n_codes"],
                    n_hiddens=args.vqgan_def["n_hiddens"],
                    downsample=args.vqgan_def["downsample"],
                    image_channels=args.image_channels).to(device)
    else:
        autoencoder = define_instance(args, "autoencoder_def").to(device)

        try:
            checkpoint_autoencoder = torch.load(args.trained_autoencoder_path, weights_only=True)
            new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint_autoencoder.items()}
            autoencoder.load_state_dict(new_state_dict)
            logger.info(f"Pretrained AE {args.trained_autoencoder_path} loaded.")
        except FileNotFoundError:
            logger.error(f"The file {args.trained_autoencoder_path} does not exist!")
        except RuntimeError as e:
            logger.error(f"Error loading the model state dict: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")




    # define diffusion Model
    # load trained diffusion model
    if args.controlnet_train["use_unsb"]:
        unet = define_instance(args, "netG_def").to(device)
        if args.trained_unsb_path is not None:
            if not os.path.exists(args.trained_unsb_path):
                raise ValueError("Please download the trained UNSB checkpoint.")
            unet.load_state_dict(
                torch.load(args.trained_unsb_path, map_location=device)
            )
            if rank == 0:
                logger.info(f"Load trained UNSB from {args.trained_unsb_path}.")
        scale_factor = 1.0

        netD = define_instance(args, "netD_def").to(device)
        netD.load_state_dict(torch.load(args.trained_netD_path, map_location=device))

        netE = define_instance(args, "netE_def").to(device)
        netE.load_state_dict(torch.load(args.trained_netE_path, map_location=device))

        netF = PatchSampleF_3d(use_mlp=True, init_type='xavier', init_gain=0.02, device=device, nc=256,resume=True).to(device)
        state_dict = torch.load(args.trained_netF_path, map_location=device)
        for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
            patch_instance_norm_state_dict(state_dict, netF, key.split('.'))
        netF.load_state_dict(state_dict)

    else:
        unet = define_instance(args, "diffusion_unet_def").to(device)
        if args.trained_diffusion_path is not None:
            if not os.path.exists(args.trained_diffusion_path):
                raise ValueError("Please download the trained diffusion unet checkpoint.")
            diffusion_model_ckpt = torch.load(args.trained_diffusion_path, map_location=device)
            unet.load_state_dict(diffusion_model_ckpt["unet_state_dict"])
            # load scale factor from diffusion model checkpoint
            scale_factor = diffusion_model_ckpt["scale_factor"]
            if rank == 0:
                logger.info(f"Load trained diffusion model from {args.trained_diffusion_path}.")
                logger.info(f"loaded scale_factor from diffusion model ckpt -> {scale_factor}.")
        else:
            scale_factor = 1.0
            if rank == 0:
                logger.info("trained diffusion model is not loaded.")
                logger.info(f"set scale_factor -> {scale_factor}.")

    # define ControlNet
    controlnet = define_instance(args, "controlnet_def").to(device)
    
    # load trained controlnet model if it is provided
    # "./models/maisi/model_maisi_controlnet-20datasets-e20wl100fold0bc_noi_dia_fsize_current_alternative.pt"
    if args.trained_controlnet_path is not None:
        if not os.path.exists(args.trained_controlnet_path):
            raise ValueError("Please download the trained ControlNet checkpoint.")
        controlnet_ckpt = torch.load(args.trained_controlnet_path, map_location=device)
        copy_model_state(controlnet, controlnet_ckpt["controlnet_state_dict"])
        current_epoch = 0 #controlnet_ckpt["epoch"] 【记得改】
        # controlnet.load_state_dict(
        #     torch.load(args.trained_controlnet_path, map_location=device)["controlnet_state_dict"]
        # )
        if rank == 0:
            logger.info(f"load trained controlnet model from {args.trained_controlnet_path}")
    else: # copy weights from the DM to the controlnet
        current_epoch=0
        copy_model_state(controlnet, unet.state_dict())
        if rank == 0:
            logger.info("train controlnet model from scratch.")
    
    # we freeze the parameters of the diffusion model.
    for p in unet.parameters():
        p.requires_grad = False
    for p in autoencoder.parameters(): # 【不确定】
        p.requires_grad = False
    

    noise_scheduler = define_instance(args, "noise_scheduler")

    if use_ddp:
        controlnet = DDP(controlnet, device_ids=[device], output_device=rank, find_unused_parameters=True)

    # Step 3: training config
    weighted_loss = args.controlnet_train["weighted_loss"]
    weighted_loss_label = args.controlnet_train["weighted_loss_label"]
    optimizer = torch.optim.AdamW(params=controlnet.parameters(), lr=args.controlnet_train["lr"])
    total_steps = (args.controlnet_train["n_epochs"] * len(train_loader.dataset)) / args.controlnet_train["batch_size"]
    logger.info(f"total number of training steps: {total_steps}.")

    lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=total_steps, power=2.0)

    # Step 4: training
    n_epochs = args.controlnet_train["n_epochs"]
    scaler = GradScaler("cuda")
    total_step = 0
    best_loss = 1e4
    use_vqgan = args.controlnet_train["use_vqgan"]

    if weighted_loss > 1.0:
        logger.info(f"apply weighted loss = {weighted_loss} on labels: {weighted_loss_label}")

    controlnet.train()
    unet.eval()
    autoencoder.eval()
    netD.eval()
    netE.eval()
    netF.eval()

    tau = args.unsb_noise_scheduler["tau"]
    T = args.unsb_noise_scheduler["num_train_timesteps"] # 5
    incs = np.array([0] + [1/(i+1) for i in range(T-1)])
    times = np.cumsum(incs)
    times = times / times[-1]
    times = 0.5 * times[-1] + 0.5 * times
    times = np.concatenate([np.zeros(1),times])
    times = torch.tensor(times).float().cuda()

    criterionGAN = networks.GANLoss(args.gan_mode).to(device)
    criterionNCE = []
    nce_layers = [0, 0, 3, 6, 9, 11, 12, 13]
    for _ in nce_layers:
        criterionNCE.append(PatchNCELoss_v2(args.controlnet_train["batch_size"],args.nce_T).to(device))


    prev_time = time.time()
    for epoch in range(n_epochs):
        epoch += current_epoch
        epoch_loss_ = 0
        for step, batch in enumerate(train_loader):
            # get image embedding and label mask and scale image embedding by the provided scale_factor
            src_img = batch["image"].to(device) # src img
            # 目前的是[inputs, conds]
            tgt_img = batch["cond"].to(device) # target img
            lbs = batch["lb"].to(device) # src lb
            # weightmap = batch["weight"].to(device) # src weight_map
            with torch.amp.autocast("cuda"):
                inputs = apply_autoencoder(autoencoder, src_img, use_vqgan=use_vqgan)
                conds = apply_autoencoder(autoencoder, tgt_img, use_vqgan=use_vqgan)
                if not use_vqgan:
                    inputs = inputs * scale_factor
                    # conds = conds * scale_factor_cond

            bs = inputs.shape[0]//2 # UNSB # 会同时两张图片的结果
            optimizer.zero_grad(set_to_none=True)
            timesteps = (torch.randint(T, size=[1]).cuda() * torch.ones(size=[1]).cuda()).long()
            with torch.no_grad():
                with autocast("cuda", enabled=True):
                    # generate random noise
                    for t in range(timesteps.int().item()+1): # 从 0 到 time_idx，一步步加噪
                    
                        if t > 0:
                            delta = times[t] - times[t-1] # 1/T
                            denom = times[-1] - times[t-1] # 1-(t-1)/T
                            inter = (delta / denom).reshape(-1,1,1,1) # 1/(T-t+1)
                            scale = (delta * (1 - delta / denom)).reshape(-1,1,1,1) # (T-t)/[(T-t+1)*T]

                        Xt       = inputs[:bs] if (t == 0) else (1-inter) * Xt + inter * Xt_1.detach() \
                            + (scale * tau).sqrt() * torch.randn_like(Xt).to(inputs.device)
                        # time_idx: (1, 1)
                        time_idx = (t * torch.ones(size=[bs]).to(inputs.device)).long()
                        
                        z        = torch.randn(size=[bs,4*64]).to(inputs.device)
                    
                        down_block_res_samples, mid_block_res_sample = controlnet(
                            x=Xt, timesteps=time_idx, controlnet_cond=lbs[:bs]
                        )
                        Xt_1 = unet(
                            x=Xt,
                            timesteps=time_idx,
                            context=z,
                            down_block_additional_residuals=down_block_res_samples,
                            mid_block_additional_residual=mid_block_res_sample,
                        )

                        Xt_2       = inputs[bs:] if (t == 0) else (1-inter) * Xt_2 + inter * Xt_12.detach() \
                            + (scale * tau).sqrt() * torch.randn_like(Xt).to(inputs.device)
                        # time_idx: (1, 1)
                        time_idx = (t * torch.ones(size=[bs]).to(inputs.device)).long()

                        z        = torch.randn(size=[bs,4*64]).to(inputs.device)
                    
                        down_block_res_samples, mid_block_res_sample = controlnet(
                            x=Xt_2, timesteps=time_idx, controlnet_cond=lbs[bs:]
                        )
                        Xt_12 = unet(
                            x=Xt_2,
                            timesteps=time_idx,
                            context=z,
                            down_block_additional_residuals=down_block_res_samples,
                            mid_block_additional_residual=mid_block_res_sample,
                        )

                        XtB = conds[:bs] if (t == 0) else (1-inter) * XtB + inter * Xt_1B.detach() + (scale * tau).sqrt() * torch.randn_like(XtB).to(conds.device)
                        time_idx = (t * torch.ones(size=[bs]).to(conds.device)).long()
                        z        = torch.randn(size=[bs,4*64]).to(conds.device)
                        # with torch.amp.autocast("cuda"):
                        Xt_1B = unet(XtB, time_idx, context=z)
            real_At = Xt.detach()
            real_At2 = Xt_2.detach()
            XtB = XtB.detach()
            with autocast("cuda", enabled=True):
                z_in    = torch.randn(size=[bs,4*64]).to(inputs.device)
                down_block_res_samples, mid_block_res_sample = controlnet(
                    x=real_At, timesteps=timesteps, controlnet_cond=lbs[:bs]
                )
                fake_B = unet(
                    x=real_At,
                    timesteps=timesteps,
                    context=z_in,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                )
                idt_B = unet(
                    x=XtB,
                    timesteps=timesteps,
                    context=z_in,
                )

                # [1, 4, 40, 40, 18]
                z_in    = torch.randn(size=[bs,4*64]).to(inputs.device)
                down_block_res_samples, mid_block_res_sample = controlnet(
                    x=real_At2, timesteps=timesteps, controlnet_cond=lbs[bs:]
                )
                fake_B2 = unet(
                    x=real_At2,
                    timesteps=timesteps,
                    context=z_in,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                )
            

            if weighted_loss > 1.0:
                # weights: [b, 4, 40, 40, 8]
                weights = torch.ones_like(real_At).to(inputs.device)
                # roi: torch.Size([b, 1, 40, 40, 8])
                roi = torch.zeros([real_At.shape[0]] + [1] + list(real_At.shape[2:])).to(real_At.device)
                # 将label: [b, 3, 160, 160, 32]下采样到和inputs一样的size [b, 3, 40, 40, 8]
                interpolate_label = F.interpolate(lbs[:bs], size=list(real_At.shape[2:]), mode="nearest")
                for label in weighted_loss_label: # label = 0
                    # roi[interpolate_label == label] = 1
                    roi[(interpolate_label == label).any(dim=1, keepdim=True)] = 1
                weights[roi.repeat(1, real_At.shape[1], 1, 1, 1) == 1] = weighted_loss
                '''
                # 将weightmap: [b, 3, 160, 160, z] -> [b, 3, 40, 40, z]
                weightmap_d = F.interpolate(weightmap, size=inputs.shape[2:], mode="nearest").to(inputs.device)
                # weights: [b, 4, 40, 40, z]
                weights = torch.cat([weightmap_d, weightmap_d[:, :1]], dim=1).to(inputs.device)
                '''
            else:
                weights = torch.ones_like(real_At).to(real_At.device)
            

            def calculate_NCE_loss(src,tgt):
                bs = src.size(0)
                z    = torch.randn(size=[2*bs,4*64]).to(inputs.device)
                '''
                src: torch.float16
                tgt: torch.float32
                '''

                input = torch.cat([tgt, src], dim=0)

                feat = unet(input, timesteps=timesteps*0, context=z, layers=nce_layers, encode_only=True)
                feat_q, feat_k = [], []
                for feat_i in feat:
                    feat_q.append(feat_i[:bs])
                    feat_k.append(feat_i[bs:])

                
                # self.opt.num_patches=256
                feat_k_pool, sample_ids = netF(feat_k, 256, None)
                feat_q_pool, _ = netF(feat_q, 256, sample_ids)

                total_nce_loss = 0.0
                for f_q, f_k, crit, _ in zip(feat_q_pool, feat_k_pool, criterionNCE, nce_layers):
                    # [256]
                    total_nce_loss += crit(f_q, f_k).mean()

                return total_nce_loss / len(nce_layers)
            
            with autocast("cuda", enabled=True):
                pred_fake = netD(fake_B,timesteps)
                # [1, 4, 3, 3, 18].mean
                loss = criterionGAN(pred_fake, True).mean() * args.lambda_GAN

                XtXt_1 = torch.cat([real_At, fake_B], dim=1)
                XtXt_2 = torch.cat([real_At2, fake_B2], dim=1)
                
                # netE: [1, 4, 3, 3, 18]
                ET_XY    = netE(XtXt_1, timesteps, XtXt_1).mean() - torch.logsumexp(netE(XtXt_1, timesteps, XtXt_2).reshape(-1), dim=0)
                # loss_SB += tau*torch.mean((Xt.detach()-fake_B)**2)

                mse=(((real_At-fake_B)**2)* weights).mean()
                # mse = torch.sum(mse) / torch.sum(weights)

                loss+=args.lambda_SB * (-(T-timesteps[0])/T*tau*ET_XY + tau*mse)
                del mse
                loss_NCE = calculate_NCE_loss(conds[:bs], idt_B)+calculate_NCE_loss(inputs[:bs], fake_B)

                loss += args.lambda_NCE*loss_NCE* 0.5
                del loss_NCE



            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            total_step += 1

            if rank == 0:
                # write train loss for each batch into tensorboard
                tensorboard_writer.add_scalar(
                    "train/train_controlnet_loss_iter", loss.detach().cpu().item(), total_step
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
        
        # val_interval = 10
        # if epoch >= 100-1:
        #     val_interval = 2
        # elif epoch >= 50-1:
        #     val_interval = 5
        val_interval = 1

        if rank == 0:
            tensorboard_writer.add_scalar("train/train_controlnet_loss_epoch", epoch_loss.cpu().item(), total_step)
            # save controlnet only on master GPU (rank 0)
            controlnet_state_dict = controlnet.module.state_dict() if world_size > 1 else controlnet.state_dict()
            torch.save(
                {
                    "epoch": epoch + 1,
                    "loss": epoch_loss,
                    "controlnet_state_dict": controlnet_state_dict,
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
                        "controlnet_state_dict": controlnet_state_dict,
                    },
                    f"{args.model_dir}/{args.exp_name}_epoch{epoch + 1}_loss{epoch_loss:.4f}.pt",
                )
            

            # 没法在训练的里面中间可视化，因为训练是latent的
            if (epoch+1) % val_interval == 0 :
                # if use_vqgan:
                #     output_size = tuple([160, 160, 18])
                # else:
                #     output_size = tuple([160, 160, 32])
                # output_size = tuple([160, 160, 18])
                # num_downsample_level = max(
                #     1,
                #     (
                #         len(args.diffusion_unet_def["num_channels"])
                #         if isinstance(args.diffusion_unet_def["num_channels"], list)
                #         else len(args.diffusion_unet_def["attention_levels"])
                #     ),
                # )
                # divisor = 2 ** (num_downsample_level - 2)
                # # 4, 4
                # logger.info(f"num_downsample_level -> {num_downsample_level}, divisor -> {divisor}.")

                # if use_vqgan:
                #     latent_shape = (args.latent_channels, output_size[0] // 4, output_size[1] // 4, output_size[2])
                # else:
                #     latent_shape = (args.latent_channels, output_size[0] // 4, output_size[1] // 4, output_size[2] // 4)
                # latent_shape = (args.latent_channels, output_size[0] // 4, output_size[1] // 4, output_size[2])
                synthetic_images= ldm_conditional_unsb(
                    autoencoder,
                    unet,
                    controlnet,
                    scale_factor,
                    device,
                    inputs, # 压缩后的src img
                    conds, # 这是压缩后的tgt img
                    lbs, # src lb 没有压缩
                    times=times,
                    T=T,
                    tau = tau,
                    for_show=True,
                    use_vqgan=use_vqgan,
                )

                torch.save(
                    {
                        "epoch": epoch + 1,
                        "loss": epoch_loss,
                        "controlnet_state_dict": controlnet_state_dict,
                    },
                    f"{args.model_dir}/{args.exp_name}_epoch{epoch + 1}_loss{epoch_loss:.4f}.pt",
                )


                output_path = args.output_dir
                # output_path=os.path.join(args.output_dir,'val/')
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                # , tensorboard_writer 无，感觉没必要
                visual_saver_with_cond(epoch+1, synthetic_images,lbs,src_img,tgt_img,output_path, logger)

        torch.cuda.empty_cache()
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
