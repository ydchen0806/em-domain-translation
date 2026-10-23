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
# 【以这个为准，在这上面改吧？】

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import random
import torch
import numpy as np
import torch.distributed as dist
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel

import monai
from torch.utils.data import DataLoader
# from monai.data import DataLoader, partition_dataset
from monai.transforms import Compose
from monai.utils import first
from monai.networks.utils import copy_model_state
import logging
from .diff_model_setting import initialize_distributed, load_config, setup_logging
from .utils import define_instance

from torch.utils.tensorboard import SummaryWriter
from data.dataset_unlb import SingleDataset,CondTargetDataset
from data.dataset_lb import CondTargetDataset_lb
from scripts.diff_model_infer_self import run_inference, visual_saver
import warnings

warnings.filterwarnings("ignore")



def load_network(args: argparse.Namespace, device: torch.device, logger: logging.Logger, net: str) -> torch.nn.Module:
    assert net in ["netE", "netG", "netD", "netF"]
    model_def = net + "_def"
    model = define_instance(args, model_def).to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[device], find_unused_parameters=True)

    if getattr(args, f"existing_{net}_path", None) is None:
        logger.info(f"Model training from scratch for {net}.")
    else:
        checkpoint_model = torch.load(f"{args.existing_ckpt_filepath}", map_location=device)
        checkpoint_model = torch.load(f"{getattr(args, f'existing_{net}_path', None)}", map_location=device)
        copy_model_state(model, checkpoint_model["model_state_dict"]) # 之后要改
        # if dist.is_initialized():
        #     unet.module.load_state_dict(checkpoint_unet["unet_state_dict"], strict=True)
        # else:
        #     unet.load_state_dict(checkpoint_unet["unet_state_dict"], strict=True)
        logger.info(f"Pretrained checkpoint {getattr(args, f'existing_{net}_path', None)} loaded.")

    return model



def load_autoencoder(args: argparse.Namespace, device: torch.device, logger: logging.Logger) -> torch.nn.Module:

    autoencoder = define_instance(args, "autoencoder_def").to(device)
    # autoencoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(autoencoder)
    if dist.is_initialized():
        autoencoder = DistributedDataParallel(autoencoder, device_ids=[device], find_unused_parameters=True)
    try:
        # path: ./models/autoencoder_epoch0.pt
        checkpoint_autoencoder = torch.load(args.trained_autoencoder_path, weights_only=True)
        new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint_autoencoder.items()}
        if dist.is_initialized():
            autoencoder.module.load_state_dict(new_state_dict)
        else:
            autoencoder.load_state_dict(new_state_dict)
        logger.info(f"Pretrained AE {args.trained_autoencoder_path} loaded.")
    except FileNotFoundError:
        logger.error(f"The file {args.trained_autoencoder_path} does not exist!")
    except RuntimeError as e:
        logger.error(f"Error loading the model state dict: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")

    return autoencoder  

def calculate_scale_factor(
        train_loader: DataLoader, 
        autoencoder: torch.nn.Module,
        device: torch.device, 
        logger: logging.Logger) -> torch.Tensor:
    """
    Calculate the scaling factor for the dataset.

    Args:
        train_loader (DataLoader): Data loader for training.
        device (torch.device): Device to use for calculation.
        logger (logging.Logger): Logger for logging information.

    Returns:
        torch.Tensor: Calculated scaling factor.
    """
    check_data = first(train_loader)
    z = check_data["image"].to(device) # [b, 1, 160, 160, 32]
    zc = check_data["cond"].to(device)
    # z = z[0].unsqueeze(0) # [1, 1, 160, 160, 32]
    # print('z:',z.shape)
    # 如果每张gpu都计算(b=8, xxx)会太大

    with torch.amp.autocast("cuda"):
        # batch = torch.from_numpy(images).float().to(device).unsqueeze(0).unsqueeze(0)
        z = autoencoder.encode_stage_2_inputs(z) # [b, 4, 40, 40, 8]
        z=z.cpu().detach().numpy()
        # z = z.squeeze().cpu().detach().numpy().transpose(1, 2, 3, 0) # (40, 40, 8, 4)
        z = torch.from_numpy(z).to(device)

        zc = autoencoder.encode_stage_2_inputs(zc)
        zc=zc.cpu().detach().numpy()
        zc = torch.from_numpy(zc).to(device)
    scale_factor = 1 / torch.std(z) # scale_factor: tensor(0.9126)
    scale_factor_cond = 1 / torch.std(zc)
    del z,zc
    logger.info(f"Scaling factor set to {scale_factor}.")

    if dist.is_initialized():
        dist.barrier()
        dist.all_reduce(scale_factor, op=torch.distributed.ReduceOp.AVG)
        dist.all_reduce(scale_factor_cond, op=torch.distributed.ReduceOp.AVG)
        
    logger.info(f"scale_factor -> {scale_factor}.")
    return scale_factor,scale_factor_cond


def create_optimizer(model: torch.nn.Module, lr: float) -> torch.optim.Optimizer:
    """
    Create optimizer for training.

    Args:
        model (torch.nn.Module): Model to optimize.
        lr (float): Learning rate.

    Returns:
        torch.optim.Optimizer: Created optimizer.
    """
    return torch.optim.Adam(params=model.parameters(), lr=lr)


def create_lr_scheduler(optimizer: torch.optim.Optimizer, total_steps: int) -> torch.optim.lr_scheduler.PolynomialLR:
    """
    Create learning rate scheduler.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer to schedule.
        total_steps (int): Total number of training steps.

    Returns:
        torch.optim.lr_scheduler.PolynomialLR: Created learning rate scheduler.
    """
    return torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=total_steps, power=2.0)


def train_one_epoch(
    epoch: int,
    autoencoder: torch.nn.Module,
    unet: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.PolynomialLR,
    loss_pt: torch.nn.L1Loss,
    scaler: GradScaler,
    scale_factor: torch.Tensor,
    scale_factor_cond: torch.Tensor,
    noise_scheduler: torch.nn.Module,
    num_images_per_batch: int,
    num_train_timesteps: int,
    device: torch.device,
    logger: logging.Logger,
    local_rank: int,
    amp: bool = True,
) -> torch.Tensor:
    """
    Train the model for one epoch.

    Args:
        epoch (int): Current epoch number.
        unet (torch.nn.Module): UNet model.
        train_loader (DataLoader): Data loader for training.
        optimizer (torch.optim.Optimizer): Optimizer.
        lr_scheduler (torch.optim.lr_scheduler.PolynomialLR): Learning rate scheduler.
        loss_pt (torch.nn.L1Loss): Loss function.
        scaler (GradScaler): Gradient scaler for mixed precision training.
        scale_factor (torch.Tensor): Scaling factor.
        noise_scheduler (torch.nn.Module): Noise scheduler.
        num_images_per_batch (int): Number of images per batch.
        num_train_timesteps (int): Number of training timesteps.
        device (torch.device): Device to use for training.
        logger (logging.Logger): Logger for logging information.
        local_rank (int): Local rank for distributed training.
        amp (bool): Use automatic mixed precision training.

    Returns:
        torch.Tensor: Training loss for the epoch.
    """
    if local_rank == 0:
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"Epoch {epoch + 1}, lr {current_lr}.")

    _iter = 0
    scale_factor_cond = 1.0
    loss_torch = torch.zeros(2, dtype=torch.float, device=device)

    unet.train()


    for train_data in train_loader:
        current_lr = optimizer.param_groups[0]["lr"]

        _iter += 1

        batch = train_data["image"].to(device) # [b, 1, 160, 160, 32]
        
        cond = train_data["cond"].to(device)
        # print('cond:',cond.shape)
        
        with torch.amp.autocast("cuda"):
            # batch = torch.from_numpy(images).float().to(device).unsqueeze(0).unsqueeze(0)
            images = autoencoder.encode_stage_2_inputs(batch) # [1, 4, 40, 40, 8]
            cond = autoencoder.encode_stage_2_inputs(cond) #  [1, 4, 40, 40, 8]
            # images = z.squeeze()
            # images = z.squeeze().cpu().detach().numpy().transpose(1, 2, 3, 0)
            # (40, 40, 8, 4)
            images = images * scale_factor
            cond = cond * scale_factor_cond




        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=amp):
            noise = torch.randn( 
                (num_images_per_batch, 4, images.size(-3), images.size(-2), images.size(-1)), device=device
            )

            timesteps = torch.randint(0, num_train_timesteps, (images.shape[0],), device=images.device).long()

            noisy_latent = noise_scheduler.add_noise(original_samples=images, noise=noise, timesteps=timesteps)
            # noisy_latent: torch.Size([b, 4, 40, 40, 8])
            # 【如果要直接concat，就需要新的一个vae对cond压缩】
            noise_pred = unet(
                x=torch.cat((noisy_latent, cond), dim=1), # [b, 8, 40, 40, 8]
                timesteps=timesteps,
                # context=cond,
            )

            loss = loss_pt(noise_pred.float(), noise.float())

        if amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        loss_torch[0] += loss.item()
        loss_torch[1] += 1.0

        if local_rank == 0:
            logger.info(
                "[{0}] epoch {1}, iter {2}/{3}, loss: {4:.4f}, lr: {5:.12f}.".format(
                    str(datetime.now())[:19], epoch + 1, _iter, len(train_loader), loss.item(), current_lr
                )
            )

    if dist.is_initialized():
        dist.all_reduce(loss_torch, op=torch.distributed.ReduceOp.SUM)

    return loss_torch


def save_checkpoint(
    epoch: int,
    unet: torch.nn.Module,
    loss_torch_epoch: float,
    num_train_timesteps: int,
    scale_factor: torch.Tensor,
    ckpt_folder: str,
    args: argparse.Namespace,
) -> None:
    """
    Save checkpoint.

    Args:
        epoch (int): Current epoch number.
        unet (torch.nn.Module): UNet model.
        loss_torch_epoch (float): Training loss for the epoch.
        num_train_timesteps (int): Number of training timesteps.
        scale_factor (torch.Tensor): Scaling factor.
        ckpt_folder (str): Checkpoint folder path.
        args (argparse.Namespace): Configuration arguments.
    """
    unet_state_dict = unet.module.state_dict() if dist.is_initialized() else unet.state_dict()
    model_filename = "diff_unet_epoch{0:04d}_loss{1:.4f}.pt".format(epoch+1, loss_torch_epoch)
    torch.save(
        {
            "epoch": epoch + 1,
            "loss": loss_torch_epoch,
            "num_train_timesteps": num_train_timesteps,
            "scale_factor": scale_factor,
            "unet_state_dict": unet_state_dict,
        },
        f"{ckpt_folder}/{model_filename}",
    )


def unsb_model_train(
    env_config_path: str, model_config_path: str, model_def_path: str, num_gpus: int, amp: bool = True,
) -> None:
    """
    Main function to train a diffusion model.

    Args:
        env_config_path (str): Path to the environment configuration file.
        model_config_path (str): Path to the model configuration file.
        model_def_path (str): Path to the model definition file.
        num_gpus (int): Number of GPUs to use for training.
        amp (bool): Use automatic mixed precision training.
    """
    args = load_config(env_config_path, model_config_path, model_def_path)
    local_rank, world_size, device = initialize_distributed(num_gpus)
    

    logger = setup_logging("training")
    logger.info(f"Using {device} of {world_size}")

    

    if local_rank == 0:
        logger.info(f"[config] ckpt_folder -> {args.model_dir}.")
        logger.info(f"[config] lr -> {args.diff_unsb_train['lr']}.")
        logger.info(f"[config] num_epochs -> {args.diff_unsb_train['n_epochs']}.")
        logger.info(f"[config] num_train_timesteps -> {args.noise_scheduler['num_train_timesteps']}.")
        
        Path(args.model_dir).mkdir(parents=True, exist_ok=True)

        Path(args.tfevent_path).mkdir(parents=True, exist_ok=True)
        tensorboard_path = os.path.join(args.tfevent_path, "diff_unsb")
        Path(tensorboard_path).mkdir(parents=True, exist_ok=True)
        tensorboard_writer = SummaryWriter(tensorboard_path)
        logger.info(f"Tensorboard event will be saved as {tensorboard_path}.")
        

    # 【按需修改】
    val_interval = args.diff_unsb_train["val_interval"]

    # 读取["image","cond"(target img)]
    dataset_train = CondTargetDataset_lb(src_dir='./local_data/CREMI', src_dataset='cremiC_inputs_interp.h5',\
                                          tgt_dir='./local_data/AC3_AC4/', tgt_dataset='AC4_inputs.h5', length=300)
        

    train_loader = DataLoader(dataset_train, batch_size=args.diff_unsb_train["batch_size"], \
                              num_workers=4, shuffle=True, drop_last=True)


    ### 【加载】
    autoencoder = load_autoencoder(args, device, logger)
    netG = load_network(args, device, logger,"netG")
    # 注意参数，maisi是32, unsb是64
    netD = load_network(args, device, logger,"netD")
    netF = load_network(args, device, logger,"netF")
    netE = load_network(args, device, logger,"netE")

    noise_scheduler = define_instance(args, "noise_scheduler")

    scale_factor,scale_factor_cond= calculate_scale_factor(train_loader, autoencoder.module, device, logger)
    
    optimizer = create_optimizer(netG, args.diff_unsb_train["lr"])

    total_steps = (args.diff_unsb_train["n_epochs"] * len(train_loader.dataset)) / args.diff_unsb_train[
        "batch_size"]
    lr_scheduler = create_lr_scheduler(optimizer, total_steps)
    loss_pt = torch.nn.L1Loss()
    scaler = GradScaler("cuda")

    torch.set_float32_matmul_precision("highest")
    logger.info("torch.set_float32_matmul_precision -> highest.")


    best_loss =100

    for epoch in range(args.diff_unsb_train["n_epochs"]):
         # 里面没有可视化，tensorboard, 因为网络是预测噪声
        loss_torch = train_one_epoch(
            epoch,
            autoencoder.module,
            unet,
            train_loader,
            optimizer,
            lr_scheduler,
            loss_pt,
            scaler,
            scale_factor,
            scale_factor_cond,
            noise_scheduler,
            args.diff_unsb_train["batch_size"],
            args.noise_scheduler["num_train_timesteps"],
            device,
            logger,
            local_rank,
            amp=amp,
        )

        loss_torch = loss_torch.tolist()
        if torch.cuda.device_count() == 1 or local_rank == 0:
            loss_torch_epoch = loss_torch[0] / loss_torch[1]
            logger.info(f"epoch {epoch + 1} average loss: {loss_torch_epoch:.4f}.")
            # logger.handlers[0].flush()

            tensorboard_writer.add_scalar(f"train_loss_epoch", loss_torch_epoch, epoch+1)
            tensorboard_writer.flush()
            # save_checkpoint(
            #     epoch,
            #     unet,
            #     loss_torch_epoch,
            #     args.noise_scheduler["num_train_timesteps"],
            #     scale_factor,
            #     args.model_dir,
            #     args,
            # )
            unet_state_dict = unet.module.state_dict() if dist.is_initialized() else unet.state_dict()
            model_filename = "diff_current.pt"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "loss": loss_torch_epoch,
                    "num_train_timesteps": args.noise_scheduler["num_train_timesteps"],
                    "scale_factor": scale_factor,
                    "unet_state_dict": unet_state_dict,
                },
                f"{args.model_dir}/{model_filename}",
            )

            if loss_torch_epoch < best_loss:
                best_loss = loss_torch_epoch
                best_model_filename = f"diff_unet_epoch{epoch+1:04d}_loss{best_loss:.4f}.pt"
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "loss": best_loss,
                        "num_train_timesteps": args.noise_scheduler["num_train_timesteps"],
                        "scale_factor": scale_factor,
                        "unet_state_dict": unet_state_dict,
                    },
                    f"{args.model_dir}/{best_model_filename}",
                )
        


        # 没法在训练的里面中间可视化，因为训练是latent的
        if (epoch==0 or (epoch+1) % val_interval == 0) and local_rank == 0: # (epoch+1)
            output_size = tuple([160, 160, 32])
            num_downsample_level = max(
                1,
                (
                    len(args.diffusion_unet_def["num_channels"])
                    if isinstance(args.diffusion_unet_def["num_channels"], list)
                    else len(args.diffusion_unet_def["attention_levels"])
                ),
            )
            divisor = 2 ** (num_downsample_level - 2)
            # 4, 4
            logger.info(f"num_downsample_level -> {num_downsample_level}, divisor -> {divisor}.")
            
            unet_state_dict = unet.module.state_dict() if dist.is_initialized() else unet.state_dict()
            best_model_filename = f"diff_unet_epoch{epoch+1:04d}_loss{loss_torch_epoch:.4f}.pt"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "loss": loss_torch_epoch,
                    "num_train_timesteps": args.noise_scheduler["num_train_timesteps"],
                    "scale_factor": scale_factor,
                    "unet_state_dict": unet_state_dict,
                },
                f"{args.model_dir}/{best_model_filename}",
            )

            top_region_index_tensor, bottom_region_index_tensor, spacing_tensor = None, None, None
            dataset_infer = SingleDataset(root_dir='./local_data/AC3_AC4/', train_dataset='AC4_inputs.h5',length=1)
            # infer_loader = DataLoader(dataset_infer, batch_size=1, \
            #                     num_workers=4, shuffle=True, drop_last=True)
            cond_infer = dataset_infer[0]['image'].unsqueeze(0).to(device) # cond:[1, 1, 160, 160, 32]
            data = run_inference(
                args,
                device,
                autoencoder.module,
                unet,
                scale_factor,
                top_region_index_tensor,
                bottom_region_index_tensor,
                spacing_tensor,
                output_size,
                divisor,
                logger,
                cond=cond_infer,
                scale_factor_cond= scale_factor_cond,
            )
            # output_path = "{0}/{1}_seed{2}_size{3:d}x{4:d}x{5:d}_spacing{6:.2f}x{7:.2f}x{8:.2f}_{9}_rank{10}.nii.gz".format(
            #     args.output_dir,
            #     output_prefix,
            #     random_seed,
            #     output_size[0],
            #     output_size[1],
            #     output_size[2],
            #     local_rank,
            # )
            # [1, 1, 160, 160, 32]
            cond_infer = cond_infer.squeeze().cpu().detach().numpy()

            cond_infer = cond_infer * 255.0
            np.clip(cond_infer, 0, 255)
            cond_infer=cond_infer.astype(np.uint8) 

            output_path = args.output_dir
            # output_path=os.path.join(args.output_dir,'val/')
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            visual_saver(epoch+1, data, output_path, logger, tensorboard_writer,cond=cond_infer)
            


            

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UNSB Model Training")
    parser.add_argument(
        "--env_config",
        type=str,
        default="./configs/unsb/environment_unsb.json",
        help="Path to environment configuration file",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="./configs/unsb/config_unsb_model_train.json",
        help="Path to model training/inference configuration",
    )
    parser.add_argument(
        "--model_def", type=str, default="./configs/unsb/config_unsb.json", help="Path to model definition file"
    )
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use for training")
    parser.add_argument("--no_amp", dest="amp", action="store_false", help="Disable automatic mixed precision training")

    args = parser.parse_args()
    unsb_model_train(args.env_config, args.model_config, args.model_def, args.num_gpus, args.amp)
