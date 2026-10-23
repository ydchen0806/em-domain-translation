from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

import copy

import torch
import torch.distributed as dist
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


from data.dataset_lb import SingleDataset_lb
from monai.networks.utils import copy_model_state


from .utils_self import apply_autoencoder, generate_sample, load_autoencoder, load_model
from .diff_model_setting import initialize_distributed, load_config, setup_logging
from .utils import define_instance




def create_optimizer(model: torch.nn.Module, lr: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW(params=model.parameters(), lr=lr, betas=(0.9, 0.999))


def create_lr_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, lr: float) -> torch.optim.lr_scheduler.PolynomialLR:
    return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=0)

def train_one_epoch(
    epoch: int,
    autoencoder: torch.nn.Module,
    model: torch.nn.Module,
    model_without_ddp: torch.nn.Module,
    controlnet: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.PolynomialLR,
    scaler: GradScaler,
    device: torch.device,
    logger: logging.Logger,
    local_rank: int,
    amp: bool = True,
) -> torch.Tensor:

    if local_rank == 0:
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"Epoch {epoch + 1}, lr {current_lr}.")

    _iter = 0
    loss_torch = torch.zeros(2, dtype=torch.float, device=device)
    
    if controlnet is not None: # train controlnet
        model.eval()
        controlnet.train()

        for param in model.parameters():
            param.requires_grad = False

        for param in controlnet.parameters():
            param.requires_grad = True
    else:
        model.train(True)



    for train_data in train_loader:
        current_lr = optimizer.param_groups[0]["lr"]

        _iter += 1

        batch = train_data["image"].to(device)
        labels = train_data["label"].to(device)
        
        with torch.amp.autocast("cuda"):
            images = apply_autoencoder(autoencoder, batch)

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=amp):
            loss = model(images, controlnet, labels=labels)

        if amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        if controlnet is None: 
            model_without_ddp.update_ema()

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
    loss_torch_epoch: float,
    model: torch.nn.Module,
    model_path: str,
) -> None:

    model_state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()

    if "controlnet" in model_path:

        torch.save(
            {
                "epoch": epoch + 1,
                "loss": loss_torch_epoch,
                "controlnet_state_dict": model_state_dict,
            },
            f"{model_path}",
        )
        return


    ema_state_dict1 = copy.deepcopy(model_state_dict)

    for i, (name, _) in enumerate(model.named_parameters()):
        ema_state_dict1[name] = model.ema_params1[i].detach()
    
    torch.save(
        {
            "epoch": epoch + 1,
            "loss": loss_torch_epoch,
            "unet_state_dict": model_state_dict,
            "model_ema1": ema_state_dict1,
        },
        f"{model_path}",
        
    )


def flow_model_train(
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

    if args.mode == "train_cn":
        train_name = "controlnet_train"
        exp_name = "controlnet"
    elif args.mode == "train_flow":
        train_name = "flow_matching_train"
        exp_name = "flow_model"

    if local_rank == 0:
        logger.info(f"[config] mode -> {args.mode}.")
        logger.info(f"[config] ckpt_folder -> {args.model_dir}.")
        logger.info(f"[config] lr -> {getattr(args, train_name)['lr']}.")
        logger.info(f"[config] num_epochs -> {getattr(args, train_name)['n_epochs']}.")
     
        Path(args.model_dir).mkdir(parents=True, exist_ok=True)
        Path(args.tfevent_path).mkdir(parents=True, exist_ok=True)
        
        tensorboard_path = os.path.join(args.tfevent_path, args.output_prefix)
        Path(tensorboard_path).mkdir(parents=True, exist_ok=True)
        tensorboard_writer = SummaryWriter(tensorboard_path)
        logger.info(f"Tensorboard event will be saved as {tensorboard_path}.")

        assert args.mode in ["train_cn", "train_flow"], \
        f"args.mode is 'train_cn' or 'train_flow', current: {args.mode}"
        

    val_interval = getattr(args, train_name)["val_interval"]

    dataset_train = SingleDataset_lb(root_dir = args.target_path, 
                                     train_dataset = args.target_dataset,
                                     length = args.dataloader_len,
                                     strong_aug=False,only_flip=True,norm2=args.norm2)
    
    train_loader = DataLoader(dataset_train, batch_size=getattr(args, train_name)["batch_size"], \
                              num_workers=4, shuffle=True, drop_last=True)

    dataset_eval = SingleDataset_lb(root_dir = args.target_path, 
                                     train_dataset = args.target_dataset,
                                     length=1,
                                     strong_aug=False,
                                     aug=False,norm2=args.norm2)
    
    autoencoder = load_autoencoder(args, device, logger)

    model,model_without_ddp = load_model(args, device, logger)

    if args.mode == "train_cn":
        controlnet = define_instance(args, "controlnet_def").to(device)
        if args.trained_controlnet_path is not None:
            if not os.path.exists(args.trained_controlnet_path):
                raise ValueError("Please download the trained ControlNet checkpoint.")
            controlnet_ckpt = torch.load(args.trained_controlnet_path, map_location=device)
            copy_model_state(controlnet, controlnet_ckpt["controlnet_state_dict"])
            if local_rank == 0:
                logger.info(f"load trained controlnet model from {args.trained_controlnet_path}")
        else:
            copy_model_state(controlnet, model.state_dict())
            if local_rank == 0:
                logger.info("train controlnet model from scratch.")
        if dist.is_initialized():
            controlnet = DistributedDataParallel(controlnet, device_ids=[device], output_device=local_rank, find_unused_parameters=True)
        optimizer = torch.optim.AdamW(params=controlnet.parameters(), lr=args.controlnet_train["lr"])
    else:
        controlnet = None
        optimizer = create_optimizer(model_without_ddp, getattr(args, train_name)["lr"])

    output_path = args.output_dir
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    autoencoder.eval()
    for p in autoencoder.parameters():
        p.requires_grad = False
 

    total_steps = (getattr(args, train_name)["n_epochs"] * len(train_loader.dataset)) / getattr(args, train_name)[
        "batch_size"]
    lr_scheduler = create_lr_scheduler(optimizer, total_steps, getattr(args, train_name)["lr"])

    scaler = GradScaler("cuda")

    torch.set_float32_matmul_precision("highest")
    logger.info("torch.set_float32_matmul_precision -> highest.")


    best_loss = 100

    

    for epoch in range(getattr(args, train_name)["n_epochs"]):
        epoch = epoch

        loss_torch = train_one_epoch(
            epoch,
            autoencoder.module,
            model,
            model_without_ddp,
            controlnet,
            train_loader,
            optimizer,
            lr_scheduler,
            scaler,
            1.0, 
            device,
            logger,
            local_rank,
            amp=amp,
            use_vqgan=args.use_vqgan,
        )

        loss_torch = loss_torch.tolist()

        if torch.cuda.device_count() == 1 or local_rank == 0:
            loss_torch_epoch = loss_torch[0] / loss_torch[1]
            logger.info(f"epoch {epoch + 1} average loss: {loss_torch_epoch:.4f}.")

            tensorboard_writer.add_scalar(f"train_loss_epoch", loss_torch_epoch, epoch+1)
            tensorboard_writer.flush()

            model_filename = os.path.join(args.model_dir, f"{exp_name}_current.pt")
            save_checkpoint(epoch, loss_torch_epoch, 
                            controlnet if args.mode == "train_cn" else model_without_ddp, model_filename)
            

            if loss_torch_epoch < best_loss:
                best_loss = loss_torch_epoch
                best_model_filename =  os.path.join(args.model_dir, f"{exp_name}_epoch{epoch+1:04d}_loss{best_loss:.4f}.pt")

                save_checkpoint(epoch, best_loss, 
                            controlnet if args.mode == "train_cn" else model_without_ddp, best_model_filename)
        
        if args.mode == "train_cn":
            if epoch >= 100-1:
                val_interval = 2
            elif epoch >= 50-1:
                val_interval = 5



        if  (epoch+1) % val_interval == 0 and local_rank == 0: 
            torch.cuda.empty_cache()

            model_filename = os.path.join(args.model_dir, f"{exp_name}_epoch{epoch+1:04d}_loss{best_loss:.4f}.pt")
            save_checkpoint(epoch, loss_torch_epoch,
                            controlnet if args.mode == "train_cn" else model_without_ddp, model_filename)
            
            
            tmp = dataset_eval[0]
            image = tmp['image'].unsqueeze(0).to(device)
            labels = tmp['label'].unsqueeze(0).to(device)

            with torch.no_grad():

                generate_sample(
                    model_without_ddp,
                    controlnet,
                    autoencoder.module,
                    device,
                    epoch+1,
                    output_path,
                    logger, 
                    tensorboard_writer=tensorboard_writer,
                    use_vqgan=args.use_vqgan,
                    visual = True, 
                    image = image,
                    labels = labels,
                )


            
        torch.cuda.empty_cache()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flow Matching Model Training")
    parser.add_argument(
        "--env_config",
        type=str,
        default="./configs/environment_maisi_diff_model_train.json",
        help="Path to environment configuration file",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="./configs/config_maisi_diff_model_train.json",
        help="Path to model training/inference configuration",
    )
    parser.add_argument(
        "--model_def", type=str, default="./configs/config_maisi.json", help="Path to model definition file"
    )
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use for training")
    parser.add_argument("--no_amp", dest="amp", action="store_false", help="Disable automatic mixed precision training")

    args = parser.parse_args()
    flow_model_train(args.env_config, args.model_config, args.model_def, args.num_gpus, args.amp)
