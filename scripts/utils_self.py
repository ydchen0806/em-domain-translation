import argparse
import copy
import logging
import os
import random 
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import torch.nn.functional as F

from monai.networks.utils import copy_model_state

from .sample import ReconModel
from .utils import define_instance
from networks_copy.denoiser import Denoiser


def norm(t):
    return F.normalize(t, dim=1, eps=1e-10)

def cosine_similarity(X,Y):
    '''
    compute cosine similarity for each pair of image
    Input shape: (batch,channel,H,W)
    Output shape: (batch,1)
    '''
    X_flat = X.reshape(X.shape[0], X.shape[1], -1)
    Y_flat = Y.reshape(Y.shape[0], Y.shape[1], -1)

    corr = norm(X_flat) * norm(Y_flat)
    similarity = corr.sum(dim=1).mean(dim=1)

    return similarity

def mse(x,y):
    return (x - y).square().sum(dim=tuple(range(1, x.dim())))


def load_autoencoder(args: argparse.Namespace, device: torch.device, logger: logging.Logger):
    if args.use_vqgan:
        autoencoder = define_instance(args, "vqgan_def").to(device)
        autoencoder = autoencoder.load_from_checkpoint(args.trained_autoencoder_path,
                    embedding_dim=args.latent_channels,
                    n_codes=args.vqgan_def["n_codes"],
                    n_hiddens=args.vqgan_def["n_hiddens"],
                    downsample=args.vqgan_def["downsample"],
                    image_channels=args.image_channels).to(device)
    else:
        autoencoder = define_instance(args, "autoencoder_def").to(device)
    if dist.is_initialized():
        autoencoder = DistributedDataParallel(autoencoder, device_ids=[device], find_unused_parameters=True)
    if not args.use_vqgan:
        try:
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


def load_model(args: argparse.Namespace, device: torch.device, logger: logging.Logger, infer=False, debug=False) -> torch.nn.Module:

    # # diff model (flow matching)
    model = Denoiser(args, device, debug=debug)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)

    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[device], find_unused_parameters=True)
        model_without_ddp = model.module
    else:
        model_without_ddp = model
    


    if args.trained_model_path is None:
        if args.mode == "train_cn":
            logger.info("Wrong! Training controlnet needs trained flow model!")
        else:
            logger.info("Training from scratch.")
        # # parameters()是可训练参数
        model_without_ddp.ema_params1 = copy.deepcopy(list(model_without_ddp.parameters()))
        # model_without_ddp.ema_params2 = copy.deepcopy(list(model_without_ddp.parameters()))
    else:
        checkpoint = torch.load(f"{args.trained_model_path}", map_location=device)

        if args.mode == "train_cn" or infer: # 直接加载ema1参数，模型推理
            model_without_ddp.load_state_dict(checkpoint['model_ema1'], strict=True)
            logger.info('load EMA1 for inference.')
        else:
            copy_model_state(model, checkpoint["unet_state_dict"])

            ema_state_dict1 = checkpoint['model_ema1']
            model_without_ddp.ema_params1 = [ema_state_dict1[name].cuda() for name, _ in model_without_ddp.named_parameters()]
        
        # ema_state_dict2 = checkpoint['model_ema2']
        # model_without_ddp.ema_params2 = [ema_state_dict2[name].cuda() for name, _ in model_without_ddp.named_parameters()]
        
        del checkpoint
        logger.info(f"Pretrained checkpoint {args.trained_model_path} loaded.")


    return model, model_without_ddp




def apply_autoencoder(
        autoencoder,
        inputs,
        recon=False):
    # decode
    if recon:
        z = (((inputs.permute(0,1,4,3,2) + 1.0) / 2.0) * (autoencoder.codebook.embeddings.max() -
                                                autoencoder.codebook.embeddings.min())) + autoencoder.codebook.embeddings.min()
        return autoencoder.decode(z,quantize=True).permute(0,1,4,3,2)

    # encode
    z = autoencoder.encode(inputs.permute(0,1,4,3,2),
                                            quantize=False)

    z = ((z - autoencoder.codebook.embeddings.min()) /
                    (autoencoder.codebook.embeddings.max() -
                    autoencoder.codebook.embeddings.min())) * 2.0 - 1.0
    return z.permute(0,1,4,3,2)




def visual_saver(
    epoch: int,
    data: np.ndarray,
    output_path: str,
    tensorboard_writer: SummaryWriter = None,
    gt: np.ndarray = None,
    lb: np.ndarray = None,
) -> None:
    """
    Save the generated synthetic image to a file.

    Args:
        data (np.ndarray): [H, W, C]
        output_path (str): Path to save the output image.
        logger (logging.Logger): Logger for logging information.
    """
    import imageio
    from PIL import Image

    video_path=os.path.join(output_path,'video/')
    img_path=os.path.join(output_path,'img/')

    if not os.path.exists(video_path):
        os.makedirs(video_path)
    if not os.path.exists(img_path):
        os.makedirs(img_path)

    data = np.transpose(data, (2, 1, 0))  # [H, W, C]->[C, H, W]

    

    # [32, 160, 160]
    k = random.randint(0, data.shape[0]-1)
    if tensorboard_writer is None: # ### 说明是纯采样推理，为了debug对比使用相同的切片
        k = 0
    Image.fromarray(data[k]).save(os.path.join(img_path, str(epoch).zfill(3)+'_'+str(k).zfill(2)+'.png'))

    imageio.mimsave(os.path.join(video_path, str(epoch).zfill(3)+'.mp4'), data, fps=6)

    if lb is not None and gt is not None:

        lb = lb[0,0].cpu().detach().numpy()
        lb = lb * 255.0
        lb=np.clip(lb, 0, 255)
        lb=lb.astype(np.uint8)
        lb = np.transpose(lb, (2, 1, 0))

        gt = gt[0,0].cpu().detach().numpy()
        gt = gt * 255.0
        # gt = (gt + 1.0) * 127.5
        gt=np.clip(gt, 0, 255)
        gt=gt.astype(np.uint8)
        gt = np.transpose(gt, (2, 1, 0))

        Image.fromarray(lb[k]).save(os.path.join(img_path, str(epoch).zfill(3)+'_'+str(k).zfill(2)+'_lb.png'))
        imageio.mimsave(os.path.join(video_path, str(epoch).zfill(3)+'_lb.mp4'), lb, fps=6)
        
        Image.fromarray(gt[k]).save(os.path.join(img_path, str(epoch).zfill(3)+'_'+str(k).zfill(2)+'_gt.png'))
        imageio.mimsave(os.path.join(video_path, str(epoch).zfill(3)+'_gt.mp4'), gt, fps=6)

    # logger.info(f"Saved {output_path}.")

    if tensorboard_writer is not None:
        tensorboard_writer.add_image(
                    "val_img",
                    torch.from_numpy(data[k]).unsqueeze(0),
                    epoch,
                )
        if lb is not None and gt is not None:
            tensorboard_writer.add_image(
                    "val_lb",
                    torch.from_numpy(lb[k]).unsqueeze(0),
                    epoch,
                )
            tensorboard_writer.add_image(
                    "val_gt",
                    torch.from_numpy(gt[k]).unsqueeze(0),
                    epoch,
                )
        tensorboard_writer.flush()


def visual_saver_with_cond(
    epoch: int,
    data, # syn_img  ## list
    lb: np.ndarray, # src_lb
    src_img: torch.Tensor, # src_img [1,1,160,160,32]
    tgt_img: torch.Tensor, # tgt_img
    output_path: str,
    logger: logging.Logger,
    tensorboard_writer: SummaryWriter = None,
) -> None:
    """
    Save the generated synthetic image to a file.

    Args:
        data (np.ndarray): [H, W, C]
        output_path (str): Path to save the output image.
        logger (logging.Logger): Logger for logging information.
    """
    import imageio
    from PIL import Image

    # video_path=os.path.join(output_path,'video/')
    img_path=os.path.join(output_path,'img/')

    # if not os.path.exists(video_path):
    #     os.makedirs(video_path)
    if not os.path.exists(img_path):
        os.makedirs(img_path)

    
    lb = lb[0,0].cpu().detach().numpy()
    lb = lb * 255.0
    lb = np.clip(lb, 0, 255)
    lb=lb.astype(np.uint8)

    src_img = src_img[0,0].cpu().detach().numpy()
    src_img = src_img * 255.0
    src_img = np.clip(src_img, 0, 255)
    src_img=src_img.astype(np.uint8)

    tgt_img = tgt_img[0,0].cpu().detach().numpy()
    tgt_img = tgt_img * 255.0
    tgt_img = np.clip(tgt_img, 0, 255)
    tgt_img=tgt_img.astype(np.uint8)

    # [H, W, 32]->[32, H, W]
    lb = np.transpose(lb, (2, 1, 0))
    src_img = np.transpose(src_img, (2, 1, 0))
    tgt_img = np.transpose(tgt_img, (2, 1, 0))

    k = random.randint(0, src_img.shape[0]-1)
    if tensorboard_writer is None: # ### 说明是纯采样推理，为了debug对比使用相同的切片
        k = 0

    data = np.transpose(data, (2, 1, 0))
    
    Image.fromarray(data[k]).save(os.path.join(img_path, str(epoch).zfill(3)+'_'+str(k).zfill(2)+'.png'))
    Image.fromarray(lb[k]).save(os.path.join(img_path, str(epoch).zfill(3)+'_'+str(k).zfill(2)+'_lb.png'))
    Image.fromarray(src_img[k]).save(os.path.join(img_path, str(epoch).zfill(3)+'_'+str(k).zfill(2)+'_src.png'))
    Image.fromarray(tgt_img[k]).save(os.path.join(img_path, str(epoch).zfill(3)+'_'+str(k).zfill(2)+'_tgt.png'))


    # logger.info(f"Saved {output_path}.")

    if tensorboard_writer is not None:
        tensorboard_writer.add_image(
                    "val_img",
                    torch.from_numpy(data[k]).unsqueeze(0),
                    epoch,
                )
        tensorboard_writer.add_image(
                    "val_lb",
                    torch.from_numpy(lb[k]).unsqueeze(0),
                    epoch,
                )
        tensorboard_writer.add_image(
                    "val_src",
                    torch.from_numpy(src_img[k]).unsqueeze(0),
                    epoch,
                )
        tensorboard_writer.add_image(
                    "val_tgt",
                    torch.from_numpy(tgt_img[k]).unsqueeze(0),
                    epoch,
                )
        tensorboard_writer.flush()



def generate_sample(
        model_without_ddp, 
        controlnet,
        autoencoder, 
        device, 
        epoch, 
        output_path,
        logger, 
        tensorboard_writer = None,
        use_vqgan = True, 
        visual = True, 
        image = None,
        labels = None,
        t=0.5, # 0.25
    ):

    model_without_ddp.eval()


    # switch to ema params, 默认是EMA1
    if controlnet is None: # train model
        model_state_dict = copy.deepcopy(model_without_ddp.state_dict())
        ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
        for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
            assert name in ema_state_dict
            ema_state_dict[name] = model_without_ddp.ema_params1[i]
        print("Switch to ema")
        model_without_ddp.load_state_dict(ema_state_dict)
    else:
        assert labels is not None
        controlnet.eval()


    if not use_vqgan:
        recon_model = ReconModel(autoencoder=autoencoder, scale_factor=1.0).to(device)
    
    autoencoder.eval()

    with torch.amp.autocast("cuda", enabled=True):
        # (images, controlnet, labels=labels)
        images = apply_autoencoder(autoencoder, image, use_vqgan=use_vqgan)
        latent_x = model_without_ddp.generate(controlnet, labels=labels, x=images, t_start=t)
        synthetic_images = apply_autoencoder(
            recon_model if not use_vqgan else autoencoder, 
            latent_x, 
            use_vqgan=use_vqgan, 
            recon=True, 
            scale_factor=1.0)

    if visual:
        data = synthetic_images.squeeze().squeeze().cpu().detach().numpy()
        data = data * 255.0
        # data = (data + 1.0) * 127.5
        data = np.clip(data, 0, 255)
        data=data.astype(np.uint8) 

        visual_saver(epoch, data, output_path, logger, tensorboard_writer, gt=image, lb=labels)


    # back to no ema
    if controlnet is None:
        print("Switch back from ema")
        model_without_ddp.load_state_dict(model_state_dict)

    return synthetic_images


def generate_sample_translation(
        model_without_ddp, 
        controlnet,
        autoencoder, 
        device, 
        epoch, 
        output_path,
        logger, 
        tensorboard_writer = None,
        use_vqgan = True, 
        visual = True, 
        src_img = None,
        src_lb = None,
        tgt_img = None,
        t=0.5, # 0.25
        lambda_list = [0,0],
    ):

    model_without_ddp.eval()


    # switch to ema params, 默认是EMA1
    if controlnet is None: # train model
        model_state_dict = copy.deepcopy(model_without_ddp.state_dict())
        ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
        for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
            assert name in ema_state_dict
            ema_state_dict[name] = model_without_ddp.ema_params1[i]
        print("Switch to ema")
        model_without_ddp.load_state_dict(ema_state_dict)
    else:
        assert src_lb is not None
        controlnet.eval()


    if not use_vqgan:
        recon_model = ReconModel(autoencoder=autoencoder, scale_factor=1.0).to(device)
    
    autoencoder.eval()

    with torch.amp.autocast("cuda", enabled=True):
        # (images, controlnet, labels=labels)
        src_images = apply_autoencoder(autoencoder, src_img, use_vqgan=use_vqgan)
        tgt_images = apply_autoencoder(autoencoder, tgt_img, use_vqgan=use_vqgan)
        latent_x = model_without_ddp.translation(controlnet, src_lb, src_images, tgt_images, t_start=t, lambda_list = lambda_list)
        synthetic_images = apply_autoencoder(
            recon_model if not use_vqgan else autoencoder, 
            latent_x, 
            use_vqgan=use_vqgan, 
            recon=True, 
            scale_factor=1.0)

    if visual:
        data = synthetic_images.squeeze().squeeze().cpu().detach().numpy()
        data = data * 255.0
        # data = (data + 1.0) * 127.5
        data = np.clip(data, 0, 255)
        data=data.astype(np.uint8) 

        visual_saver_with_cond(epoch, data, src_lb, src_img, tgt_img, output_path, logger, tensorboard_writer)


    # back to no ema
    if controlnet is None:
        print("Switch back from ema")
        model_without_ddp.load_state_dict(model_state_dict)

    return synthetic_images
