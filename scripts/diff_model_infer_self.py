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

from __future__ import annotations

import argparse
import logging
import os
import random
from datetime import datetime

import nibabel as nib
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from PIL import Image

from monai.inferers import sliding_window_inference
from monai.utils import set_determinism

from data.dataset_unlb import SingleDataset
from .diff_model_setting import initialize_distributed, load_config, setup_logging
from .sample import ReconModel, check_input
from .utils import define_instance


def apply_autoencoder(
        autoencoder,
        inputs, # [b, 1, 160, 160, z]  -> [b, 4, 40, 40, z]
        use_vqgan=False,
        recon=False,
        scale_factor=None):
    # decode
    if recon: 
        if use_vqgan:
            scale_factor = 1.0
            # z = ((inputs.permute(0,1,4,3,2)/scale_factor) * (autoencoder.codebook.embeddings.max() -
            #                                         autoencoder.codebook.embeddings.min())) + autoencoder.codebook.embeddings.min()
            z = (((inputs.permute(0,1,4,3,2) + 1.0) / 2.0) * (autoencoder.codebook.embeddings.max() -
                                                    autoencoder.codebook.embeddings.min())) + autoencoder.codebook.embeddings.min()
            # z = inputs.permute(0,1,4,3,2) # /scale_factor
            return autoencoder.decode(z,quantize=True).permute(0,1,4,3,2)
        else:
            return autoencoder(inputs) # Recon_model
    # encode
    if use_vqgan:
        z = autoencoder.encode(inputs.permute(0,1,4,3,2),
                                               quantize=False)
        # z = ((z - autoencoder.codebook.embeddings.min()) /
        #              (autoencoder.codebook.embeddings.max() -
        #               autoencoder.codebook.embeddings.min())).detach()

        z = ((z - autoencoder.codebook.embeddings.min()) /
                     (autoencoder.codebook.embeddings.max() -
                      autoencoder.codebook.embeddings.min())) * 2.0 - 1.0
        return z.permute(0,1,4,3,2)
    else: # 【注意用vae的话要同步修改dataset_lb和single_dataset为[32,160,160]，[16]不行】
        return autoencoder.encode_stage_2_inputs(inputs)

def set_random_seed(seed: int) -> int:
    """
    Set random seed for reproducibility.

    Args:
        seed (int): Random seed.

    Returns:
        int: Set random seed.
    """
    random_seed = random.randint(0, 99999) if seed is None else seed
    set_determinism(random_seed)
    return random_seed


def load_models(args: argparse.Namespace, device: torch.device, logger: logging.Logger) -> tuple:
    """
    Load the autoencoder and UNet models.

    Args:
        args (argparse.Namespace): Configuration arguments.
        device (torch.device): Device to load models on.
        logger (logging.Logger): Logger for logging information.

    Returns:
        tuple: Loaded autoencoder, UNet model, and scale factor.
    """
    autoencoder = define_instance(args, "autoencoder_def").to(device)
    # autoencoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(autoencoder)
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

    unet = define_instance(args, "diffusion_unet_def").to(device)
    checkpoint = torch.load(f"{args.infer_model}", map_location=device, weights_only=False)
    unet.load_state_dict(checkpoint["unet_state_dict"], strict=True)
    logger.info(f"checkpoints {args.infer_model} loaded.")

    scale_factor = checkpoint["scale_factor"]
    logger.info(f"scale_factor -> {scale_factor}.")

    return autoencoder, unet, scale_factor


def prepare_tensors(args: argparse.Namespace, device: torch.device) -> tuple:
    """
    Prepare necessary tensors for inference.

    Args:
        args (argparse.Namespace): Configuration arguments.
        device (torch.device): Device to load tensors on.

    Returns:
        tuple: Prepared top_region_index_tensor, bottom_region_index_tensor, and spacing_tensor.
    """
    top_region_index_tensor = np.array(args.diffusion_unet_inference["top_region_index"]).astype(float) * 1e2
    bottom_region_index_tensor = np.array(args.diffusion_unet_inference["bottom_region_index"]).astype(float) * 1e2
    spacing_tensor = np.array(args.diffusion_unet_inference["spacing"]).astype(float) * 1e2

    top_region_index_tensor = torch.from_numpy(top_region_index_tensor[np.newaxis, :]).half().to(device)
    bottom_region_index_tensor = torch.from_numpy(bottom_region_index_tensor[np.newaxis, :]).half().to(device)
    spacing_tensor = torch.from_numpy(spacing_tensor[np.newaxis, :]).half().to(device)

    return top_region_index_tensor, bottom_region_index_tensor, spacing_tensor


def run_inference(
    args: argparse.Namespace,
    device: torch.device,
    autoencoder: torch.nn.Module,
    unet: torch.nn.Module,
    scale_factor: float,
    top_region_index_tensor: torch.Tensor,
    bottom_region_index_tensor: torch.Tensor,
    spacing_tensor: torch.Tensor,
    output_size: tuple,
    divisor: int,
    logger: logging.Logger,
    cond: torch.Tensor=None,
    scale_factor_cond: float=None,
    use_vqgan: bool=False,
) -> np.ndarray:
    """
    Run the inference to generate synthetic images.

    Args:
        args (argparse.Namespace): Configuration arguments.
        device (torch.device): Device to run inference on.
        autoencoder (torch.nn.Module): Autoencoder model.
        unet (torch.nn.Module): UNet model.
        scale_factor (float): Scale factor for the model.
        top_region_index_tensor (torch.Tensor): Top region index tensor.
        bottom_region_index_tensor (torch.Tensor): Bottom region index tensor.
        spacing_tensor (torch.Tensor): Spacing tensor.
        output_size (tuple): Output size of the synthetic image.
        divisor (int): Divisor for downsample level.
        logger (logging.Logger): Logger for logging information.

    Returns:
        np.ndarray: Generated synthetic image data.
    """
    if not use_vqgan:
        noise = torch.randn(
            (
                1,
                args.latent_channels,
                output_size[0] // divisor,
                output_size[1] // divisor,
                output_size[2], # [vae也改了]
            ),
            device=device,
        )
    else:
        noise = torch.randn(
        (
            1,
            args.latent_channels,
            output_size[0] // divisor,
            output_size[1] // divisor,
            output_size[2], # !!!注意！！！
        ),
        device=device,
        )

    logger.info(f"noise: {noise.device}, {noise.dtype}, {type(noise)}")

    image = noise # [1, 4, 40, 40, 8]
    noise_scheduler = define_instance(args, "noise_scheduler")
    noise_scheduler.set_timesteps(num_inference_steps=args.diffusion_unet_inference["num_inference_steps"])
    if not use_vqgan:
        recon_model = ReconModel(autoencoder=autoencoder, scale_factor=scale_factor).to(device)
    autoencoder.eval()
    unet.eval()

    with torch.amp.autocast("cuda", enabled=True):
        
        if cond is not None:
            cond = apply_autoencoder(autoencoder, cond, use_vqgan=use_vqgan)
            '''
            1. encode + norm
            cond max: tensor(0.7480, device='cuda:0', dtype=torch.float16)
            cond min: tensor(0.1981, device='cuda:0', dtype=torch.float16)
            2. encode
            cond max: tensor(19.5469, device='cuda:0', dtype=torch.float16)
            cond min: tensor(-25.0938, device='cuda:0', dtype=torch.float16)
            3. vae encode
            cond max: tensor(5.1797, device='cuda:0')
            cond min: tensor(-5.0187, device='cuda:0')
            4. encode + norm2
            cond max: tensor(0.4424, device='cuda:0', dtype=torch.float16)
            cond min: tensor(-0.6699, device='cuda:0', dtype=torch.float16)
            5. [-1, 1]
            cond max: tensor(0.3760, device='cuda:0', dtype=torch.float16)
            cond min: tensor(-0.7725, device='cuda:0', dtype=torch.float16)
            '''
            # print('cond max:',cond.max())
            # print('cond min:',cond.min())
            if scale_factor_cond is None or use_vqgan:
                scale_factor_cond = 1.0
            cond = cond * scale_factor_cond
        for t in tqdm(noise_scheduler.timesteps, ncols=110):
            # image: torch.Size([1, 4, x/4, y/4, z/4])
            # t: tensor(900)->800, 700,...
            # torch.Size([1, 4, 40, 40, 8])
            if cond is not None:
                model_output = unet(
                    x=torch.cat((image,cond), dim=1),
                    timesteps=torch.Tensor((t,)).to(device),
                    top_region_index_tensor=top_region_index_tensor,
                    bottom_region_index_tensor=bottom_region_index_tensor,
                    spacing_tensor=spacing_tensor,
                )
            else:
                model_output = unet(
                    x=image,
                    timesteps=torch.Tensor((t,)).to(device),
                    top_region_index_tensor=top_region_index_tensor,
                    bottom_region_index_tensor=bottom_region_index_tensor,
                    spacing_tensor=spacing_tensor,
                )
            # torch.Size([1, 4, 40, 40, 8])
            image, _ = noise_scheduler.step(model_output, t, image)
            # torch.Size([1, 4, 40, 40, 8])
        # synthetic_images = recon_model(image)
        synthetic_images = apply_autoencoder(recon_model if not use_vqgan else autoencoder, image, use_vqgan=use_vqgan, recon=True,scale_factor=scale_factor)
        
        '''
        1. recon:  norm + decode
        synthetic_images min tensor(-0.0344, device='cuda:0', dtype=torch.float16)
        synthetic_images max tensor(0.7402, device='cuda:0', dtype=torch.float16)
        2. recon:  decode
        synthetic_images min tensor(-0.1309, device='cuda:0', dtype=torch.float16)
        synthetic_images max tensor(1.0850, device='cuda:0', dtype=torch.float16)
        3. recon:  vae decode
        synthetic_images min tensor(-0.0775, device='cuda:0', dtype=torch.float16, grad_fn=<MinBackward1>)
        synthetic_images max tensor(0.9619, device='cuda:0', dtype=torch.float16, grad_fn=<MaxBackward1>)
        4. recon:  norm2 + decode
        synthetic_images min tensor(-0.0995, device='cuda:0', dtype=torch.float16)
        synthetic_images max tensor(1.0078, device='cuda:0', dtype=torch.float16)
        5. [-1,1]
        synthetic_images min tensor(-0.9346, device='cuda:0', dtype=torch.float16)
        synthetic_images max tensor(0.4253, device='cuda:0', dtype=torch.float16)
        '''
        # print('synthetic_images min',synthetic_images.min())
        # print('synthetic_images max',synthetic_images.max())

        # synthetic_images = sliding_window_inference(
        #     inputs=image,
        #     roi_size=(
        #         min(output_size[0] // divisor // 4 * 3, 96),
        #         min(output_size[1] // divisor // 4 * 3, 96),
        #         min(output_size[2] // divisor // 4 * 3, 96),
        #     ),
        #     sw_batch_size=1,
        #     predictor=recon_model,
        #     mode="gaussian",
        #     overlap=2.0 / 3.0, # 
        #     sw_device=device,
        #     device=device,
        # )
        # synthetic_images: torch.Size([1, 1, 160, 149, 32])
        # print('4.synthetic_images',synthetic_images.shape)
        data = synthetic_images.squeeze().squeeze().cpu().detach().numpy()

        # [0,1]->[-1000,1000]
        # a_min, a_max, b_min, b_max = -1000, 1000, 0, 1
        # data = (data - b_min) / (b_max - b_min) * (a_max - a_min) + a_min
        # data = np.clip(data, a_min, a_max)

        # return np.int16(data)
        if cond is not None: # 引入AC4可能会让输出偏离[0,1]
            # data = (data - data.min()) / (data.max() - data.min())
            
            # cond_img = apply_autoencoder(recon_model if not use_vqgan else autoencoder, cond, use_vqgan=use_vqgan, recon=True,scale_factor=scale_factor)
            # print('cond_img max:',cond_img.max())
            # print('cond_img min:',cond_img.min())
            '''
            1. recon:  norm + decode
            cond_img max: tensor(1.0762, device='cuda:0', dtype=torch.float16)
            cond_img min: tensor(-0.0878, device='cuda:0', dtype=torch.float16)
            2. recon:  decode
            cond_img max: tensor(1.0977, device='cuda:0', dtype=torch.float16)
            cond_img min: tensor(-0.0822, device='cuda:0', dtype=torch.float16)
            3. recon:  vae decode  
            cond_img max: tensor(1.1133, device='cuda:0', dtype=torch.float16)
            cond_img min: tensor(-0.1039, device='cuda:0', dtype=torch.float16)
            4. recon:  norm2 + decode
            cond_img max: tensor(1.0801, device='cuda:0', dtype=torch.float16)
            cond_img min: tensor(-0.0898, device='cuda:0', dtype=torch.float16)
            5. [-1, 1]
            cond_img max: tensor(1.0596, device='cuda:0', dtype=torch.float16)
            cond_img min: tensor(-0.1356, device='cuda:0', dtype=torch.float16)
            '''
            # cond_img = cond_img.squeeze().squeeze().cpu().detach().numpy()
            # cond_img = (cond_img - cond_img.min()) / (cond_img.max() - cond_img.min())
            # # cond_img =cond_img * 255.0
            # cond_img = (cond_img + 1.0) * 127.5
            # np.clip(cond_img, 0, 255)
            # cond_img=cond_img.astype(np.uint8) 
            # cond_img = np.transpose(cond_img, (2, 1, 0))  # [H, W, C]->[C, H, W]

            # Image.fromarray(cond_img[0]).save(os.path.join('./local_data/code/maisi/outputs/pred_diff_vq_tmp',
            #                                                 'cond_img.png'))


        data = data * 255.0
        # data = (data + 1.0) * 127.5
        data = np.clip(data, 0, 255)
        data = data.astype(np.uint8) 

        return data

def visual_saver_vae(
    epoch: int,
    data: np.ndarray,
    output_path: str,
    recon: np.ndarray,
    cond: np.ndarray=None,
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

    recon = recon[0,0].cpu().detach().numpy()
    recon = recon * 255.0
    np.clip(recon, 0, 255)
    recon=recon.astype(np.uint8)

    data = data[0,0].cpu().detach().numpy()
    data = data * 255.0
    np.clip(data, 0, 255)
    data=data.astype(np.uint8) 

    


    data = np.transpose(data, (2, 1, 0))  # [H, W, C]->[C, H, W]
    recon = np.transpose(recon, (2, 1, 0))
    

    # 【具体再改】
    # outputs = (outputs + 1.0) * 127.5
    # np.clip(outputs, 0, 255)
    # outputs=outputs.astype(np.uint8) 
    
    # [32, 160, 160]
    k = random.randint(0, data.shape[0]-1)
    Image.fromarray(data[k]).save(os.path.join(img_path, str(epoch).zfill(3)+'_'+str(k).zfill(2)+'.png'))
    imageio.mimsave(os.path.join(video_path, str(epoch).zfill(3)+'.mp4'), data, fps=6)

    if cond is not None:
        cond = cond[0,0].cpu().detach().numpy()
        cond = cond * 255.0
        np.clip(cond, 0, 255)
        cond=cond.astype(np.uint8)
        cond = np.transpose(cond, (2, 1, 0))
        Image.fromarray(cond[k]).save(os.path.join(img_path, str(epoch).zfill(3)+'_'+str(k).zfill(2)+'_cond.png'))
        imageio.mimsave(os.path.join(video_path, str(epoch).zfill(3)+'_cond.mp4'), cond, fps=6)

    Image.fromarray(recon[k]).save(os.path.join(img_path, str(epoch).zfill(3)+'_'+str(k).zfill(2)+'_recon.png'))
    imageio.mimsave(os.path.join(video_path, str(epoch).zfill(3)+'_recon.mp4'), recon, fps=6)


def visual_saver(
    epoch: int,
    data: np.ndarray,
    output_path: str,
    logger: logging.Logger,
    tensorboard_writer: SummaryWriter = None,
    cond: np.ndarray=None,
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
    Image.fromarray(data[k]).save(os.path.join(img_path, str(epoch).zfill(3)+'_'+str(k).zfill(2)+'.png'))

    imageio.mimsave(os.path.join(video_path, str(epoch).zfill(3)+'.mp4'), data, fps=6)

    if cond is not None:
        if isinstance(cond, torch.Tensor):
            cond = cond[0,0].cpu().detach().numpy()
            cond = cond * 255.0
            cond = np.clip(cond, 0, 255)
            cond = cond.astype(np.uint8)
        cond = np.transpose(cond, (2, 1, 0))


        Image.fromarray(cond[k]).save(os.path.join(img_path, str(epoch).zfill(3)+'_'+str(k).zfill(2)+'_cond.png'))
        imageio.mimsave(os.path.join(video_path, str(epoch).zfill(3)+'_cond.mp4'), cond, fps=6)

    logger.info(f"Saved {output_path}.")

    if tensorboard_writer is not None:
        tensorboard_writer.add_image(
                    "val_img",
                    torch.from_numpy(data[k]).unsqueeze(0),
                    epoch,
                )
        if cond is not None:
            tensorboard_writer.add_image(
                    "val_cond",
                    torch.from_numpy(cond[k]).unsqueeze(0),
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
    lb = lb.astype(np.uint8)

    src_img = src_img[0,0].cpu().detach().numpy()
    src_img = src_img * 255.0
    src_img = np.clip(src_img, 0, 255)
    src_img = src_img.astype(np.uint8)

    tgt_img = tgt_img[0,0].cpu().detach().numpy()
    tgt_img = tgt_img * 255.0
    tgt_img = np.clip(tgt_img, 0, 255)
    tgt_img = tgt_img.astype(np.uint8)

    # [H, W, 32]->[32, H, W]
    lb = np.transpose(lb, (2, 1, 0))
    src_img = np.transpose(src_img, (2, 1, 0))
    tgt_img = np.transpose(tgt_img, (2, 1, 0))

    k = random.randint(0, src_img.shape[0]-1)
    # [32, 160, 160]
    if isinstance(data, list):
        for i,img in enumerate(data):
            if len(img.shape)==5:
                img = img[0,0].cpu().detach().numpy()
                img = img * 255.0
                img = np.clip(img, 0, 255)
                img = img.astype(np.uint8) 
            img = np.transpose(img, (2, 1, 0))
            Image.fromarray(img[k]).save(os.path.join(img_path, str(epoch).zfill(3)+'_'+str(k).zfill(2)+'_'+str(i+1).zfill(1)+'.png'))
    else:
        if len(data.shape)==5:
            data = data[0,0].cpu().detach().numpy()
            data = data * 255.0
            data = np.clip(data, 0, 255)
            data=data.astype(np.uint8) 
        data = np.transpose(data, (2, 1, 0))
        
        Image.fromarray(data[k]).save(os.path.join(img_path, str(epoch).zfill(3)+'_'+str(k).zfill(2)+'.png'))
    
    
    
    Image.fromarray(lb[k]).save(os.path.join(img_path, str(epoch).zfill(3)+'_'+str(k).zfill(2)+'_lb.png'))
    Image.fromarray(src_img[k]).save(os.path.join(img_path, str(epoch).zfill(3)+'_'+str(k).zfill(2)+'_src.png'))
    Image.fromarray(tgt_img[k]).save(os.path.join(img_path, str(epoch).zfill(3)+'_'+str(k).zfill(2)+'_tgt.png'))

    # imageio.mimsave(os.path.join(video_path, str(epoch).zfill(3)+'.mp4'), data, fps=6)
    # imageio.mimsave(os.path.join(video_path, str(epoch).zfill(3)+'_lb.mp4'), lb, fps=6)
    # imageio.mimsave(os.path.join(video_path, str(epoch).zfill(3)+'_src.mp4'), src_img, fps=6)
    # imageio.mimsave(os.path.join(video_path, str(epoch).zfill(3)+'_tgt.mp4'), tgt_img, fps=6)

    logger.info(f"Saved {output_path}.")

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

def save_image(
    data: np.ndarray,
    output_size: tuple,
    out_spacing: tuple,
    output_path: str,
    logger: logging.Logger,
) -> None:
    """
    Save the generated synthetic image to a file.

    Args:
        data (np.ndarray): Synthetic image data.
        output_size (tuple): Output size of the image.
        out_spacing (tuple): Spacing of the output image.
        output_path (str): Path to save the output image.
        logger (logging.Logger): Logger for logging information.
    """
    out_affine = np.eye(4)
    for i in range(3):
        out_affine[i, i] = out_spacing[i]

    new_image = nib.Nifti1Image(data, affine=out_affine)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    nib.save(new_image, output_path)
    logger.info(f"Saved {output_path}.")


@torch.inference_mode()
def diff_model_infer(env_config_path: str, model_config_path: str, model_def_path: str, num_gpus: int,sample: int) -> None:
    """
    Main function to run the diffusion model inference.

    Args:
        env_config_path (str): Path to the environment configuration file.
        model_config_path (str): Path to the model configuration file.
        model_def_path (str): Path to the model definition file.
    """
    args = load_config(env_config_path, model_config_path, model_def_path)
    local_rank, world_size, device = initialize_distributed(num_gpus)
    logger = setup_logging("inference")
    random_seed = set_random_seed(
        args.diffusion_unet_inference["random_seed"] + local_rank
        if args.diffusion_unet_inference["random_seed"]
        else None
    )
    logger.info(f"Using {device} of {world_size} with random seed: {random_seed}")

    output_size = tuple(args.diffusion_unet_inference["dim"])
    # out_spacing = tuple(args.diffusion_unet_inference["spacing"])
    out_spacing = None
    output_prefix = args.output_prefix
    ckpt_filepath = f"{args.model_dir}/{args.model_filename}"

    if local_rank == 0:
        logger.info(f"[config] ckpt_filepath -> {ckpt_filepath}.")
        logger.info(f"[config] random_seed -> {random_seed}.")
        logger.info(f"[config] output_prefix -> {output_prefix}.")
        logger.info(f"[config] output_size -> {output_size}.")
        # logger.info(f"[config] out_spacing -> {out_spacing}.")

    # check_input(None, None, None, output_size, out_spacing, None)

    autoencoder, unet, scale_factor = load_models(args, device, logger)
    num_downsample_level = max(
        1,
        (
            len(args.diffusion_unet_def["num_channels"])
            if isinstance(args.diffusion_unet_def["num_channels"], list)
            else len(args.diffusion_unet_def["attention_levels"])
        ),
    )
    divisor = 2 ** (num_downsample_level - 2)
    logger.info(f"num_downsample_level -> {num_downsample_level}, divisor -> {divisor}.")

    top_region_index_tensor, bottom_region_index_tensor, spacing_tensor = None, None, None
    dataset_infer = SingleDataset(root_dir='./local_data/AC3_AC4/', train_dataset='AC4_inputs.h5',length=sample)
    args.diffusion_unet_inference["num_inference_steps"]=1000
    for i in range(sample):
        cond_infer = dataset_infer[i]['image'].unsqueeze(0).to(device) # cond:[1, 1, 160, 160, 32]
        data = run_inference(
            args,
            device,
            autoencoder,
            unet,
            scale_factor,
            top_region_index_tensor,
            bottom_region_index_tensor,
            spacing_tensor,
            output_size,
            divisor,
            logger,
            cond=cond_infer,
            # scale_factor_cond= scale_factor_cond,
        )

        # timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        # output_path = "{0}/{1}_seed{2}_size{3:d}x{4:d}x{5:d}_spacing{6:.2f}x{7:.2f}x{8:.2f}_{9}_rank{10}.nii.gz".format(
        #     args.output_dir,
        #     output_prefix,
        #     random_seed,
        #     output_size[0],
        #     output_size[1],
        #     output_size[2],
        #     out_spacing[0],
        #     out_spacing[1],
        #     out_spacing[2],
        #     timestamp,
        #     local_rank,
        # )
        # save_image(data, output_size, out_spacing, output_path, logger)

        infer_path = args.infer_path
        if not os.path.exists(infer_path):
            os.makedirs(infer_path)
        # visual_saver(0, data, infer_path, logger)

        cond_infer = cond_infer.squeeze().cpu().detach().numpy()
        cond_infer = cond_infer * 255.0
        np.clip(cond_infer, 0, 255)
        cond_infer=cond_infer.astype(np.uint8) 

        visual_saver(i, data, infer_path, logger, cond=cond_infer)

        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diffusion Model Inference")
    parser.add_argument(
        "--env_config",
        type=str,
        default="./configs/environment_maisi_diff_model.json",
        help="Path to environment configuration file",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="./configs/config_maisi_diff_model.json",
        help="Path to model training/inference configuration",
    )
    parser.add_argument(
        "--model_def",
        type=str,
        default="./configs/config_maisi.json",
        help="Path to model definition file",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for distributed inference",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=10,
        help="Number of GPUs to use for distributed inference",
    )

    args = parser.parse_args()
    diff_model_infer(args.env_config, args.model_config, args.model_def, args.num_gpus,args.sample)
