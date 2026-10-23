# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# &nbsp;&nbsp;&nbsp;&nbsp;http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# # MAISI Inference Script
import argparse
import json
import logging
import os
import tempfile
import sys

import random
import monai
import torch
import numpy as np
from PIL import Image
import h5py

import torch.distributed as dist
from monai.apps import download_url
from monai.config import print_config
from monai.transforms import LoadImage, Orientation
from monai.utils import RankFilter,set_determinism
from scripts.sample import LDMSampler, check_input
from scripts.utils import define_instance,setup_ddp
from scripts.utils_plot import find_label_center_loc, get_xyz_plot, show_image
from .sample import ldm_conditional_sample_translated_image, ldm_conditional_sample_translated_image_lb



from monai.data import DataLoader
from data.dataset_lb import CondTargetDataset_lb

import warnings
warnings.filterwarnings("ignore")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

# 【舍弃，用分成单GPU，inference_per_gpu.py即可】
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
    # parser.add_argument(
    #     "-e",
    #     "--environment-file",
    #     default="./configs/environment.json",
    #     help="environment json file that stores environment path",
    # )
    parser.add_argument(
        "-c",
        "--config-file",
        default="./configs/config_maisi.json",
        help="config json file that stores network hyper-parameters",
    )
    parser.add_argument(
        "-i",
        "--inference-file",
        default="./configs/config_infer.json",
        help="config json file that stores inference hyper-parameters",
    )
    parser.add_argument(
        "-s",
        "--random-seed",
        default=None,
        help="random seed, can be None or int",
    )
    parser.add_argument("-g", "--gpus", default=1, type=int, help="number of gpus per node")
    args = parser.parse_args()
    # Step 0: configuration
    logger = logging.getLogger("maisi.inference")
    use_ddp = args.gpus > 1
    if use_ddp:
        rank = int(os.environ["LOCAL_RANK"]) # 0,...,7
        world_size = int(os.environ["WORLD_SIZE"]) # 8
        device = setup_ddp(rank, world_size)
        logger.addFilter(RankFilter())
    else:
        rank = 0
        world_size = 1
        device = torch.device(f"cuda:{rank}")
    
    

    # ## Set deterministic training for reproducibility
    if args.random_seed is not None:
        set_determinism(seed=args.random_seed)

    # with open(args.environment_file, "r") as env_file:
    #     env_dict = json.load(env_file)
    with open(args.config_file, "r") as config_file:
        config_dict = json.load(config_file)
    with open(args.inference_file, "r") as inference_config_file:
        inference_config_dict = json.load(inference_config_file)

    # for k, v in env_dict.items():
    #     setattr(args, k, v)
    for k, v in config_dict.items():
        setattr(args, k, v)
    for k, v in inference_config_dict.items():
        setattr(args, k, v)

    images_per_gpu = args.samples // world_size # 2500
    start_list=[1123,3623,6122,8623] # 最后结尾的已有的
    end_list=[2500,5000,7500,10000] #  # 先固定，之后要再加

    start_idx = start_list[rank]
    end_idx = end_list[rank]

    # print('rank, start_idx, end_idx:',rank, start_idx, end_idx)


    # Step 1: set data loader [生成的一共数量是samples]
    dataset = CondTargetDataset_lb(src_dir='./local_data/CREMI', src_dataset='cremiC_inputs_interp.h5',\
                            tgt_dir='./local_data/AC3_AC4/', tgt_dataset='AC4_inputs.h5', \
                            length=(end_idx-start_idx),strong_aug=False,src_aug=False)
    dataloader = DataLoader(dataset, batch_size=1, \
                              num_workers=4, shuffle=True, drop_last=True)
    if rank==0:
        print("Network definition and inference inputs have been loaded.")

    # ## Initialize networks and noise scheduler, then load the trained model weights.
    # The networks and noise scheduler are defined in `config_file`. We will read them in and load the model weights.
    noise_scheduler = define_instance(args, "noise_scheduler")

    device = torch.device("cuda")

    # Step 2.1: Load VAE or VQGAN
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


    # Step 2.2: Load diffusion Model
    unet = define_instance(args, "diffusion_unet_def").to(device)
    diffusion_model_ckpt = torch.load(args.trained_diffusion_path, map_location=device)
    unet.load_state_dict(diffusion_model_ckpt["unet_state_dict"])
    scale_factor = diffusion_model_ckpt["scale_factor"]



    # Step 2.3: Load ControlNet
    controlnet = define_instance(args, "controlnet_def").to(device)
    controlnet_ckpt = torch.load(args.trained_controlnet_path, map_location=device)
    monai.networks.utils.copy_model_state(controlnet, unet.state_dict())
    controlnet.load_state_dict(controlnet_ckpt["controlnet_state_dict"], strict=True)

    if rank==0:
        print("All the trained model weights have been loaded.")

    # 设置固定参数
    output_size = args.output_size
    num_downsample_level = max(
        1,
        (
            len(args.diffusion_unet_def["num_channels"])
            if isinstance(args.diffusion_unet_def["num_channels"], list)
            else len(args.diffusion_unet_def["attention_levels"])
        ),
    )
    divisor = 2 ** (num_downsample_level - 2)

    top_region_index_tensor, bottom_region_index_tensor, spacing_tensor = None, None, None
    if args.use_vqgan or args.use_144:
        latent_shape = (args.latent_channels, output_size[0] // 4, output_size[1] // 4, output_size[2])
    else:
        latent_shape = (args.latent_channels, output_size[0] // 4, output_size[1] // 4, output_size[2] // 4)
    
    output_path = args.inference_dir
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    ## 
    controlnet.eval()
    unet.eval()
    autoencoder.eval()

    torch.cuda.empty_cache()
    # if use_ddp:
    #     dist.barrier()

    for step, batch in enumerate(dataloader):
        src_img = batch["image"].to(device) # src img
        # 目前的是[inputs, conds]
        tgt_img = batch["cond"].to(device) # target img无增强
        lbs = batch["lb"].to(device) # src lb
        sample_num = batch["num"]
        # weightmap = batch["weight"].to(device) # src weight_map
        with torch.amp.autocast("cuda"):
            # inputs = apply_autoencoder(autoencoder, src_img, use_vqgan=args.use_vqgan)
            conds = apply_autoencoder(autoencoder, tgt_img, use_vqgan=args.use_vqgan)
            # if not args.use_vqgan:
            #     inputs = inputs * scale_factor


        torch.cuda.empty_cache()
        data = ldm_conditional_sample_translated_image_lb(
            autoencoder,
            unet,
            controlnet,
            noise_scheduler,
            scale_factor,
            device,
            conds, # 这是压缩后的tgt img
            lbs, # src lb 没有压缩
            top_region_index_tensor,
            bottom_region_index_tensor,
            spacing_tensor,
            latent_shape=latent_shape,
            output_size=output_size,
            noise_factor=1.0,
            num_inference_steps=args.num_inference_steps,
            # reduce it when GPU memory is limited
            autoencoder_sliding_window_infer_size=args.autoencoder_sliding_window_infer_size,
            for_show=False,
            use_vqgan=args.use_vqgan,
        )

        # 【其他部分到时候分割模型训练时临时读取？】

        image_filename = f"sample_{step+1+start_idx:05d}_{sample_num[0].item()}_{sample_num[1].item()}_{sample_num[2].item()}.{args.save_type}"


        # 【还需确定：1.保存成uint8 or 直接float】
        # 2. 不做增强，因为会改变lb，但不保存lb
        # data = (data-data.min())/(data.max()-data.min())
        data = data[0,0].cpu().detach().numpy()
        data = np.transpose(data, (2, 1, 0)) 

        
        

        # [1,1,160,160,18]
        if args.save_type =="png": # 莫名报错，4gpu
            
            data = data * 255.0
            np.clip(data, 0, 255)
            data=data.astype(np.uint8) 
            

            k = random.randint(0, data.shape[0]-1)
            Image.fromarray(data[k]).save(os.path.join(output_path, image_filename))

            src_img = src_img[0,0].cpu().detach().numpy()
            src_img = src_img * 255.0
            np.clip(src_img, 0, 255)
            src_img=src_img.astype(np.uint8) 
            src_img = np.transpose(src_img, (2, 1, 0)) 
            Image.fromarray(src_img[k]).save(os.path.join(output_path, f"src_{step+1+rank*images_per_gpu:05d}.png"))

            tgt_img = tgt_img[0,0].cpu().detach().numpy()
            tgt_img = tgt_img * 255.0
            np.clip(tgt_img, 0, 255)
            tgt_img=tgt_img.astype(np.uint8)
            tgt_img = np.transpose(tgt_img, (2, 1, 0))
            Image.fromarray(tgt_img[k]).save(os.path.join(output_path, f"tgt_{step+1+rank*images_per_gpu:05d}.png"))
        elif args.save_type =="h5":
            
            # print('data type:',data.dtype) # 
            # float16 654.04KB  6.54GB
            # uint8 353.19KB  3.53GB
            # 10000个样本
            f = h5py.File(os.path.join(output_path, image_filename), 'w')
            f.create_dataset('main', data=data, dtype=data.dtype, compression='gzip')
            f.close()

            # data = data * 255.0
            # np.clip(data, 0, 255)
            # data=data.astype(np.uint8) 

            # f = h5py.File(os.path.join(output_path, image_filename), 'w')
            # f.create_dataset('main', data=data, dtype=data.dtype, compression='gzip')
            # f.close()            
        torch.cuda.empty_cache()
    if use_ddp:
        dist.barrier()
        dist.destroy_process_group()




if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    torch.cuda.reset_peak_memory_stats()
    main()
    peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)  # Convert to GB
    print(f"Peak GPU memory usage: {peak_memory_gb:.2f} GB")
