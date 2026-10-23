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

from monai.utils import RankFilter,set_determinism

from scripts.utils import define_instance,setup_ddp
from scripts.sample import ldm_conditional_sample_translated_image_lb


from monai.data import DataLoader
from data.dataset_lb import CondTargetDataset_lb

import warnings
warnings.filterwarnings("ignore")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def apply_autoencoder(
        autoencoder,
        inputs):

    z = autoencoder.encode(inputs.permute(0,1,4,3,2),
                                            quantize=False, include_embeddings=True)

    z = ((z - autoencoder.codebook.embeddings.min()) /
                    (autoencoder.codebook.embeddings.max() -
                    autoencoder.codebook.embeddings.min())) * 2.0 - 1.0

    return z.permute(0,1,4,3,2)

    
def main(args):
    
    logger = logging.getLogger("maisi.inference")
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
    
    

    # ## Set deterministic training for reproducibility
    if args.random_seed is not None:
        set_determinism(seed=args.random_seed)


    with open(args.config_file, "r") as config_file:
        config_dict = json.load(config_file)
    with open(args.inference_file, "r") as inference_config_file:
        inference_config_dict = json.load(inference_config_file)


    for k, v in config_dict.items():
        setattr(args, k, v)
    for k, v in inference_config_dict.items():
        setattr(args, k, v)

    images_per_gpu = args.samples // world_size
    start_list=[0,1500,3000,4500,6000,7500,9000,10500]
    end_list=[1500,3000,4500,6000,7500,9000,10500,12000] 

    start_idx = start_list[args.num]
    end_idx = end_list[args.num]



    # dataset = CondTargetDataset_lb(src_dir='./local_data/CREMI', src_dataset='cremiC_inputs_interp.h5',\
    #                         tgt_dir='./local_data/AC3_AC4/', tgt_dataset='AC4_inputs.h5', \
    #                         length=(end_idx-start_idx),strong_aug=False,src_aug=False)
    dataset = CondTargetDataset_lb(src_dir='./local_data/AC3_AC4/', src_dataset='AC4_inputs.h5',\
                            tgt_dir='./local_data/CREMI/', tgt_dataset='cremiC_inputs_interp.h5', \
                            length=(end_idx-start_idx),strong_aug=False,src_aug=False)
    dataloader = DataLoader(dataset, batch_size=1, \
                              num_workers=4, shuffle=False, drop_last=True)

    noise_scheduler = define_instance(args, "noise_scheduler")

    device = torch.device("cuda")

    # Step 2.1: Load autoencoder
    autoencoder = define_instance(args, "vqgan_def").to(device)
    autoencoder = autoencoder.load_from_checkpoint(args.trained_autoencoder_path,
                embedding_dim=args.latent_channels,
                n_codes=args.vqgan_def["n_codes"],
                n_hiddens=args.vqgan_def["n_hiddens"],
                downsample=args.vqgan_def["downsample"],
                image_channels=args.image_channels).to(device)


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


    output_size = args.output_size
    num_downsample_level = max(
        1,
        (
            len(args.diffusion_unet_def["num_channels"])
            if isinstance(args.diffusion_unet_def["num_channels"], list)
            else len(args.diffusion_unet_def["attention_levels"])
        ),
    )

    top_region_index_tensor, bottom_region_index_tensor, spacing_tensor = None, None, None

    latent_shape = (args.latent_channels, output_size[0] // 4, output_size[1] // 4, output_size[2])

    output_path = args.inference_dir
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    controlnet.eval()
    unet.eval()
    autoencoder.eval()

    torch.cuda.empty_cache()


    for step, batch in enumerate(dataloader):
        src_img = batch["image"].to(device)
        tgt_img = batch["cond"].to(device)
        lbs = batch["lb"].to(device)
        sample_num = batch["num"]

        with torch.amp.autocast("cuda"):
            conds = apply_autoencoder(autoencoder, tgt_img)



        torch.cuda.empty_cache()
        data = ldm_conditional_sample_translated_image_lb(
            autoencoder,
            unet,
            controlnet,
            noise_scheduler,
            scale_factor,
            device,
            conds,
            lbs,
            top_region_index_tensor,
            bottom_region_index_tensor,
            spacing_tensor,
            latent_shape=latent_shape,
            output_size=output_size,
            noise_factor=1.0,
            num_inference_steps=args.num_inference_steps,
            autoencoder_sliding_window_infer_size=args.autoencoder_sliding_window_infer_size,
            for_show=False,
            use_vqgan=args.use_vqgan,
        )


        image_filename = f"sample_{step+1+start_idx:05d}_{sample_num[0].item()}_{sample_num[1].item()}_{sample_num[2].item()}.{args.save_type}"

        data = data[0,0].cpu().detach().numpy()
        data = np.transpose(data, (2, 1, 0)) 

        
        if args.save_type =="png":
            
            data = data * 255.0
            data = np.clip(data, 0, 255)
            data = data.astype(np.uint8) 
            

            k = 0 # random.randint(0, data.shape[0]-1)
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
            

            f = h5py.File(os.path.join(output_path, image_filename), 'w')
            f.create_dataset('main', data=data, dtype=data.dtype, compression='gzip')
            f.close()
       
        torch.cuda.empty_cache()
    if use_ddp:
        dist.barrier()
        dist.destroy_process_group()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="maisi.controlnet.training")
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
    parser.add_argument("-n", "--num", default=0, type=int, help="number gpu: 0-7")
    args = parser.parse_args()
    main(args)
