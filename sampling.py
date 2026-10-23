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

from monai.data import DataLoader
from monai.utils import RankFilter,set_determinism
from monai.networks.utils import copy_model_state


from scripts.utils import define_instance,setup_ddp

from scripts.utils_self import load_autoencoder, apply_autoencoder, load_model, generate_sample, generate_sample_translation
from scripts.utils_eval_self import calculate_imagebased, eval_on_piq
from scripts.sample import ldm_conditional_sample_translated_image_lb


from data.dataset_lb import CondTargetDataset_lb
from data.provider_valid import ValidDataset_seg, ValidDataset_translation

import warnings
warnings.filterwarnings("ignore")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"





def sampling_debug(args, device, logger):
    dataset = ValidDataset_seg(root_dir='./local_data/AC3_AC4/', 
                                     train_dataset='AC4_inputs.h5',
                                     crop_size=[18,160,160],recon_all=False)
    dataloader = DataLoader(dataset, batch_size=1, \
                            num_workers=4, shuffle=False)
    
    autoencoder = load_autoencoder(args, device, logger)

    model, _ = load_model(args, device, logger, infer=True, debug=True)

    controlnet = define_instance(args, "controlnet_def").to(device)
    controlnet_ckpt = torch.load(args.trained_controlnet_path, map_location=device, weights_only=True)

    copy_model_state(controlnet, controlnet_ckpt["controlnet_state_dict"])

    logger.info(f"Pretrained checkpoint {args.trained_controlnet_path} loaded.")

    output_path = os.path.join(args.inference_dir, f"{args.prefix}cfg{args.cfg}_steps{args.steps}_t{args.t}")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    log_path = os.path.join(output_path, "log")
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    f_txt = open(os.path.join(log_path, 'valid.txt'), 'a')
    f_txt.write(f"NUM:{args.num}, CFG:{args.cfg}, timesteps:{args.steps}, generate_t_start:{args.t}")
    f_txt.write('\n')


    controlnet.eval()
    model.eval()
    autoencoder.eval()


    torch.cuda.empty_cache()

    real_list=[]
    recon_list=[]
    ssim_list,ssim_piq_list,msssim_list,psnr_list,mse_list=[],[],[],[],[]

    for step, batch in enumerate(dataloader):
        x = batch[0].permute(0, 1, 4, 3, 2).to(device)
        labels = batch[1].permute(0, 1, 4, 3, 2).to(device)


        with torch.no_grad():

            sample = generate_sample(
                model,
                controlnet,
                autoencoder,
                device,
                step+1,
                output_path,
                logger, 
                tensorboard_writer=None,
                use_vqgan=args.use_vqgan,
                visual = True if args.save_type =="png" else False, 
                image = x,
                labels = labels,
                t = args.t,
            )


        with torch.amp.autocast("cuda"):
            ssim,ssim_piq,msssim,psnr,mse=calculate_imagebased(x,sample)

        ssim_list.append(ssim)
        ssim_piq_list.append(ssim_piq)
        msssim_list.append(msssim)
        psnr_list.append(psnr)
        mse_list.append(mse)

        real_list.append(x)
        recon_list.append(sample)

        f_txt.write(f'SSIM:{ssim:.6f}, SSIM_piq:{ssim_piq:.6f}, MS-SSIM: {msssim:.6f}, PSNR: {psnr:.6f}, MSE: {mse:.6f}')
        f_txt.write('\n')

           
        torch.cuda.empty_cache()
        
    
    
    ssim_mean=sum(ssim_list)/len(ssim_list)
    ssim_piq_mean=sum(ssim_piq_list)/len(ssim_piq_list)
    msssim_mean=sum(msssim_list)/len(msssim_list)
    psnr_mean=sum(psnr_list)/len(psnr_list)
    mse_mean=sum(mse_list)/len(mse_list)

    f_txt.write('--------------------------------------------------------------')
    f_txt.write('\n')

    f_txt.write(f'[Total] SSIM:{ssim_mean:.6f}, SSIM_piq:{ssim_piq_mean:.6f}, MS-SSIM:{msssim_mean:.6f}, PSNR:{psnr_mean:.6f}, MSE:{mse_mean:.6f}')
    f_txt.write('\n')
    f_txt.flush()

    fid, kid, is_score, gs, msid, pre, rec = eval_on_piq(real_list, recon_list)

    f_txt.write(f'[Total]FID:{fid:.6f}, KID:{kid:.6f}, IS:{is_score:.6f}, GS:{gs:.6f}, MSID:{msid:.6f}, Precision:{pre:.6f}, Recall:{rec:.6f}')

    f_txt.close()




def sampling_debug_translation(args, device, logger):
    Twodata = ['./local_data/AC3_AC4/', './local_data/CREMI']
    Twodataset = ['AC4_inputs.h5', 'cremiC_inputs_interp.h5']

    dataset = ValidDataset_translation(src_dir = Twodata[0] if args.AtoC else Twodata[1], 
                                       src_dataset = Twodataset[0] if args.AtoC else Twodataset[1],
                            tgt_dir = Twodata[1] if args.AtoC else Twodata[0], 
                            tgt_dataset = Twodataset[1] if args.AtoC else Twodataset[0],
                            crop_size=[18,160,160], norm2=False, recon_all=False)
    dataloader = DataLoader(dataset, batch_size=1, \
                            num_workers=4, shuffle=True, drop_last=True)
    
    autoencoder = load_autoencoder(args, device, logger)

    model, _ = load_model(args, device, logger, infer=True, debug=True)

    controlnet = define_instance(args, "controlnet_def").to(device)
    controlnet_ckpt = torch.load(args.trained_controlnet_path, map_location=device, weights_only=True)

    copy_model_state(controlnet, controlnet_ckpt["controlnet_state_dict"])

    logger.info(f"Pretrained checkpoint {args.trained_controlnet_path} loaded.")

    lambda_style, lambda_struct = args.lambdalist
    output_path = os.path.join(args.inference_dir, f"{args.prefix}cfg{args.cfg}_steps{args.steps}_t{args.t}_lsy{lambda_style}_lst{lambda_struct}")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    controlnet.eval()
    model.eval()
    autoencoder.eval()


    torch.cuda.empty_cache()


    for step, batch in enumerate(dataloader):

        x = batch[0].permute(0, 1, 4, 3, 2).to(device)
        labels = batch[1].permute(0, 1, 4, 3, 2).to(device)
        tgt = batch[2].permute(0, 1, 4, 3, 2).to(device) 

        sample = generate_sample_translation(
            model,
            controlnet,
            autoencoder,
            device,
            step+1,
            output_path,
            logger, 
            tensorboard_writer=None,
            use_vqgan=args.use_vqgan,
            visual = True if args.save_type =="png" else False, 
            src_img = x,
            src_lb = labels,
            tgt_img = tgt,
            t = args.t,
            lambda_list = args.lambdalist,
        )


        if args.save_type =="h5":
            data = data[0,0].cpu().detach().numpy()
            data = np.transpose(data, (2, 1, 0)) 
            image_filename = f"sample_{step+1:05d}_{sample_num[0].item()}_{sample_num[1].item()}_{sample_num[2].item()}.{args.save_type}"
            f = h5py.File(os.path.join(output_path, image_filename), 'w')
            f.create_dataset('main', data=data, dtype=data.dtype, compression='gzip')
            f.close()


        torch.cuda.empty_cache()
        

    

def visual2(
    step,
    data,
    lb: np.ndarray,
    src_img: torch.Tensor,
    tgt_img: torch.Tensor,
    output_path: str,
    sample_index,
) -> None:
    from PIL import Image

    img_path=os.path.join(output_path,'img/')


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

    lb = np.transpose(lb, (2, 1, 0))
    src_img = np.transpose(src_img, (2, 1, 0))
    tgt_img = np.transpose(tgt_img, (2, 1, 0))

    k = 0
    z,y,x = sample_index

    data = np.transpose(data, (2, 1, 0))
    Image.fromarray(data[k]).save(os.path.join(img_path, f'sample_{step:04d}_z{z.item()}_y{y.item()}_x{x.item()}_ours.png'))
    Image.fromarray(lb[k]).save(os.path.join(img_path, f'sample_{step:04d}_z{z.item()}_y{y.item()}_x{x.item()}_lb.png'))
    Image.fromarray(src_img[k]).save(os.path.join(img_path, f'sample_{step:04d}_z{z.item()}_y{y.item()}_x{x.item()}_src.png'))
    Image.fromarray(tgt_img[k]).save(os.path.join(img_path, f'sample_{step:04d}_z{z.item()}_y{y.item()}_x{x.item()}_tgt.png'))


def sampling_translation(args, device, logger):
    if args.no_split:
        data_len = args.samples_num
        start_idx = 0
    else:
        start_list=[0,1500,3000,4500,6000,7500,9000,10500]
        end_list=[1500,3000,4500,6000,7500,9000,10500,12000]

        start_idx = start_list[args.num]
        data_len = end_list[args.num] - start_list[args.num]

    Twodata = ['./local_data/AC3_AC4/', './local_data/CREMI']
    Twodataset = ['AC4_inputs.h5', 'cremiC_inputs_interp.h5']

    dataset = CondTargetDataset_lb(src_dir = Twodata[0] if args.AtoC else Twodata[1], 
                                       src_dataset = Twodataset[0] if args.AtoC else Twodataset[1],
                            tgt_dir = Twodata[1] if args.AtoC else Twodata[0], 
                            tgt_dataset = Twodataset[1] if args.AtoC else Twodataset[0],
                            length=data_len,strong_aug=False,src_aug=False)
    dataloader = DataLoader(dataset, batch_size=1, \
                            num_workers=4, shuffle=True, drop_last=True)
    

    autoencoder = load_autoencoder(args, device, logger)

    model, _ = load_model(args, device, logger, infer=True, debug=True)

    controlnet = define_instance(args, "controlnet_def").to(device)
    controlnet_ckpt = torch.load(args.trained_controlnet_path, map_location=device, weights_only=True)

    copy_model_state(controlnet, controlnet_ckpt["controlnet_state_dict"])

    logger.info(f"Pretrained checkpoint {args.trained_controlnet_path} loaded.")


    output_path = os.path.join(args.inference_dir)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    

    controlnet.eval()
    model.eval()
    autoencoder.eval()


    torch.cuda.empty_cache()


    for step, batch in enumerate(dataloader):
        x = batch["image"].to(device)
        tgt = batch["cond"].to(device) 
        labels = batch["lb"].to(device)
        sample_index = batch["num"]


        sample = generate_sample_translation(
            model,
            controlnet,
            autoencoder,
            device,
            step+1,
            output_path,
            logger, 
            tensorboard_writer=None,
            use_vqgan=args.use_vqgan,
            visual = False,
            src_img = x,
            src_lb = labels,
            tgt_img = tgt,
            t = args.t,
            lambda_list = args.lambdalist,
        )

        data = sample.squeeze().squeeze().cpu().detach().numpy()
        data = data * 255.0
        data = np.clip(data, 0, 255)
        data=data.astype(np.uint8) 

        visual2(step+1,data, labels, x, tgt, output_path,sample_index)



        if args.save_type =="h5":
            sample = sample[0,0].cpu().detach().numpy()
            sample = np.transpose(sample, (2, 1, 0)) 
            image_filename = f"sample_{step+1+start_idx:05d}_{sample_index[0].item()}_{sample_index[1].item()}_{sample_index[2].item()}.{args.save_type}"
            f = h5py.File(os.path.join(output_path, image_filename), 'w')
            f.create_dataset('main', data=sample, dtype=sample.dtype, compression='gzip')
            f.close()


        torch.cuda.empty_cache()
        
    



def main(args):
    
    logger = logging.getLogger("Inference")
    use_ddp = args.gpus > 1
    if use_ddp:
        rank = int(os.environ["LOCAL_RANK"]) 
        world_size = int(os.environ["WORLD_SIZE"])
        device = setup_ddp(rank, world_size)
        logger.addFilter(RankFilter())
    else: # this
        rank = 0
        world_size = 1
        device = torch.device(f"cuda:{rank}")
    

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


    device = torch.device("cuda")


    if args.mode == "debug-ac4":
        sampling_debug(args, device, logger)
    elif args.mode == "debug-c2ac4":
        sampling_debug_translation(args, device, logger)
    elif args.mode == "infer":
        sampling_translation(args, device, logger) 
    elif args.mode == "infer2":
        sampling_translation(args, device, logger)        
        






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flow Matching Translation")
    parser.add_argument(
        "-c",
        "--config-file",
        default="./configs/flow/config_flow.json",
        help="config json file that stores network hyper-parameters",
    )
    parser.add_argument(
        "-i",
        "--inference-file",
        default="./configs/flow/config_flow_matching_infer.json",
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
    # ---------------
    parser.add_argument("--samples_num", default=10000, type=int, help="total generate amount")
    parser.add_argument("--no_split", action='store_true', help="infer: generating [samples_num] using on gpu or multigpu")

    parser.add_argument("--inference_dir", default="./local_data/samples_flow_a2c_tmp")
    parser.add_argument("--prefix", default="", help="[debug]remember to include _ : output_path/prefix_param")
    
    parser.add_argument("--mode", default="infer", help="debug-ac4, debug-c2ac4, infer, infer2")
    parser.add_argument("--save_type", default="png", help="png,h5 ")

    # ---------------
    parser.add_argument("--cfg", default=2.9, type=float, help="Classifier-Free Guidance Weight")
    parser.add_argument("--steps", default=50, type=int, help="num_sampling_steps")
    parser.add_argument("-t", default=0.875, type=float, help="sampling start time(0-1): 1 is clean image, 0 is noise")
    parser.add_argument("--w", default=50, type=int, help="num_sampling_steps")
    parser.add_argument("--lambdalist", default=[0.0, 0.0], nargs=2, type=float, metavar=("DAG", "SCG"), help="cosine-similarity weight, then pooled-structure weight")
    args = parser.parse_args()
    main(args)
