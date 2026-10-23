import os
import cv2
import h5py
import yaml
import torch
import argparse
import numpy as np
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
import time
import random
from tqdm import tqdm
from networks_copy.seg.model_superhuman import UNet_PNI
from data.provider_valid import ValidDataset_seg

from utils.show import draw_fragments_3d
from utils.fragment import watershed, relabel

import glob
from scripts.utils_eval_self import calculate_imagebased, eval_on_piq


import warnings
warnings.filterwarnings("ignore")

def _set_seed(seed):
    """设置所有相关的随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def eval_on_samples(args):

    if args.source_data == "cremi-c":
        source_data_path = "./local_data/CREMI/cremiC_inputs_interp.h5"
        data_split = 100
    elif args.source_data == "ac4":
        source_data_path = "./local_data/AC3_AC4/AC4_inputs.h5"
        data_split = 80

    _set_seed(args.seed)
    padding = args.padding

    random_crop = args.random_crop
    cal_psnr_and_ssim = args.cal_psnr_and_ssim
    eval_all = args.eval_all

    

    f_raw = h5py.File(source_data_path, 'r')
    data = f_raw['main'][:data_split]

    log_path = args.log_path
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    f_txt = open(os.path.join(log_path, f'{args.log_name}.txt'), 'a')

    all_files = sorted(glob.glob(os.path.join(args.sample_path, "*.h5")))
    if eval_all:
        selected_files = all_files
    else:
        selected_files = random.sample(all_files, args.count_num)

    
    real_list=[]
    recon_list=[]

    if cal_psnr_and_ssim: # # SSIM 结构相似度，是和src算
        ssim_list,ssim_piq_list,msssim_list,psnr_list,psnr_piq_list,mse_list=[],[],[],[],[],[]

    for file_path in selected_files: # 一定量的生成图
        # --- 1、读取生成图 (Sample) ---
        with h5py.File(file_path, 'r') as f:
            sample = f['main'][:] # [0,1]
            crop_size = sample.shape
            # LDM  # 注意padding去掉

            sample = sample.astype(np.float32) # /255.0 
            
            
        
        # --- 2、解析文件名坐标 ---
        
        if random_crop:
            start_z = random.randint(0, data.shape[0] - crop_size[0])
            start_y = random.randint(0, data.shape[1] - crop_size[1])
            start_x = random.randint(0, data.shape[2] - crop_size[2])
        else:
            # sample_00003_Z_Y_X.h5 -> 取 Z Y X
            filename = os.path.basename(file_path)
            base_name = os.path.splitext(filename)[0]
            parts = base_name.split("_")
            orig_z, orig_y, orig_x = map(int, parts[2:5])
            
            start_z = orig_z
            start_y = orig_y + padding
            start_x = orig_x + padding


        x = data[
                start_z : start_z + crop_size[0],
                start_y : start_y + crop_size[1],
                start_x : start_x + crop_size[2]
            ]
        x = x.astype(np.float32) / 255.0


        x = torch.tensor(x).unsqueeze(0).unsqueeze(0) #.permute(0, 1, 4, 3, 2) 
        sample = torch.tensor(sample).unsqueeze(0).unsqueeze(0) #.permute(0, 1, 4, 3, 2) 
        if args.clip:
            sample = torch.clamp(sample, 0.0, 1.0)
        # [1, 1, 18, 160, 160] =》 [1, 1, 160, 160, 18]
            
        if cal_psnr_and_ssim:
            with torch.amp.autocast("cuda"):
                ssim,ssim_piq,msssim,psnr,psnr_piq,mse=calculate_imagebased(x,sample, use_piq=args.piq,  transpose=True)

            ssim_list.append(ssim)
            ssim_piq_list.append(ssim_piq)
            msssim_list.append(msssim)
            psnr_list.append(psnr)
            psnr_piq_list.append(psnr_piq)
            mse_list.append(mse)

        real_list.append(x)
        recon_list.append(sample)

        # f_txt.write(f'SSIM:{ssim:.6f}, SSIM_piq:{ssim_piq:.6f}, MS-SSIM: {msssim:.6f}, PSNR: {psnr:.6f}, MSE: {mse:.6f}')
        # f_txt.write('\n')



        torch.cuda.empty_cache()
        

    f_txt.write('\n')
    f_txt.write('--------------------------------------------------------------\n')
    f_txt.write(f'seed: {args.seed}\n')
    f_txt.write(f'padding: {args.padding}\n')
    f_txt.write(f'random_crop: {args.random_crop}\n')
    f_txt.write(f'cal_psnr_and_ssim: {args.cal_psnr_and_ssim}\n')
    f_txt.write(f'eval_all: {args.eval_all}\n')
    f_txt.write(f'clip: {args.clip}\n')
    f_txt.write(f'piq: {args.piq}\n')

    if cal_psnr_and_ssim:
        ssim_mean=sum(ssim_list)/len(ssim_list)
        ssim_piq_mean=sum(ssim_piq_list)/len(ssim_piq_list)
        msssim_mean=sum(msssim_list)/len(msssim_list)
        psnr_mean=sum(psnr_list)/len(psnr_list)
        psnr_piq_mean=sum(psnr_piq_list)/len(psnr_piq_list)
        mse_mean=sum(mse_list)/len(mse_list)

        f_txt.write(f'[Total] SSIM:{ssim_mean:.6f}, SSIM_piq:{ssim_piq_mean:.6f}, MS-SSIM:{msssim_mean:.6f}, PSNR:{psnr_mean:.6f}, PSNR_piq:{psnr_piq_mean:.6f}, MSE:{mse_mean:.6f}')
        f_txt.write('\n')
        f_txt.flush()

    # (len,2048)
    fid, kid, is_score, gs, msid, pre, rec = eval_on_piq(real_list, recon_list)

    f_txt.write(f'[Total]FID:{fid:.6f}, KID:{kid:.12f}, IS:{is_score:.12f}, GS:{gs:.6f}, MSID:{msid:.6f}, Precision:{pre:.6f}, Recall:{rec:.6f}')

    f_txt.close()


            
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_path', type=str, default='./local_data/samples_cyclegan', help='path to config file')
    parser.add_argument('--log_path', type=str, default='./local_data/log_tmp', help='path to config file')
    parser.add_argument('--log_name', type=str, default='C2A_sample_metrics_tmp', help='path to config file')
    parser.add_argument('--source_data', type=str, default='cremi-c', help='cremi-c, ac4')
    parser.add_argument('--count_num', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--padding', type=int, default=20)
    parser.add_argument('--random_crop', action='store_true', default=False)
    parser.add_argument('--cal_psnr_and_ssim', action='store_true', default=False)
    parser.add_argument('--eval_all', action='store_true', default=False)
    parser.add_argument('--clip', action='store_true', default=False)
    parser.add_argument('--piq', action='store_true', default=False)


    args = parser.parse_args()


    eval_on_samples(args)



