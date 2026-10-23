# MMD, 3D-FID
# Hierarchical Amortized GAN for 3D High Resolution Medical Image Synthesis
# https://github.com/batmanlab/HA-GAN
import sys
import os

import torch
import torch.nn as nn
import numpy as np


from torch import nn
import torch
import torch.nn.functional as F
from math import exp
import collections

from piq import FID,MSID,KID,PR,IS, GS

from PIL import Image
from scipy import linalg

# 专门计算指标的网络
from networks_copy.resnet3D import resnet50

# torch.manual_seed(0)
# torch.cuda.manual_seed_all(0)
# torch.backends.cudnn.benchmark = True




def trim_state_dict_name(ckpt):
    new_state_dict = collections.OrderedDict()
    for k, v in ckpt.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    return new_state_dict

class Flatten(torch.nn.Module):
    def forward(self, inp):
        return inp.view(inp.size(0), -1)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_activations_from_dataloader(model, data_loader):
    batch_size=1
    device = torch.device("cuda:0" )
    pred_arr = np.empty((len(data_loader), 2048)) # dims [resnet50] 512 * 4(bottleneck_expansion) = 2048
    for i, batch in enumerate(data_loader):
        if i % 10 == 0:
            print('\rPropagating batch %d' % i, end='', flush=True)
        
        batch = batch.float().to(device)
        with torch.no_grad():
            pred = model(batch)

        if i*batch_size > pred_arr.shape[0]:
            pred_arr[i*batch_size:] = pred.cpu().numpy()
        else:
            pred_arr[i*batch_size:(i+1)*batch_size] = pred.cpu().numpy()
    print(' done')
    return pred_arr

# 公式计算
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)

def post_process(act):
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

# model resnet50
#--------------------------------- 修改ckpt
def get_feature_extractor():
    model = resnet50(shortcut_type='B')
    model.conv_seg = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)),
                                   Flatten()) # (N, 512)
    # ckpt from https://drive.google.com/file/d/1399AsrYpQDi1vq6ciKRQkfknLsQQyigM/view?usp=sharing
    # 另外还有一个'resnet_50_23dataset'
    ckpt = torch.load("./local_data/MedicalNet_FID/resnet_50.pth")
    ckpt = trim_state_dict_name(ckpt["state_dict"])
    model.load_state_dict(ckpt) # No conv_seg module in ckpt
    device = torch.device("cuda" ) # 

    model = model.to(device)
    model.eval()
    print("Feature extractor weights loaded")
    return model



def eval_on_piq(real_list, recon_list):
    model_v = get_feature_extractor()
            
    feat_real = get_activations_from_dataloader(model_v, real_list) # (len,2048)
    feat_recon = get_activations_from_dataloader(model_v, recon_list)

    feat_real=torch.from_numpy(feat_real).float()
    feat_recon=torch.from_numpy(feat_recon).float()

    fid_metric = FID()
    fid = fid_metric(feat_real, feat_recon)
    
    kid_metric = KID()
    kid = kid_metric(feat_real, feat_recon) # mmd

    is_metric = IS()
    is_score = is_metric(feat_real, feat_recon)

    gs_metric = GS()
    gs = gs_metric(feat_real, feat_recon)

    msid_metric = MSID()
    msid = msid_metric(feat_real, feat_recon)

    pr_metric = PR()
    precision, recall = pr_metric(feat_real, feat_recon)

    return fid, kid, is_score, gs, msid, precision, recall 



    


def calculate_imagebased(x, y, use_piq=False, transpose=False):
    from piq import ssim,multi_scale_ssim,psnr
    if len(x.shape)==5:
        x=x.squeeze(0)
    if len(y.shape)==5:
        y=y.squeeze(0)

    if transpose:
        x = x.transpose(0, 1)
        y = y.transpose(0, 1)
        
    if use_piq: # [18, 1, 160, 160]
        # kernel_size=11, >161.
        msssim = multi_scale_ssim( x, y, kernel_size=9,data_range=1.0).item() # 
        ssim_piq = ssim( x, y, data_range=1.0).item()
        psnr_piq = psnr(x,y).item()
    else:
        msssim=0.0
        ssim_piq=0.0
        psnr_piq=0.0
    
    ssim_loss = SSIM(window_size=11, size_average=True)
    ssim_self = ssim_loss(x, y)

    psnr=cal_psnr(x, y)
    
    mse=nn.MSELoss()
    mse_loss=mse(y,x).item()
    
    
    return ssim_self,ssim_piq,msssim,psnr,psnr_piq,mse_loss


class SSIM(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window(self, window_size, channel=1):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        window = self.create_window(self.window_size, channel)

        if img1.is_cuda:
            window = window.cuda(img1.get_device())
        window = window.type_as(img1)

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    

def cal_psnr(x, y):
    # psnr = 10 * torch.log10(max_pixcel_value / mse)
    mse=nn.MSELoss()
    mse_loss=mse(y,x).item()
    psnr = 10 * np.log10(1 / mse_loss)
    return psnr







