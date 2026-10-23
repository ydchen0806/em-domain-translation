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
# from sklearn.metrics import f1_score, average_precision_score, roc_auc_score

# from provider_valid import Provider_valid
# from loss.loss import BCELoss, WeightedBCE, MSELoss

from networks_copy.seg.model_superhuman import UNet_PNI
from data.provider_valid import ValidDataset_seg

from utils.show import draw_fragments_3d
from utils.fragment import watershed, relabel
# from utils.shift_channels import shift_func
# from utils.lmc import mc_baseline
# from utils.fragment import watershed, randomlabel, relabel, elf_watershed

import waterz
from utils.lmc import mc_baseline
# import evaluate as ev
from skimage.metrics import adapted_rand_error as adapted_rand_ref
from skimage.metrics import variation_of_information as voi_ref
from data.data_segmentation import seg_widen_border
import warnings
warnings.filterwarnings("ignore")


def visual_draw_box(output_path):
    output_path= './local_data/code/maisi/checkpoint/figure'
    image_name = 'seg_cyclegan.png'

    from PIL import Image, ImageDraw

    image = Image.open(os.path.join(output_path,image_name)).convert('RGB')

    w=4
    # 橘黄：(255, 192, 0)
    # 
    color_1 =(255, 192, 0) # red:(222, 49, 35)  # yellow: (237, 230, 50)
    color_int1 = color_1[0] + (color_1[1] << 8) + (color_1[2] << 16)

    color_2 =(222, 49, 35)# red:(222, 49, 35)  # yellow: (237, 230, 50)
    color_int2 = color_2[0] + (color_2[1] << 8) + (color_2[2] << 16)

    

    draw = ImageDraw.Draw(image)

    bbox1=(100, 176, 190, 230)
    a,b,c,d = bbox1
    bbox2 = (a,b+240,c, d+240)
    draw.rectangle(bbox1, outline=color_int1, width=w)
    draw.rectangle(bbox2, outline=color_int1, width=w)

    bbox3=(5, 10, 68, 80)
    a,b,c,d = bbox3
    bbox4 = (a,b+240,c, d+240)
    draw.rectangle(bbox3, outline=color_int2, width=w)
    draw.rectangle(bbox4, outline=color_int2, width=w)



    image.save(os.path.join(output_path,'c2a_cyclegan.png'))

def visual(output_path):
    # SEED = 42
    crop = [240,240]
    # names = ['GT','unsb','cyclegan','ldm','ours']
    names = ['GT','ours','cyclegan']
    num = 1
    IMG_PATH = "./local_data/CREMI/cremiC_inputs_interp.h5"
    # ./local_data/CREMI/cremiC_inputs_interp.h5
    # "./local_data/AC3_AC4/AC4_inputs.h5"
    IMG_SPLIT = 100 # cremi-c , 100 # AC4  80

    save_path = os.path.join(output_path,"figure")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i in range(num):
        for name in names:
            # if name=='GT':
            #     file_path = os.path.join(output_path,name+'_seg.hdf')
            # else:
            #     file_path = os.path.join(output_path,name+'_seg_w.hdf')
            file_path = os.path.join(output_path,name+'_seg.hdf')
            with h5py.File(file_path, 'r') as f:
                seg = f['main'][:]
            f.close()

            file_path = os.path.join(output_path,name+'_affs.hdf')
            with h5py.File(file_path, 'r') as f:
                affs = f['main'][:]
            f.close()

            if name=='GT':
                z,y,x  = seg.shape
                start_z = 19 #random.randint(0, z-1)
                start_y = 685 #random.randint(0, y-crop[0])
                start_x = 717 #random.randint(0, x-crop[1])
                # gt_seg = seg.copy()
                # start_z = 8
                # start_y = 597
                # start_x = 305

            
            
            seg = draw_fragments_3d(seg)
            # if not name =='GT': # cremi的话不用
            #     seg[gt_seg ==0] = 0
            cat1 = affs[2,start_z,start_y:start_y+crop[0],start_x:start_x+crop[1]]
            cat1 = np.stack([cat1]*3, axis=-1)
            # cat1 = cat1.convert('RGB')
            # cat1 = cv2.applyColorMap((cat1).astype(np.uint8), cv2.COLORMAP_JET)
            cat2 = seg[start_z,start_y:start_y+crop[0],start_x:start_x+crop[1]]

            cat1 = (cat1 * 255).astype(np.uint8)
            # print('cat1:',cat1.shape)
            # print('cat2:',cat2.shape)
            im_cat = np.concatenate([cat1, cat2], axis=0)
            # f'seg_{str(i).zfill(4)}_{name}_z{start_z}_y{start_y}_x{start_x}.png'
            cv2.imwrite(os.path.join(save_path, f'seg_z{start_z}_y{start_y}_x{start_x}_{name}.png'), im_cat)


        f_raw = h5py.File(IMG_PATH, 'r')
        data = f_raw['main'][IMG_SPLIT:]

        img = data[start_z,start_y:start_y+crop[0],start_x:start_x+crop[1]]
        cv2.imwrite(os.path.join(save_path,f'seg_z{start_z}_y{start_y}_x{start_x}_img.png'),img)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default='./local_data/code/maisi/checkpoint', help='path to config file')
    parser.add_argument('--model_path', type=str, default='./local_data/code/maisi/models_lmc', help='path to config file')
    parser.add_argument('--model_name', type=str, default='model.pt', help='path to config file')
    # parser.add_argument('-c', '--cfg', type=str, default='seg_3d_ac4_data80', help='path to config file')
    # parser.add_argument('-mn', '--model_name', type=str, default='2023-10-02--15-36-33_seg_3d_ac4_data80')
    # parser.add_argument('-id', '--model_id', type=int, default=188000)
    # parser.add_argument('-m', '--mode', type=str, default='ac3')
    # parser.add_argument('-ts', '--test_split', type=int, default=100)
    # parser.add_argument('-sm', '--seg_mode', type=str, default='waterz')
    # parser.add_argument('-f', '--fragment', type=str, default='mahotas')  # 'elf', 'mahotas'
    # parser.add_argument('-pm', '--pixel_metric', action='store_true', default=False)
    # parser.add_argument('-s', '--save', action='store_false', default=True)
    # parser.add_argument('-sw', '--show', action='store_true', default=True)
    # parser.add_argument('-malis', '--malis_loss', action='store_true', default=False)
    args = parser.parse_args()

    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print('out_path: ' + save_path)

    only_visual=False

    if only_visual:
        visual_draw_box(args.save_path)
    else:

        device = torch.device('cuda:0')

        model = UNet_PNI(in_planes=1,
                        out_planes=3,
                        filters=[28, 36, 48, 64, 80],
                        upsample_mode="bilinear",
                        decode_ratio=1,
                        merge_mode="add",
                        pad_mode="zero",
                        bn_mode="async",
                        relu_mode="elu",
                        init_mode="kaiming_normal").to(device)

        ckpt_path = os.path.join(args.model_path,args.model_name)
        checkpoint = torch.load(ckpt_path)

        new_state_dict = OrderedDict()
        # state_dict = checkpoint['seg_state_dict']
        # for k, v in state_dict.items():
        #     # name = k[7:] # remove module.
        #     name = k
        #     new_state_dict[name] = v
        # model.load_state_dict(new_state_dict,strict=True)

        model.load_state_dict(checkpoint['seg_state_dict'],strict=True)
        model = model.to(device)

        # valid_provider = Provider_valid(cfg, valid_data=args.mode, test_split=args.test_split)

        valid_provider = ValidDataset_seg(root_dir='./local_data/AC3_AC4/',train_dataset='AC4_inputs.h5',
                                        crop_size=[18,160,160],if_train=False)
        # valid_provider = ValidDataset_seg(root_dir='./local_data/CREMI',train_dataset='cremiC_inputs_interp.h5',
        #                                 crop_size=[18,160,160],if_train=False)
        val_loader = torch.utils.data.DataLoader(valid_provider, batch_size=1, num_workers=0,
                                                    shuffle=False, drop_last=False, pin_memory=True)



        model.eval()
        print('the number of sub-volume:', len(valid_provider))
        t1 = time.time()
        pbar = tqdm(total=len(valid_provider))
        for data in val_loader:
            inputs, target, weightmap = data
            inputs = inputs.cuda()
            target = target.cuda()
            weightmap = weightmap.cuda()
            with torch.no_grad():
                pred = model(inputs)

            valid_provider.add_vol(np.squeeze(pred.data.cpu().numpy()))
            pbar.update(1)
        pbar.close()
        cost_time = time.time() - t1
        print('Inference time=%.6f' % cost_time)

        output_affs = valid_provider.get_results()
        gt_affs = valid_provider.get_gt_affs().copy()
        gt_seg = valid_provider.get_gt_lb()
        valid_provider.reset_output()
        # gt_seg = gt_seg.astype(np.uint32)
        output_affs = output_affs[:3]


        # print('save affs...')
        # f = h5py.File(os.path.join(save_path, 'affs.hdf'), 'w')
        # f.create_dataset('main', data=output_affs, dtype=np.float32, compression='gzip')
        # f.close()

        # f = h5py.File(os.path.join(save_path, 'GT_affs.hdf'), 'w')
        # f.create_dataset('main', data=gt_affs, dtype=np.float32, compression='gzip')
        # f.close()

        # f = h5py.File(os.path.join(save_path, 'GT_seg.hdf'), 'w')
        # f.create_dataset('main', data=gt_seg, dtype=np.float32, compression='gzip')
        # f.close()

        # segmentation
        print('Segmentation...')
        fragments = watershed(output_affs, 'maxima_distance')
        sf = 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256>>'
        # sf = 'OneMinus<EdgeStatisticValue<RegionGraphType, MeanAffinityProvider<RegionGraphType, ScoreValue>>>'
        seg_waterz = list(waterz.agglomerate(output_affs, [0.50],
                                            fragments=fragments,
                                            scoring_function=sf,
                                            discretize_queue=256))[0]

        # seg_waterz = relabel(seg_waterz).astype(np.uint64)
        arand = adapted_rand_ref(gt_seg, seg_waterz, ignore_labels=(0))[0]
        voi_split, voi_merge = voi_ref(gt_seg, seg_waterz, ignore_labels=(0))
        voi_sum = voi_split + voi_merge
        print('[Waterz]VOI-split=%.6f, VOI-merge=%.6f, VOI-sum=%.6f, ARAND=%.6f' %
            (voi_split, voi_merge, voi_sum, arand))

        # f = h5py.File(os.path.join(save_path, 'seg.hdf'), 'w')
        # f.create_dataset('main', data=segmentation, dtype=segmentation.dtype, compression='gzip')
        # f.close()

        # baseline,utils/lmc.py
        seg_lmc = mc_baseline(output_affs)#产生分割
        # seg_lmc = relabel(seg_lmc).astype(np.uint64)
        arand_lmc = adapted_rand_ref(gt_seg, seg_lmc, ignore_labels=(0))[0]
        voi_split2, voi_merge2 = voi_ref(gt_seg, seg_lmc, ignore_labels=(0))
        voi_sum_lmc = voi_split2 + voi_merge2

        print('[LMC]VOI-split=%.6f, VOI-merge=%.6f, VOI-sum=%.6f, ARAND=%.6f' %
            (voi_split2, voi_merge2, voi_sum_lmc, arand_lmc))


    # segmentation = mc_baseline(output_affs)
    # segmentation = relabel(segmentation).astype(np.uint64)
    # print('the max id = %d' % np.max(segmentation))
    # f = h5py.File(os.path.join(out_affs, 'seg_lmc.hdf'), 'w')
    # f.create_dataset('main', data=segmentation, dtype=segmentation.dtype, compression='gzip')
    # f.close()

    # arand = adapted_rand_ref(gt_seg, segmentation, ignore_labels=(0))[0]
    # voi_split, voi_merge = voi_ref(gt_seg, segmentation, ignore_labels=(0))
    # voi_sum = voi_split + voi_merge
    # print('LMC: voi_split=%.6f, voi_merge=%.6f, voi_sum=%.6f, arand=%.6f' % \
    #     (voi_split, voi_merge, voi_sum, arand))

    
    # output_affs_prop = output_affs


    # show
    # if args.show:
    #     print('show affs...')
    #     output_affs_prop = (output_affs_prop * 255).astype(np.uint8)
    #     gt_affs = (gt_affs * 255).astype(np.uint8)
    #     for i in range(output_affs_prop.shape[1]):
    #         cat1 = np.concatenate([output_affs_prop[0,i], output_affs_prop[1,i], output_affs_prop[2,i]], axis=1)
    #         cat2 = np.concatenate([gt_affs[0,i], gt_affs[1,i], gt_affs[2,i]], axis=1)
    #         im_cat = np.concatenate([cat1, cat2], axis=0)
    #         cv2.imwrite(os.path.join(affs_img_path, str(i).zfill(4)+'.png'), im_cat)
        
    #     print('show seg...')
    #     segmentation[gt_seg==0] = 0
    #     color_seg = draw_fragments_3d(segmentation)
    #     color_gt = draw_fragments_3d(gt_seg)
    #     for i in range(color_seg.shape[0]):
    #         im_cat = np.concatenate([color_seg[i], color_gt[i]], axis=1)
    #         cv2.imwrite(os.path.join(seg_img_path, str(i).zfill(4)+'.png'), im_cat)
    # print('Done')

