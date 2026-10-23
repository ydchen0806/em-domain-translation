import os
import torch
import h5py
import cv2
import numpy as np
from torch.utils.data import Dataset
from skimage import exposure
from PIL import Image
import matplotlib.pyplot as plt

from data.data_segmentation import seg_widen_border, weight_binary_ratio
from data.data_affinity import seg_to_aff

def histogram_equalization_3d(data):
    # 存储每个切片的累积分布函数（CDF）
    cdfs = []
    
    # 对每个切片进行直方图均衡化
    for i in range(data.shape[0]):
        hist, bins = np.histogram(data[i].flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * (hist.max() / cdf.max())
        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
        cdf_m = np.ma.filled(cdf_m, 0).astype('uint8')
        cdfs.append(cdf_m)
        data[i] = cdf_m[data[i]]
    
    return data #, cdfs

def plot_histogram_3d(data, output_dir):
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 遍历每个切片
    for i in range(data.shape[0]):
        # 计算直方图
        hist, bins = np.histogram(data[i].flatten(), 256, [0, 256])
        
        # 绘制直方图
        plt.figure(figsize=(10, 5))
        plt.bar(bins[:-1], hist, width=1, color='blue', alpha=0.7)
        plt.title(f'3D Data Slice {i} Histogram')
        plt.xlabel('Gray Level')
        plt.ylabel('Frequency')
        
        # 保存直方图图片
        plt.savefig(os.path.join(output_dir, f'histogram_slice_AC4_{i}.png'))
        plt.close()

def gamma_trans(input, gamma=2, eps=0 ):
    return (input + eps) ** gamma


# 用于src和tgt之间的翻译任务。需要对src和tgt做相同的滑窗裁块，且总数一致。目前的只适用于C2AC4
# 【拼接对应的代码没写，未使用
class ValidDataset_translation(Dataset):
    def __init__(self, 
                 src_dir, src_dataset, tgt_dir, tgt_dataset, 
                 crop_size = [18, 160, 160], norm2=False, recon_all=True):

        self.norm2 = norm2
        self.recon_all = recon_all
 
        # num = (volume_length + 2 * padding - crop_length) // stride + 1
        if src_dataset == 'cremiC_inputs_interp.h5' and tgt_dataset=='AC4_inputs.h5':

            src_label='cremiC_labels.h5'
            if self.recon_all: # 最后拼接成一大块，所以两侧有padding，会导致中间每一块的结果是错的
                # 1-cremic: (100, 1024, 1024)
                self.stride = [10, 80, 80]
                self.valid_padding = [4, 48, 48]

                # 2-AC4: (80, 1024, 1024)
                self.stride2 = [8, 80, 80]
                self.valid_padding2 = [5, 48, 48]

                self.num_zyx = [10, 13, 13]
            else: # 不重建，则无需padding
                # 1-cremic: (100, 1024, 1024)
                self.stride = [41, 59, 59]
                self.valid_padding = [0, 0, 0]

                # 2-AC4: (80, 1024, 1024)
                self.stride2 = [31, 59, 59]
                self.valid_padding2 = [0, 0, 0]

                self.num_zyx = [3, 17, 17]

        else: 
            raise ValueError("Invalid dataset name")
        
        self.num_per_dataset = self.num_zyx[0] * self.num_zyx[1] * self.num_zyx[2]

        self.crop_size = crop_size
        self.net_padding = [0, 0, 0]
        self.out_size = [self.crop_size[k] - 2 * self.net_padding[k] for k in range(len(self.crop_size))] # 减掉两侧padding


        # -------------- load data and label --------------------
        f_raw = h5py.File(os.path.join(src_dir, src_dataset), 'r')
        data = f_raw['main'][:]
        f_raw.close()
        self.src_dataset = data[:100]


        f_raw = h5py.File(os.path.join(tgt_dir, tgt_dataset), 'r')
        data = f_raw['main'][:]
        f_raw.close()
        self.tgt_dataset = data[:80]


        f_label = h5py.File(os.path.join(src_dir, src_label), 'r')
        label = f_label['main'][:]
        f_label.close()
        label = seg_widen_border(label[:100], tsz_h=1)
        self.labels = label

        # ------------- padding -----------------------
        
        self.src_dataset = np.pad(self.src_dataset, ((self.valid_padding[0], self.valid_padding[0]), \
                                                (self.valid_padding[1], self.valid_padding[1]), \
                                                (self.valid_padding[2], self.valid_padding[2])), mode='reflect')
        self.tgt_dataset = np.pad(self.tgt_dataset, ((self.valid_padding2[0], self.valid_padding2[0]), \
                                                (self.valid_padding2[1], self.valid_padding2[1]), \
                                                (self.valid_padding2[2], self.valid_padding2[2])), mode='reflect')
        self.labels = np.pad(self.labels, ((self.valid_padding[0], self.valid_padding[0]), \
                                                (self.valid_padding[1], self.valid_padding[1]), \
                                                (self.valid_padding[2], self.valid_padding[2])), mode='reflect')
        
        self.raw_data_shape = list(self.src_dataset.shape)


        # 以下在普通采样调试中不会用到，只有需要完整拼接的才有。get item里面的weightmap也去掉了
        # self.gt_affs = seg_to_aff(self.labels).astype(np.float32)

        # self.reset_output()
        # self.weight_vol = self.get_weight()

        


    def __len__(self):
        return self.num_per_dataset


    def __getitem__(self, index):

        pre_data = index % self.num_per_dataset
        pos_z = pre_data // (self.num_zyx[1] * self.num_zyx[2])
        pos_xy = pre_data % (self.num_zyx[1] * self.num_zyx[2])
        pos_x = pos_xy // self.num_zyx[2]
        pos_y = pos_xy % self.num_zyx[2]

        fromz = pos_z * self.stride[0]
        endz = fromz + self.crop_size[0]
        if endz > self.raw_data_shape[0]:
            endz = self.raw_data_shape[0]
            fromz = endz - self.crop_size[0]
        fromy = pos_y * self.stride[1]
        endy = fromy + self.crop_size[1]
        if endy > self.raw_data_shape[1]:
            endy = self.raw_data_shape[1]
            fromy = endy - self.crop_size[1]
        fromx = pos_x * self.stride[2]
        endx = fromx + self.crop_size[2]
        if endx > self.raw_data_shape[2]:
            endx = self.raw_data_shape[2]
            fromx = endx - self.crop_size[2]
        self.pos = [fromz, fromy, fromx]

        fromz2 = pos_z * self.stride2[0]
        endz2 = fromz + self.crop_size[0]


        src_imgs = self.src_dataset[fromz:endz, fromx:endx, fromy:endy].copy()
        tgt_imgs = self.tgt_dataset[fromz2:endz2, fromx:endx, fromy:endy].copy()
        lb = self.labels[fromz:endz, fromx:endx, fromy:endy].copy()

        # imgs = exposure.rescale_intensity(imgs)
        src_imgs = src_imgs.astype(np.float32) / 255.0
        src_imgs = src_imgs[np.newaxis, ...]
        src_imgs = np.ascontiguousarray(src_imgs, dtype=np.float32)
        if self.norm2:
            src_imgs = 2*src_imgs-1.0
        src_imgs = torch.from_numpy(src_imgs)

        tgt_imgs = tgt_imgs.astype(np.float32) / 255.0
        tgt_imgs = tgt_imgs[np.newaxis, ...]
        tgt_imgs = np.ascontiguousarray(tgt_imgs, dtype=np.float32)
        if self.norm2:
            tgt_imgs = 2*tgt_imgs-1.0
        tgt_imgs = torch.from_numpy(tgt_imgs)

        lb_affs = seg_to_aff(lb).astype(np.float32)
        # weightmap = weight_binary_ratio(lb_affs)

        lb_affs = np.ascontiguousarray(lb_affs, dtype=np.float32)
        # weightmap = np.ascontiguousarray(weightmap, dtype=np.float32)
        lb_affs = torch.from_numpy(lb_affs) # (3, 18, 160, 160)
        # weightmap = torch.from_numpy(weightmap)

        return src_imgs,lb_affs, tgt_imgs #, weightmap



    def reset_output(self):

        self.outputs = np.zeros(tuple([3]+self.raw_data_shape), dtype=np.float32)
        self.weight_map = np.zeros(tuple([1]+self.raw_data_shape), dtype=np.float32)


    def get_weight(self, sigma=0.2, mu=0.0):

        zz, yy, xx = np.meshgrid(np.linspace(-1, 1, self.out_size[0], dtype=np.float32),
                                np.linspace(-1, 1, self.out_size[1], dtype=np.float32),
                                np.linspace(-1, 1, self.out_size[2], dtype=np.float32), indexing='ij')
        dd = np.sqrt(zz * zz + yy * yy + xx * xx)
        weight = 1e-6 + np.exp(-((dd - mu) ** 2 / (2.0 * sigma ** 2)))
        weight = weight[np.newaxis, ...]
        return weight

    def add_vol(self, sub_vol):
        # if len(sub_vol.shape) == 3:
        #     sub_vol = np.transpose(sub_vol, (2, 1, 0)) # maisi
        # else:
        #     sub_vol = np.transpose(sub_vol, (0, 3, 2, 1)) # [3, 160, 160, 18]
        fromz, fromy, fromx = self.pos

        self.outputs[:, fromz:fromz+self.out_size[0], \
                            fromx:fromx+self.out_size[1], \
                            fromy:fromy+self.out_size[2]] += sub_vol * self.weight_vol    #权重
        self.weight_map[:, fromz:fromz+self.out_size[0], \
                            fromx:fromx+self.out_size[1], \
                            fromy:fromy+self.out_size[2]] += self.weight_vol


    def get_results(self):  # 去掉padding
        self.outputs = self.outputs / self.weight_map
        if self.valid_padding[0] == 0:
            self.outputs = self.outputs[:, :, \
                                            self.valid_padding[1]:-self.valid_padding[1], \
                                            self.valid_padding[2]:-self.valid_padding[2]]
        else:
            self.outputs = self.outputs[:, self.valid_padding[0]:-self.valid_padding[0], \
                                            self.valid_padding[1]:-self.valid_padding[1], \
                                            self.valid_padding[2]:-self.valid_padding[2]]
        return self.outputs

    # def get_gt_affs(self):
    #     return self.gt_affs.copy()

    # def get_gt_lb(self):
    #     return self.gt_affs
    def get_gt_affs(self, num_data=0):
        return self.gt_affs # 好像是只用了gt_affs[0]就是z轴的?

    def get_gt_lb(self, num_data=0):
        # lbs = self.labels_origin[num_data].copy()
        # lbs = lbs[self.valid_padding[0]:-self.valid_padding[0], \
        #           self.valid_padding[1]:-self.valid_padding[1], \
        #           self.valid_padding[2]:-self.valid_padding[2]]
        return self.labels_origin

    def get_raw_data(self):
        out = self.dataset.copy()
        return out[self.valid_padding[0]:-self.valid_padding[0], \
                    self.valid_padding[1]:-self.valid_padding[1], \
                    self.valid_padding[2]:-self.valid_padding[2]]



class ValidDataset_seg(Dataset):
    def __init__(self, root_dir,train_dataset,crop_size=[18, 160, 160],if_train=True, norm2=False, recon_all=True):
        # basic settings
        self.folder_name = root_dir
        self.train_dataset = train_dataset
        self.norm2 = norm2
        self.recon_all = recon_all
        
        # num = (length + 2 * padding - window_length) // stride + 1
        if self.train_dataset == 'cremiC_inputs_interp.h5': # (125, 1024, 1024)
            train_label='cremiC_labels.h5'
            if crop_size[0]==18:
                self.train_split=100
                self.test_split=25
                if if_train: 
                    self.stride = [10, 80, 80]
                    self.valid_padding = [4, 48, 48]
                    self.num_zyx = [10, 13, 13]
                else:
                    self.stride = [15, 80, 80]
                    self.valid_padding = [4, 48, 48]
                    self.num_zyx = [2, 13, 13]
            else: # 32
                self.train_split=75
                self.test_split=50
                if if_train: # 75
                    self.stride = [17, 80, 80]
                    self.valid_padding = [4, 48, 48]
                    self.num_zyx = [4, 13, 13]
                else:
                    self.stride = [10, 80, 80]
                    self.valid_padding = [6, 48, 48]
                    self.num_zyx = [4, 13, 13]
        elif self.train_dataset == 'AC4_inputs.h5': # (100, 1024, 1024)
            train_label='AC4_labels.h5'
            if crop_size[0]==18:
                self.train_split=80
                self.test_split=20
                if if_train: # 80
                    if self.recon_all:
                        self.stride = [10, 80, 80]
                        self.valid_padding = [4, 48, 48]
                        self.num_zyx = [8, 13, 13] # 1352
                    else:
                        self.stride = [31, 59, 59]
                        self.valid_padding = [0, 0, 0]
                        self.num_zyx = [3, 17, 17] # 1352
                else:
                    self.stride = [10, 80, 80]
                    self.valid_padding = [4, 48, 48]
                    self.num_zyx = [2, 13, 13]
            else:
                self.train_split=50
                self.test_split=50
            
            # new: (50, 1024, 1024)
            # (32, 160, 160)
                self.stride = [10, 80, 80]
                self.valid_padding = [6, 48, 48]
                self.num_zyx = [4, 13, 13]
        elif self.train_dataset == 'AC3_inputs.h5': # (256, 1024, 1024) ##还没想好
            self.train_split=50
            self.test_split=50
            train_label='AC3_labels.h5'
            self.stride = [10, 80, 80]
            self.valid_padding = [6, 48, 48]
            self.num_zyx = [4, 13, 13]
        else: 
            raise ValueError("Invalid dataset name")
        

        
        self.if_train=if_train


        self.crop_size = crop_size
        self.net_padding = [0, 0, 0]
        self.out_size = [self.crop_size[k] - 2 * self.net_padding[k] for k in range(len(self.crop_size))]

        # load raw data
        f_raw = h5py.File(os.path.join(self.folder_name, self.train_dataset), 'r')
        data = f_raw['main'][:]
        f_raw.close()
        if self.if_train:
            data = data[:self.train_split] ### 注意调整
        else:
            data=data[-self.test_split:] # (50, 1024, 1024)
        self.dataset = data

        f_label = h5py.File(os.path.join(self.folder_name, train_label), 'r')
        label = f_label['main'][:]
        f_label.close()
        if self.if_train:
            label = label[:self.train_split]
        else:
            label=label[-self.test_split:]
        
        self.labels_origin = label.copy() # AC4的label在这里就是含背景的了


        label = seg_widen_border(label, tsz_h=1)
        self.labels = label
        

        # self.dataset = histogram_equalization_3d(data)
        # gamma_trans(data/255.0, gamma=0.25)*255.0 # [0, 255]

        # self.dataset[self.dataset==0] = 10

        # plot_histogram_3d(data, './local_data/code/maisi/predictions/ac3/')

        
        # data = gamma_trans(data, gamma=0.5)
        # Image.fromarray(self.dataset[0]).save(os.path.join('./local_data/code/maisi/predictions', '2.png'))
        # data = gamma_trans(data, gamma=2.0)
        # Image.fromarray(data[0]).convert('L').save(os.path.join('./local_data/code/maisi/predictions', '3.png'))
        
        self.origin_data_shape = list(self.dataset.shape)


 
        # (25, 1024, 1024)，用superhuman里的拼接参数
        # (18, 160, 160)
        # self.stride = [15, 80, 80]
        # self.valid_padding = [4, 48, 48]
        # self.num_zyx = [2, 13, 13]

        
        # num = (length + 2 * padding - window_length) // stride + 1

        self.gt_affs = seg_to_aff(self.labels).astype(np.float32)

        self.dataset = np.pad(self.dataset, ((self.valid_padding[0], self.valid_padding[0]), \
                                                (self.valid_padding[1], self.valid_padding[1]), \
                                                (self.valid_padding[2], self.valid_padding[2])), mode='reflect')
        self.labels = np.pad(self.labels, ((self.valid_padding[0], self.valid_padding[0]), \
                                                (self.valid_padding[1], self.valid_padding[1]), \
                                                (self.valid_padding[2], self.valid_padding[2])), mode='reflect')
        

        # the training dataset size
        self.raw_data_shape = list(self.dataset.shape)

        self.reset_output()
        self.weight_vol = self.get_weight()

        self.num_per_dataset = self.num_zyx[0] * self.num_zyx[1] * self.num_zyx[2]
        self.iters_num = self.num_per_dataset #* len(self.dataset)

    def __getitem__(self, index):

        pre_data = index % self.num_per_dataset
        pos_z = pre_data // (self.num_zyx[1] * self.num_zyx[2])
        pos_xy = pre_data % (self.num_zyx[1] * self.num_zyx[2])
        pos_x = pos_xy // self.num_zyx[2]
        pos_y = pos_xy % self.num_zyx[2]

        # find position因为有crop
        fromz = pos_z * self.stride[0]
        endz = fromz + self.crop_size[0]
        if endz > self.raw_data_shape[0]:
            endz = self.raw_data_shape[0]
            fromz = endz - self.crop_size[0]
        fromy = pos_y * self.stride[1]
        endy = fromy + self.crop_size[1]
        if endy > self.raw_data_shape[1]:
            endy = self.raw_data_shape[1]
            fromy = endy - self.crop_size[1]
        fromx = pos_x * self.stride[2]
        endx = fromx + self.crop_size[2]
        if endx > self.raw_data_shape[2]:
            endx = self.raw_data_shape[2]
            fromx = endx - self.crop_size[2]
        self.pos = [fromz, fromy, fromx]

        imgs = self.dataset[fromz:endz, fromx:endx, fromy:endy].copy()
        lb = self.labels[fromz:endz, fromx:endx, fromy:endy].copy()

        # imgs = exposure.rescale_intensity(imgs)
        imgs = imgs.astype(np.float32) / 255.0

        # imgs = np.transpose(imgs, (2, 1, 0)) # maisi # seg就不用了

        imgs = imgs[np.newaxis, ...]
        imgs = np.ascontiguousarray(imgs, dtype=np.float32)
        if self.norm2:
            imgs = 2*imgs-1.0

        imageout = torch.from_numpy(imgs)
        # if self.if_mask:
        #     lb_affs = seg_to_aff(lb).astype(np.float32)
        #     # weightmap = weight_binary_ratio(lb_affs)
        #     lb_affs = np.ascontiguousarray(lb_affs, dtype=np.float32)
        #     # weightmap = np.ascontiguousarray(weightmap, dtype=np.float32)
        #     lb_affs = torch.from_numpy(lb_affs) # (3, 18, 160, 160)
        #     lb_affs = lb_affs.permute(0, 3, 2, 1) # (3, 160, 160, 32)
        #     return {
        #         "image": lb_affs,}
        # else:
        #     return {
        # "image": imageout,}
        lb_affs = seg_to_aff(lb).astype(np.float32)
        weightmap = weight_binary_ratio(lb_affs)

        lb_affs = np.ascontiguousarray(lb_affs, dtype=np.float32)
        weightmap = np.ascontiguousarray(weightmap, dtype=np.float32)
        lb_affs = torch.from_numpy(lb_affs) # (3, 18, 160, 160)
        weightmap = torch.from_numpy(weightmap)
        # lb_affs = lb_affs.permute(0, 3, 2, 1) # (3, 160, 160, 32)
        return imageout,lb_affs,weightmap

    def __len__(self):
        return self.num_per_dataset # self.iters_num

    def reset_output(self):

        self.outputs = np.zeros(tuple([3]+self.raw_data_shape), dtype=np.float32)
        self.weight_map = np.zeros(tuple([1]+self.raw_data_shape), dtype=np.float32)


    def get_weight(self, sigma=0.2, mu=0.0):

        zz, yy, xx = np.meshgrid(np.linspace(-1, 1, self.out_size[0], dtype=np.float32),
                                np.linspace(-1, 1, self.out_size[1], dtype=np.float32),
                                np.linspace(-1, 1, self.out_size[2], dtype=np.float32), indexing='ij')
        dd = np.sqrt(zz * zz + yy * yy + xx * xx)
        weight = 1e-6 + np.exp(-((dd - mu) ** 2 / (2.0 * sigma ** 2)))
        weight = weight[np.newaxis, ...]
        return weight

    def add_vol(self, sub_vol):
        # if len(sub_vol.shape) == 3:
        #     sub_vol = np.transpose(sub_vol, (2, 1, 0)) # maisi
        # else:
        #     sub_vol = np.transpose(sub_vol, (0, 3, 2, 1)) # [3, 160, 160, 18]
        fromz, fromy, fromx = self.pos

        self.outputs[:, fromz:fromz+self.out_size[0], \
                            fromx:fromx+self.out_size[1], \
                            fromy:fromy+self.out_size[2]] += sub_vol * self.weight_vol    #权重
        self.weight_map[:, fromz:fromz+self.out_size[0], \
                            fromx:fromx+self.out_size[1], \
                            fromy:fromy+self.out_size[2]] += self.weight_vol


    def get_results(self):  # 去掉padding
        self.outputs = self.outputs / self.weight_map
        if self.valid_padding[0] == 0:
            self.outputs = self.outputs[:, :, \
                                            self.valid_padding[1]:-self.valid_padding[1], \
                                            self.valid_padding[2]:-self.valid_padding[2]]
        else:
            self.outputs = self.outputs[:, self.valid_padding[0]:-self.valid_padding[0], \
                                            self.valid_padding[1]:-self.valid_padding[1], \
                                            self.valid_padding[2]:-self.valid_padding[2]]
        return self.outputs

    # def get_gt_affs(self):
    #     return self.gt_affs.copy()

    # def get_gt_lb(self):
    #     return self.gt_affs
    def get_gt_affs(self, num_data=0):
        return self.gt_affs # 好像是只用了gt_affs[0]就是z轴的?

    def get_gt_lb(self, num_data=0):
        # lbs = self.labels_origin[num_data].copy()
        # lbs = lbs[self.valid_padding[0]:-self.valid_padding[0], \
        #           self.valid_padding[1]:-self.valid_padding[1], \
        #           self.valid_padding[2]:-self.valid_padding[2]]
        return self.labels_origin

    def get_raw_data(self):
        out = self.dataset.copy()
        return out[self.valid_padding[0]:-self.valid_padding[0], \
                    self.valid_padding[1]:-self.valid_padding[1], \
                    self.valid_padding[2]:-self.valid_padding[2]]
    
# 这个是用来vae重建的，要么用img要么用mask
class validDataset(Dataset):
    def __init__(self, root_dir,train_dataset,if_train=True,if_mask=False,norm2=False,crop=[32, 160, 160]):
        # basic settings
        self.folder_name = root_dir
        self.train_dataset = train_dataset
        self.norm2=norm2

        self.crop_size = [18, 160, 160]
        
        if self.train_dataset == 'cremiC_inputs_interp.h5': # (125, 1024, 1024)
            self.train_split=100
            self.test_split=25
            train_label='cremiC_labels.h5'
            # num = (length + 2 * padding - window_length) // stride + 1
            if if_train: # 100
                # 75
                # self.stride = [17, 80, 80]
                # self.valid_padding = [4, 48, 48]
                # self.num_zyx = [4, 13, 13]
                self.stride = [10, 80, 80]
                self.valid_padding = [4, 48, 48]
                self.num_zyx = [10, 13, 13]
            else:
                # 50
                # self.stride = [10, 80, 80]
                # self.valid_padding = [6, 48, 48]
                # self.num_zyx = [4, 13, 13]
                self.stride = [15, 80, 80]
                self.valid_padding = [4, 48, 48]
                self.num_zyx = [2, 13, 13]
        elif self.train_dataset == 'AC4_inputs.h5': # (100, 1024, 1024)
            
            train_label='AC4_labels.h5'
            # new: (50, 1024, 1024)
            # (32, 160, 160)
            # self.train_split=50
            # self.test_split=50
            # self.stride = [10, 80, 80]
            # self.valid_padding = [6, 48, 48]
            # self.num_zyx = [4, 13, 13]
            self.train_split=80
            self.test_split=20
            if if_train: # 80
                self.stride = [10, 80, 80]
                self.valid_padding = [4, 48, 48]
                self.num_zyx = [8, 13, 13]
            else:
                self.stride = [10, 80, 80]
                self.valid_padding = [4, 48, 48]
                self.num_zyx = [2, 13, 13]
        elif self.train_dataset == 'AC3_inputs.h5': # (256, 1024, 1024) ##还没想好
            self.train_split=50
            self.test_split=50
            train_label='AC3_labels.h5'
            self.stride = [10, 80, 80]
            self.valid_padding = [6, 48, 48]
            self.num_zyx = [4, 13, 13]
        else: 
            raise ValueError("Invalid dataset name")
        

        
        self.if_train=if_train
        self.if_mask=if_mask

        
        self.net_padding = [0, 0, 0]
        self.out_size = [self.crop_size[k] - 2 * self.net_padding[k] for k in range(len(self.crop_size))]

        # load raw data
        f_raw = h5py.File(os.path.join(self.folder_name, self.train_dataset), 'r')
        data = f_raw['main'][:]
        f_raw.close()
        if self.if_train:
            data = data[:self.train_split] ### 注意调整
        else:
            data=data[-self.test_split:] # (50, 1024, 1024)
        self.dataset = data

        f_label = h5py.File(os.path.join(self.folder_name, train_label), 'r')
        label = f_label['main'][:]
        f_label.close()
        if self.if_train:
            label = label[:self.train_split]
        else:
            label=label[-self.test_split:]
        
        label = seg_widen_border(label, tsz_h=1)
        self.labels = label
        

        # self.dataset = histogram_equalization_3d(data)
        # gamma_trans(data/255.0, gamma=0.25)*255.0 # [0, 255]

        # self.dataset[self.dataset==0] = 10

        # plot_histogram_3d(data, './local_data/code/maisi/predictions/ac3/')

        
        # data = gamma_trans(data, gamma=0.5)
        # Image.fromarray(self.dataset[0]).save(os.path.join('./local_data/code/maisi/predictions', '2.png'))
        # data = gamma_trans(data, gamma=2.0)
        # Image.fromarray(data[0]).convert('L').save(os.path.join('./local_data/code/maisi/predictions', '3.png'))
        
        self.origin_data_shape = list(self.dataset.shape)


 
        # (25, 1024, 1024)，用superhuman里的拼接参数
        # (18, 160, 160)
        # self.stride = [15, 80, 80]
        # self.valid_padding = [4, 48, 48]
        # self.num_zyx = [2, 13, 13]

        
        # num = (length + 2 * padding - window_length) // stride + 1

        self.gt_affs = seg_to_aff(self.labels).astype(np.float32)

        self.dataset = np.pad(self.dataset, ((self.valid_padding[0], self.valid_padding[0]), \
                                                (self.valid_padding[1], self.valid_padding[1]), \
                                                (self.valid_padding[2], self.valid_padding[2])), mode='reflect')
        self.labels = np.pad(self.labels, ((self.valid_padding[0], self.valid_padding[0]), \
                                                (self.valid_padding[1], self.valid_padding[1]), \
                                                (self.valid_padding[2], self.valid_padding[2])), mode='reflect')
        

        # the training dataset size
        self.raw_data_shape = list(self.dataset.shape)

        self.reset_output()
        self.weight_vol = self.get_weight()

        self.num_per_dataset = self.num_zyx[0] * self.num_zyx[1] * self.num_zyx[2]
        self.iters_num = self.num_per_dataset #* len(self.dataset)

    def __getitem__(self, index):

        pre_data = index % self.num_per_dataset
        pos_z = pre_data // (self.num_zyx[1] * self.num_zyx[2])
        pos_xy = pre_data % (self.num_zyx[1] * self.num_zyx[2])
        pos_x = pos_xy // self.num_zyx[2]
        pos_y = pos_xy % self.num_zyx[2]

        # find position因为有crop
        fromz = pos_z * self.stride[0]
        endz = fromz + self.crop_size[0]
        if endz > self.raw_data_shape[0]:
            endz = self.raw_data_shape[0]
            fromz = endz - self.crop_size[0]
        fromy = pos_y * self.stride[1]
        endy = fromy + self.crop_size[1]
        if endy > self.raw_data_shape[1]:
            endy = self.raw_data_shape[1]
            fromy = endy - self.crop_size[1]
        fromx = pos_x * self.stride[2]
        endx = fromx + self.crop_size[2]
        if endx > self.raw_data_shape[2]:
            endx = self.raw_data_shape[2]
            fromx = endx - self.crop_size[2]
        self.pos = [fromz, fromy, fromx]

        imgs = self.dataset[fromz:endz, fromx:endx, fromy:endy].copy()
        lb = self.labels[fromz:endz, fromx:endx, fromy:endy].copy()

        # imgs = exposure.rescale_intensity(imgs)
        imgs = imgs.astype(np.float32) / 255.0

        imgs = np.transpose(imgs, (2, 1, 0)) # maisi

        imgs = imgs[np.newaxis, ...]
        imgs = np.ascontiguousarray(imgs, dtype=np.float32)
        if self.norm2:
            imgs = 2*imgs-1.0

        imageout = torch.from_numpy(imgs)
        if self.if_mask:
            lb_affs = seg_to_aff(lb).astype(np.float32)
            # weightmap = weight_binary_ratio(lb_affs)
            lb_affs = np.ascontiguousarray(lb_affs, dtype=np.float32)
            # weightmap = np.ascontiguousarray(weightmap, dtype=np.float32)
            lb_affs = torch.from_numpy(lb_affs) # (3, 18, 160, 160)
            lb_affs = lb_affs.permute(0, 3, 2, 1) # (3, 160, 160, 32)
            if self.norm2:
                lb_affs = 2*lb_affs-1.0
            return {
                "image": lb_affs,}
        else:
            return {
        "image": imageout,}

    def __len__(self):
        return self.num_per_dataset # self.iters_num

    def reset_output(self):
        if self.if_mask:
            self.outputs = np.zeros(tuple([3]+self.raw_data_shape), dtype=np.float32)
        else:
            self.outputs = np.zeros(tuple([1]+self.raw_data_shape), dtype=np.float32)
        self.weight_map = np.zeros(tuple([1]+self.raw_data_shape), dtype=np.float32)


    def get_weight(self, sigma=0.2, mu=0.0):

        zz, yy, xx = np.meshgrid(np.linspace(-1, 1, self.out_size[0], dtype=np.float32),
                                np.linspace(-1, 1, self.out_size[1], dtype=np.float32),
                                np.linspace(-1, 1, self.out_size[2], dtype=np.float32), indexing='ij')
        dd = np.sqrt(zz * zz + yy * yy + xx * xx)
        weight = 1e-6 + np.exp(-((dd - mu) ** 2 / (2.0 * sigma ** 2)))
        weight = weight[np.newaxis, ...]
        return weight

    def add_vol(self, sub_vol):
        if len(sub_vol.shape) == 3:
            sub_vol = np.transpose(sub_vol, (2, 1, 0)) # maisi
        else:
            sub_vol = np.transpose(sub_vol, (0, 3, 2, 1)) # [3, 160, 160, 18]
        fromz, fromy, fromx = self.pos

        self.outputs[:, fromz:fromz+self.out_size[0], \
                            fromx:fromx+self.out_size[1], \
                            fromy:fromy+self.out_size[2]] += sub_vol * self.weight_vol    #权重
        self.weight_map[:, fromz:fromz+self.out_size[0], \
                            fromx:fromx+self.out_size[1], \
                            fromy:fromy+self.out_size[2]] += self.weight_vol


    def get_results(self):  # 去掉padding
        self.outputs = self.outputs / self.weight_map
        if self.valid_padding[0] == 0:
            self.outputs = self.outputs[:, :, \
                                            self.valid_padding[1]:-self.valid_padding[1], \
                                            self.valid_padding[2]:-self.valid_padding[2]]
        else:
            self.outputs = self.outputs[:, self.valid_padding[0]:-self.valid_padding[0], \
                                            self.valid_padding[1]:-self.valid_padding[1], \
                                            self.valid_padding[2]:-self.valid_padding[2]]
        return self.outputs

    def get_gt_affs(self):
        return self.gt_affs.copy()

    def get_gt_lb(self):
        return self.gt_affs

    def get_raw_data(self):
        out = self.dataset.copy()
        return out[self.valid_padding[0]:-self.valid_padding[0], \
                    self.valid_padding[1]:-self.valid_padding[1], \
                    self.valid_padding[2]:-self.valid_padding[2]]

