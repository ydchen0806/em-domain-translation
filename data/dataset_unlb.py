from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys
import torch
import h5py
import random
import numpy as np
from torch.utils.data import Dataset

from augmentation import Flip
from augmentation import Elastic
from augmentation import Grayscale
from augmentation import Rotate
from augmentation import Rescale
# from utils.utils import center_crop
from data.data_segmentation import seg_widen_border, weight_binary_ratio
from data.data_affinity import seg_to_aff

# from utils.affinity_ours import gen_affs_mutex_3d
def center_crop(image, det_shape=[18, 160, 160]):
    # To prevent overflow
    image = np.pad(image, ((2,2),(20,20),(20,20)), mode='reflect')
    src_shape = image.shape
    shift0 = (src_shape[0] - det_shape[0]) // 2 # 每边的相差
    shift1 = (src_shape[1] - det_shape[1]) // 2
    shift2 = (src_shape[2] - det_shape[2]) // 2
    assert shift0 > 0 or shift1 > 0 or shift2 > 0, "overflow in center-crop" # 断言，表达式为false报错
    image = image[shift0:shift0+det_shape[0], shift1:shift1+det_shape[1], shift2:shift2+det_shape[2]]
    return image

class SingleDataset(Dataset):
    def __init__(self, root_dir, train_dataset,length=300,if_train=True,if_mask=False,norm2=False,aug=True):
        self.norm2=norm2
        self.folder_name = root_dir
        self.length=length
        self.aug=aug
        
        # self.train_datasets = ['cremiA_inputs_interp.h5', 'cremiB_inputs_interp.h5', 'cremiC_inputs_interp.h5']
        # self.train_labels = ['cremiA_labels.h5', 'cremiB_labels.h5', 'cremiC_labels.h5']
        
        self.train_dataset = train_dataset
        
        if self.train_dataset == 'cremiC_inputs_interp.h5': # (125, 1024, 1024)
            self.train_split=100
            self.test_split=25
            train_label='cremiC_labels.h5'
        elif self.train_dataset == 'AC4_inputs.h5': # (100, 1024, 1024)
            self.train_split=80 #50
            self.test_split=20 #50
            train_label='AC4_labels.h5'
        elif self.train_dataset == 'AC3_inputs.h5': # (256, 1024, 1024) ##还没想好
            self.train_split=50
            self.test_split=50
            train_label='AC3_labels.h5'
        else: 
            raise ValueError("Invalid dataset name")
        

        self.crop_size = [18, 160, 160]
        
        self.net_padding = [0, 0, 0]
		
        self.out_size = [self.crop_size[k] - 2 * self.net_padding[k] for k in range(len(self.crop_size))] # 减掉两侧padding

        self.crop_from_origin = [0, 0, 0]
        self.padding = 20
        self.crop_from_origin[0] = self.crop_size[0]
        self.crop_from_origin[1] = self.crop_size[1] + 2 * self.padding
        self.crop_from_origin[2] = self.crop_size[2] + 2 * self.padding

        self.if_scale_aug = False
        self.if_filp_aug = True
        self.if_elastic_aug = True
        self.if_intensity_aug = True
        self.if_rotation_aug = True

        self.if_train=if_train
        self.if_mask=if_mask

	
        f_raw = h5py.File(os.path.join(self.folder_name, self.train_dataset), 'r')
        data = f_raw['main'][:]
        f_raw.close()
        if self.if_train:
            data = data[:self.train_split]
        else:
            data=data[-self.test_split:]
        self.dataset = data # (100,1024,1024)

        f_label = h5py.File(os.path.join(self.folder_name, train_label), 'r')
        label = f_label['main'][:]
        f_label.close()
        if self.if_train:
            label = label[:self.train_split]
        else:
            label=label[-self.test_split:]
        label = seg_widen_border(label, tsz_h=1)
        self.labels = label

		# the training dataset size
        self.raw_data_shape = list(self.dataset.shape)

        

		# augmentation initoalization
        self.augs_init()

    def __len__(self):
        # return int(sys.maxsize)
        return self.length # 100*1024*1024/18/160/160

    def __getitem__(self, index):
		# 随机选择一个dataset如果contain many datasets
        # k = random.randint(0, len(self.train_datasets)-1)
        used_data = self.dataset
        used_label = self.labels

		# 全部减去crop
        random_z = random.randint(0, self.raw_data_shape[0]-self.crop_from_origin[0])
        random_y = random.randint(0, self.raw_data_shape[1]-self.crop_from_origin[1])
        random_x = random.randint(0, self.raw_data_shape[2]-self.crop_from_origin[2])
        imgs = used_data[random_z:random_z+self.crop_from_origin[0], \
						random_y:random_y+self.crop_from_origin[1], \
						random_x:random_x+self.crop_from_origin[2]].copy()
        lb = used_label[random_z:random_z+self.crop_from_origin[0], \
						random_y:random_y+self.crop_from_origin[1], \
						random_x:random_x+self.crop_from_origin[2]].copy()

        imgs = imgs.astype(np.float32) / 255.0 # [0,1]
        data = {'image': imgs, 'label': lb}
		# p=0.5 for augmentation
        if self.if_train and self.aug and np.random.rand() < 0.5:
            data = self.augs_mix(data) 
        imgs = data['image']
        lb = data['label']

        imgs = center_crop(imgs, det_shape=self.crop_size)
        
        # (z, y, x) -> (x, y, z)
        imgs = np.transpose(imgs, (2, 1, 0))


        imgs = imgs[np.newaxis, ...] # (1, 160, 160, 18)
        imgs = np.ascontiguousarray(imgs, dtype=np.float32)

        # imgs = 2*imgs-1 # [-1,1]
        imageout = torch.from_numpy(imgs)
        if self.if_mask:
            lb = center_crop(lb, det_shape=self.crop_size)
            lb_affs = seg_to_aff(lb).astype(np.float32)
            lb_affs = np.ascontiguousarray(lb_affs, dtype=np.float32)
            lb_affs = torch.from_numpy(lb_affs) # (3, 18, 160, 160)
            lb_affs=lb_affs.permute(0, 3, 2, 1) # (3, 160, 160, 18)
            return {
        "image": lb_affs,}
        else:
            return {
        "image": imageout,}

    

    def augs_init(self):
        # https://zudi-lin.github.io/pytorch_connectomics/build/html/notes/dataloading.html#data-augmentation
        self.aug_rotation = Rotate(p=0.5) # 借助cv2
        self.aug_rescale = Rescale(p=0.5)
        self.aug_flip = Flip(p=1.0, do_ztrans=0)
        self.aug_elastic = Elastic(p=0.75, alpha=16, sigma=4.0) # warp.py
        self.aug_grayscale = Grayscale(p=0.75) # if_intensity_aug

	# TO DO增强的几种类型，感觉是随机做增强?
    def augs_single(self, data):
        random_id = np.random.randint(1, 5+1)#随机
        if random_id == 1:
            data = self.aug_rotation(data)
        elif random_id == 2:
            data = self.aug_rescale(data)
        elif random_id == 3:
            data = self.aug_flip(data)
        elif random_id == 4:
            data = self.aug_elastic(data)
        elif random_id == 5:
            data = self.aug_grayscale(data)
        else:
            raise NotImplementedError
        return data

    def augs_mix(self, data): # 如果true且掷骰子成功...
        if self.if_filp_aug and random.random() > 0.5:
            data = self.aug_flip(data)
        if self.if_rotation_aug and random.random() > 0.5:
            data = self.aug_rotation(data)
        if self.if_scale_aug and random.random() > 0.5:
            data = self.aug_rescale(data)
        if self.if_elastic_aug and random.random() > 0.5:
            data = self.aug_elastic(data)
        if self.if_intensity_aug and random.random() > 0.5:
            data = self.aug_grayscale(data)
        return data

class CondTargetDataset(Dataset):
    def __init__(self, src_dir, src_dataset, tgt_dir, tgt_dataset, length=300,if_train=True):

        
        if src_dataset == 'cremiC_inputs_interp.h5' and tgt_dataset=='AC4_inputs.h5':
             # (125, 1024, 1024)
            self.train_split=[75,50]
            self.test_split=[50,50]
        # elif self.train_dataset == 'AC4_inputs.h5': # (100, 1024, 1024)
        #     self.train_split=50
        #     self.test_split=50
        # elif self.train_dataset == 'AC3_inputs.h5': # (256, 1024, 1024) ##还没想好
        #     self.train_split=50
        #     self.test_split=50
        else: 
            raise ValueError("Invalid dataset name")
        

        self.crop_size = [32, 160, 160]
        self.length=length
        self.net_padding = [0, 0, 0]
		
        self.out_size = [self.crop_size[k] - 2 * self.net_padding[k] for k in range(len(self.crop_size))] # 减掉两侧padding

        self.crop_from_origin = [0, 0, 0]
        self.padding = 20
        self.crop_from_origin[0] = self.crop_size[0]
        self.crop_from_origin[1] = self.crop_size[1] + 2 * self.padding
        self.crop_from_origin[2] = self.crop_size[2] + 2 * self.padding

        self.if_scale_aug = False
        self.if_filp_aug = True
        self.if_elastic_aug = True
        self.if_intensity_aug = True
        self.if_rotation_aug = True

        self.if_train=if_train

	
        f_raw = h5py.File(os.path.join(src_dir, src_dataset), 'r')
        data = f_raw['main'][:]
        f_raw.close()
        if self.if_train:
            data = data[:self.train_split[0]]
        else:
            data=data[-self.test_split[0]:]
        self.src_dataset = data # (75,1024,1024)


        f_raw = h5py.File(os.path.join(tgt_dir, tgt_dataset), 'r')
        data = f_raw['main'][:]
        f_raw.close()
        if self.if_train:
            data = data[:self.train_split[1]]
        else:
            data=data[-self.test_split[1]:]
        self.tgt_dataset = data # (50,1024,1024)


        # f_label = h5py.File(cfg.DATA.data_dir_src_lb, 'r')
        # label = f_label['main'][:]
        # f_label.close()
        # label = label[:self.train_split]
        # if self.if_dilate:
        #     label = seg_widen_border(label, tsz_h=1)
        # self.labels = label

		# the training dataset size
        self.raw_data_shape_src = list(self.src_dataset.shape)
        self.raw_data_shape_tgt = list(self.tgt_dataset.shape)

        

		# augmentation initoalization
        self.augs_init()

    def __len__(self):
        # return int(sys.maxsize)
        return self.length # 100*1024*1024/18/160/160

    def __getitem__(self, index):
		# 随机选择一个dataset如果contain many datasets
        # k = random.randint(0, len(self.train_datasets)-1)
        # used_data = self.dataset
        # used_label = self.labels

		# 全部减去crop
        random_z = random.randint(0, self.raw_data_shape_src[0]-self.crop_from_origin[0])
        random_y = random.randint(0, self.raw_data_shape_src[1]-self.crop_from_origin[1])
        random_x = random.randint(0, self.raw_data_shape_src[2]-self.crop_from_origin[2])
        imgs = self.src_dataset[random_z:random_z+self.crop_from_origin[0], \
						random_y:random_y+self.crop_from_origin[1], \
						random_x:random_x+self.crop_from_origin[2]].copy()
        
        random_z = random.randint(0, self.raw_data_shape_tgt[0]-self.crop_from_origin[0])
        random_y = random.randint(0, self.raw_data_shape_tgt[1]-self.crop_from_origin[1])
        random_x = random.randint(0, self.raw_data_shape_tgt[2]-self.crop_from_origin[2])
        imgs_tgt = self.tgt_dataset[random_z:random_z+self.crop_from_origin[0], \
                        random_y:random_y+self.crop_from_origin[1], \
                        random_x:random_x+self.crop_from_origin[2]].copy()

        imgs = imgs.astype(np.float32) / 255.0 # [0,1]
        imgs_tgt = imgs_tgt.astype(np.float32) / 255.0 # [0,1]

        # 【考虑先tgt不增强吧】
        data = {'image': imgs}
		# p=0.5 for augmentation
        if self.if_train and np.random.rand() < 0.5:
            data = self.augs_mix(data) 
        imgs = data['image']

        imgs = center_crop(imgs, det_shape=self.crop_size)
        imgs_tgt = center_crop(imgs_tgt, det_shape=self.crop_size)
        # (z, y, x) -> (x, y, z)
        imgs = np.transpose(imgs, (2, 1, 0))
        imgs_tgt = np.transpose(imgs_tgt, (2, 1, 0))


        imgs = imgs[np.newaxis, ...] # (1, 160, 160, 18)
        imgs = np.ascontiguousarray(imgs, dtype=np.float32)
        imgs_tgt = imgs_tgt[np.newaxis, ...]
        imgs_tgt = np.ascontiguousarray(imgs_tgt, dtype=np.float32)

        # imgs = 2*imgs-1 # [-1,1]
        imageout = torch.from_numpy(imgs)
        imageout_tgt = torch.from_numpy(imgs_tgt)
        return {
        "image": imageout,"cond":imageout_tgt,}

    

    def augs_init(self):
        # https://zudi-lin.github.io/pytorch_connectomics/build/html/notes/dataloading.html#data-augmentation
        self.aug_rotation = Rotate(p=0.5) # 借助cv2
        self.aug_rescale = Rescale(p=0.5)
        self.aug_flip = Flip(p=1.0, do_ztrans=0)
        self.aug_elastic = Elastic(p=0.75, alpha=16, sigma=4.0) # warp.py
        self.aug_grayscale = Grayscale(p=0.75) # if_intensity_aug

	# TO DO增强的几种类型，感觉是随机做增强?
    def augs_single(self, data):
        random_id = np.random.randint(1, 5+1)#随机
        if random_id == 1:
            data = self.aug_rotation(data)
        elif random_id == 2:
            data = self.aug_rescale(data)
        elif random_id == 3:
            data = self.aug_flip(data)
        elif random_id == 4:
            data = self.aug_elastic(data)
        elif random_id == 5:
            data = self.aug_grayscale(data)
        else:
            raise NotImplementedError
        return data

    def augs_mix(self, data): # 如果true且掷骰子成功...
        if self.if_filp_aug and random.random() > 0.5:
            data = self.aug_flip(data)
        if self.if_rotation_aug and random.random() > 0.5:
            data = self.aug_rotation(data)
        if self.if_scale_aug and random.random() > 0.5:
            data = self.aug_rescale(data)
        if self.if_elastic_aug and random.random() > 0.5:
            data = self.aug_elastic(data)
        if self.if_intensity_aug and random.random() > 0.5:
            data = self.aug_grayscale(data)
        return data