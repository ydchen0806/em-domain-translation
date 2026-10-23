from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys
import torch
import h5py
import random

import glob
import pickle
import numpy as np
from torch.utils.data import Dataset

from augmentation import Flip,Elastic,Grayscale,Rotate,Rescale
from data.consistency_aug_perturbations import GaussNoise,GaussBlur,Cutout

from data.data_segmentation import seg_widen_border, weight_binary_ratio
from data.data_affinity import seg_to_aff



def center_crop(image, det_shape=[18, 160, 160]):
    # To prevent overflow
    image = np.pad(image, ((2,2),(20,20),(20,20)), mode='reflect')
    src_shape = image.shape
    shift0 = (src_shape[0] - det_shape[0]) // 2
    shift1 = (src_shape[1] - det_shape[1]) // 2
    shift2 = (src_shape[2] - det_shape[2]) // 2
    assert shift0 > 0 or shift1 > 0 or shift2 > 0, "overflow in center-crop"
    image = image[shift0:shift0+det_shape[0], shift1:shift1+det_shape[1], shift2:shift2+det_shape[2]]
    return image

class SingleDataset_lb(Dataset):
    def __init__(self, root_dir, train_dataset, length=300,if_train=True,aug=True,strong_aug=True,only_flip=False):
        self.folder_name = root_dir
        self.length=length
        self.aug=aug
        self.strong_aug=strong_aug
        self.only_flip=only_flip
        
        # self.train_datasets = ['cremiA_inputs_interp.h5', 'cremiB_inputs_interp.h5', 'cremiC_inputs_interp.h5']
        # self.train_labels = ['cremiA_labels.h5', 'cremiB_labels.h5', 'cremiC_labels.h5']
        
        self.train_dataset = train_dataset
        
        if self.train_dataset == 'cremiC_inputs_interp.h5':
            self.train_split=100
            self.test_split=25
            train_label='cremiC_labels.h5'
        elif self.train_dataset == 'AC4_inputs.h5':
            self.train_split=80 #50
            self.test_split=20 #50
            train_label='AC4_labels.h5'
        elif self.train_dataset == 'AC3_inputs.h5':
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

	
        f_raw = h5py.File(os.path.join(self.folder_name, self.train_dataset), 'r')
        data = f_raw['main'][:]
        f_raw.close()
        if self.if_train:
            data = data[:self.train_split]
        else:
            data=data[-self.test_split:]
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

        self.raw_data_shape = list(self.dataset.shape)

        self.augs_init()

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        used_data = self.dataset
        used_label = self.labels

        random_z = random.randint(0, self.raw_data_shape[0]-self.crop_from_origin[0])
        random_y = random.randint(0, self.raw_data_shape[1]-self.crop_from_origin[1])
        random_x = random.randint(0, self.raw_data_shape[2]-self.crop_from_origin[2])
        imgs = used_data[random_z:random_z+self.crop_from_origin[0], \
						random_y:random_y+self.crop_from_origin[1], \
						random_x:random_x+self.crop_from_origin[2]].copy()
        lb = used_label[random_z:random_z+self.crop_from_origin[0], \
						random_y:random_y+self.crop_from_origin[1], \
						random_x:random_x+self.crop_from_origin[2]].copy()

        imgs = imgs.astype(np.float32) / 255.0
        data = {'image': imgs, 'label': lb}
        if self.if_train and self.aug and np.random.rand() < 0.5:
            data = self.augs_mix(data) 
        imgs = data['image']
        lb = data['label']

        imgs = center_crop(imgs, det_shape=self.crop_size)
        
        imgs = np.transpose(imgs, (2, 1, 0))
        imgs = imgs[np.newaxis, ...]
        imgs = np.ascontiguousarray(imgs, dtype=np.float32)

        imageout = torch.from_numpy(imgs)

        lb = center_crop(lb, det_shape=self.crop_size)
        lb_affs = seg_to_aff(lb).astype(np.float32)
        lb_affs = np.ascontiguousarray(lb_affs, dtype=np.float32)
        lb_affs = torch.from_numpy(lb_affs)
        lb_affs=lb_affs.permute(0, 3, 2, 1)

        return {
                "image": imageout,
                "label": lb_affs,
                "num": list([random_z, random_y+self.padding, random_x+self.padding])
                }


    

    def augs_init(self):
        # https://zudi-lin.github.io/pytorch_connectomics/build/html/notes/dataloading.html#data-augmentation
        self.aug_rotation = Rotate(p=0.5)
        self.aug_rescale = Rescale(p=0.5)
        self.aug_flip = Flip(p=1.0, do_ztrans=0)
        self.aug_elastic = Elastic(p=0.75, alpha=16, sigma=4.0)
        self.aug_grayscale = Grayscale(p=0.75)

        self.per_gaussnoise = GaussNoise(min_std=0.01, max_std=0.2, norm_mode='trunc')
        self.per_gaussblur = GaussBlur()
        self.per_cutout = Cutout()


    def augs_single(self, data):
        random_id = np.random.randint(1, 6)
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

    def augs_mix(self, data):
        if self.if_filp_aug and random.random() > 0.5:
            data = self.aug_flip(data)
        

        # strong augmentation
        if self.strong_aug:
            if random.random() > 0.5:
                data['image'] = self.per_gaussnoise(data['image'])
            if random.random() > 0.5:
                data['image'] = self.per_gaussblur(data['image'])
            if random.random() > 0.5:
                data['image'] = self.per_cutout(data['image'])
        if not self.only_flip:
            if self.if_rotation_aug and random.random() > 0.5:
                data = self.aug_rotation(data)
            if self.if_elastic_aug and random.random() > 0.5:
                data = self.aug_elastic(data)
            if self.if_intensity_aug and random.random() > 0.5:
                data = self.aug_grayscale(data)

        return data



class CondTargetDataset_sampling(Dataset):
    def __init__(self, 
                 src_dir, src_dataset, tgt_dir, tgt_dataset, 
                 sampling_dir, crop_size = [18, 160, 160],
                 length=300,if_train=True,strong_aug=True,src_aug=True):


        self.strong_aug = strong_aug
        self.src_aug = src_aug 
        self.sampling_dir = sampling_dir  

        if src_dataset == 'cremiC_inputs_interp.h5' and tgt_dataset=='AC4_inputs.h5':
            self.train_split=[100,80]
            self.test_split=[25,20]
            src_label='cremiC_labels.h5'
        elif src_dataset == 'AC4_inputs.h5' and tgt_dataset=='cremiC_inputs_interp.h5':
            self.train_split=[80,100]
            self.test_split=[20,25]
            src_label='AC4_labels.h5'
        else: 
            raise ValueError("Invalid dataset name")
        

        self.crop_size = crop_size
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
        self.if_elastic_aug = False
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
        self.tgt_dataset = data


        f_label = h5py.File(os.path.join(src_dir, src_label), 'r')
        label = f_label['main'][:]
        f_label.close()
        if self.if_train:
            label = label[:self.train_split[0]]
        else:
            label = label[-self.test_split[0]:]
        label = seg_widen_border(label, tsz_h=1)
        self.labels = label

		# the training dataset size
        self.raw_data_shape_src = list(self.src_dataset.shape)
        self.raw_data_shape_tgt = list(self.tgt_dataset.shape)

        
		# augmentation initoalization
        self.augs_init()

    def __len__(self):
        return self.length

    def __getitem__(self, index):



        
        random_z = random.randint(0, self.raw_data_shape_src[0]-self.crop_from_origin[0])
        random_y = random.randint(0, self.raw_data_shape_src[1]-self.crop_from_origin[1])
        random_x = random.randint(0, self.raw_data_shape_src[2]-self.crop_from_origin[2])
        imgs = self.src_dataset[random_z:random_z+ self.crop_from_origin[0], \
						random_y:random_y+ self.crop_from_origin[1], \
						random_x:random_x+ self.crop_from_origin[2]].copy()

        lb = self.labels[random_z:random_z+self.crop_from_origin[0], \
						random_y:random_y+self.crop_from_origin[1], \
						random_x:random_x+self.crop_from_origin[2]].copy()
        
        random_z2 = random.randint(0, self.raw_data_shape_tgt[0]-self.crop_from_origin[0])
        random_y2 = random.randint(0, self.raw_data_shape_tgt[1]-self.crop_from_origin[1])
        random_x2 = random.randint(0, self.raw_data_shape_tgt[2]-self.crop_from_origin[2])
        cond = self.tgt_dataset[random_z2:random_z2+self.crop_from_origin[0], \
                        random_y2:random_y2+self.crop_from_origin[1], \
                        random_x2:random_x2+self.crop_from_origin[2]].copy()


        imgs = imgs.astype(np.float32) / 255.0
        cond = cond.astype(np.float32) / 255.0
        sample = sample.astype(np.float32)


        lb_tmp = lb.copy()

        data = {'image': imgs, 'label': lb, 'sample': sample}
        if self.src_aug:
            data = self.augs_mix(data) 
        imgs = data['image']
        lb = data['label']
        sample = data['sample']

        data = {'image': cond, 'label': lb_tmp, 'sample': cond}
        if self.if_train:
            data = self.augs_mix_consist(data) 
        imgs_tgt = data['image']
        cond = data['sample']

        imgs = center_crop(imgs, det_shape=self.crop_size)
        lb = center_crop(lb, det_shape=self.crop_size)
        
        imgs_tgt = center_crop(imgs_tgt, det_shape=self.crop_size)
        cond = center_crop(cond, det_shape=self.crop_size)

        lb_affs = seg_to_aff(lb).astype(np.float32)


        imgs = np.transpose(imgs, (2, 1, 0))
        sample = np.transpose(sample, (2, 1, 0))
        imgs_tgt = np.transpose(imgs_tgt, (2, 1, 0))
        cond = np.transpose(cond, (2, 1, 0))
        lb_affs = np.transpose(lb_affs, (0, 3, 2, 1))

        weightmap = weight_binary_ratio(lb_affs)


        imgs = imgs[np.newaxis, ...] 
        imgs = np.ascontiguousarray(imgs, dtype=np.float32)


        lb_affs = np.ascontiguousarray(lb_affs, dtype=np.float32)


        imgs_tgt = imgs_tgt[np.newaxis, ...]
        imgs_tgt = np.ascontiguousarray(imgs_tgt, dtype=np.float32)
        
        cond = cond[np.newaxis, ...]
        cond = np.ascontiguousarray(cond, dtype=np.float32)

        sample = sample[np.newaxis, ...]
        sample = np.ascontiguousarray(sample, dtype=np.float32)


        weightmap = np.ascontiguousarray(weightmap, dtype=np.float32)


        imageout = torch.from_numpy(imgs)
        lb_affs = torch.from_numpy(lb_affs)
        imageout_tgt = torch.from_numpy(imgs_tgt)
        cond = torch.from_numpy(cond)
        weightmap = torch.from_numpy(weightmap)
        sample = torch.from_numpy(sample)

        return {
        "image": imageout,
        "sample": sample,
        "tgt":imageout_tgt,
        "cond":cond,
        "lb":lb_affs,
        "weight":weightmap,
        "num": list([random_z, random_y, random_x])}

    

    def augs_init(self):
        # https://zudi-lin.github.io/pytorch_connectomics/build/html/notes/dataloading.html#data-augmentation
        self.aug_rotation = Rotate(p=0.5)
        self.aug_rescale = Rescale(p=0.5)
        self.aug_flip = Flip(p=1.0, do_ztrans=0)
        self.aug_elastic = Elastic(p=0.75, alpha=16, sigma=4.0)
        self.aug_grayscale = Grayscale(p=0.75)

        self.per_gaussnoise = GaussNoise(min_std=0.01, max_std=0.2, norm_mode='trunc')
        self.per_gaussblur = GaussBlur()
        self.per_cutout = Cutout()

    def augs_single(self, data):
        random_id = np.random.randint(1, 5+1)
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

    def augs_mix(self, data):
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
        # strong augmentation
        if self.strong_aug and random.random() > 0.8:
            random_id = np.random.randint(1, 4+1)
            if random_id == 1:
                data['image'] = self.per_gaussnoise(data['image'])
            if random_id == 2:
                data['image'] = self.per_gaussblur(data['image'])
            if random_id == 3:
                data['image'] = self.per_cutout(data['image'])
            if random_id == 4:
                if random.random() > 0.8:
                    data['image'] = self.per_gaussnoise(data['image'])
                if random.random() > 0.8:
                    data['image'] = self.per_gaussblur(data['image'])
                if random.random() > 0.8:
                    data['image'] = self.per_cutout(data['image'])
        return data
    
    def augs_mix_consist(self, data):
        if random.random() > 0.5:
            data = self.aug_flip(data)
        if random.random() > 0.5:
            data = self.aug_rotation(data)
        
        random_id = np.random.randint(1, 5+1)
        if random_id == 1:
            data = self.aug_grayscale(data) 
        if random_id == 2:
            data['image'] = self.per_gaussnoise(data['image'])
        if random_id == 3:
            data['image'] = self.per_gaussblur(data['image'])
        if random_id == 4:
            data['image'] = self.per_cutout(data['image'])
        if random_id == 5:
            if random.random() > 0.5:
                data = self.aug_grayscale(data) 
            if random.random() > 0.8:
                data['image'] = self.per_gaussnoise(data['image'])
            if random.random() > 0.8:
                data['image'] = self.per_gaussblur(data['image'])
            if random.random() > 0.8:
                data['image'] = self.per_cutout(data['image'])
        return data



class CondTargetDataset_lb(Dataset):
    def __init__(self, src_dir, src_dataset, tgt_dir, tgt_dataset, crop_size=[18, 160, 160],length=300,if_train=True,strong_aug=True,src_aug=True,tgt_aug=True):

        self.strong_aug = strong_aug
        self.src_aug = src_aug   
        self.tgt_aug = tgt_aug   

        if src_dataset == 'cremiC_inputs_interp.h5' and tgt_dataset=='AC4_inputs.h5':
            self.train_split=[100,80]
            self.test_split=[25,20]
            src_label = 'cremiC_labels.h5'
            tgt_label = 'AC4_labels.h5'
        elif src_dataset == 'AC4_inputs.h5' and tgt_dataset=='cremiC_inputs_interp.h5':
            self.train_split=[80,100]
            self.test_split=[20,25]
            src_label = 'AC4_labels.h5'
            tgt_label = 'cremiC_labels.h5'
        else: 
            raise ValueError("Invalid dataset name")
        

        self.crop_size = crop_size
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
        self.src_dataset = data


        f_raw = h5py.File(os.path.join(tgt_dir, tgt_dataset), 'r')
        data = f_raw['main'][:]
        f_raw.close()
        if self.if_train:
            data = data[:self.train_split[1]]
        else:
            data=data[-self.test_split[1]:]
        self.tgt_dataset = data


        f_label = h5py.File(os.path.join(src_dir, src_label), 'r')
        label = f_label['main'][:]
        f_label.close()
        if self.if_train:
            label = label[:self.train_split[0]]
        else:
            label = label[-self.test_split[0]:]
        label = seg_widen_border(label, tsz_h=1)
        self.labels = label


        f_label = h5py.File(os.path.join(tgt_dir, tgt_label), 'r')
        label = f_label['main'][:]
        f_label.close()
        if self.if_train:
            label = label[:self.train_split[1]]
        else:
            label = label[-self.test_split[1]:]
        label = seg_widen_border(label, tsz_h=1)
        self.labels_tgt = label


		# the training dataset size
        self.raw_data_shape_src = list(self.src_dataset.shape)
        self.raw_data_shape_tgt = list(self.tgt_dataset.shape)


		# augmentation initoalization
        self.augs_init()

    def __len__(self):
        if not self.visual:
            return self.length
        else:
            return len(self.files)

    def __getitem__(self, index):

        
        random_z = random.randint(0, self.raw_data_shape_src[0]-self.crop_from_origin[0])
        random_y = random.randint(0, self.raw_data_shape_src[1]-self.crop_from_origin[1])
        random_x = random.randint(0, self.raw_data_shape_src[2]-self.crop_from_origin[2])
        

        imgs = self.src_dataset[random_z:random_z+self.crop_from_origin[0], \
						random_y:random_y+self.crop_from_origin[1], \
						random_x:random_x+self.crop_from_origin[2]].copy()

        lb = self.labels[random_z:random_z+self.crop_from_origin[0], \
						random_y:random_y+self.crop_from_origin[1], \
						random_x:random_x+self.crop_from_origin[2]].copy()
        
        random_z2 = random.randint(0, self.raw_data_shape_tgt[0]-self.crop_from_origin[0])
        random_y2 = random.randint(0, self.raw_data_shape_tgt[1]-self.crop_from_origin[1])
        random_x2 = random.randint(0, self.raw_data_shape_tgt[2]-self.crop_from_origin[2])
        cond = self.tgt_dataset[random_z2:random_z2+self.crop_from_origin[0], \
                        random_y2:random_y2+self.crop_from_origin[1], \
                        random_x2:random_x2+self.crop_from_origin[2]].copy()
        lb_tgt = self.labels_tgt[random_z2:random_z2+self.crop_from_origin[0], \
                        random_y2:random_y2+self.crop_from_origin[1], \
                        random_x2:random_x2+self.crop_from_origin[2]].copy()

        imgs = imgs.astype(np.float32) / 255.0
        cond = cond.astype(np.float32) / 255.0

        data = {'image': imgs, 'label': lb}
        if self.if_train and self.src_aug:
            data = self.augs_mix(data) 
        imgs = data['image']
        lb = data['label']


        data = {'image': cond, 'label': lb}
        if self.if_train and self.tgt_aug:
            data = self.augs_mix_consist(data)
        imgs_tgt = data['image']


        imgs = center_crop(imgs, det_shape=self.crop_size)
        lb = center_crop(lb, det_shape=self.crop_size)
        imgs_tgt = center_crop(imgs_tgt, det_shape=self.crop_size)
        cond = center_crop(cond, det_shape=self.crop_size)
        lb_tgt = center_crop(lb_tgt, det_shape=self.crop_size)

        lb_affs = seg_to_aff(lb).astype(np.float32)
        lb_affs_tgt = seg_to_aff(lb_tgt).astype(np.float32)


        imgs = np.transpose(imgs, (2, 1, 0))
        imgs_tgt = np.transpose(imgs_tgt, (2, 1, 0))
        cond = np.transpose(cond, (2, 1, 0))
        lb_affs = np.transpose(lb_affs, (0, 3, 2, 1))
        lb_affs_tgt = np.transpose(lb_affs_tgt, (0, 3, 2, 1))

        weightmap = weight_binary_ratio(lb_affs)

        imgs = imgs[np.newaxis, ...] 
        imgs = np.ascontiguousarray(imgs, dtype=np.float32)

        lb_affs = np.ascontiguousarray(lb_affs, dtype=np.float32)
        lb_affs_tgt = np.ascontiguousarray(lb_affs_tgt, dtype=np.float32)

        imgs_tgt = imgs_tgt[np.newaxis, ...]
        imgs_tgt = np.ascontiguousarray(imgs_tgt, dtype=np.float32)
        cond = cond[np.newaxis, ...]
        cond = np.ascontiguousarray(cond, dtype=np.float32)


        weightmap = np.ascontiguousarray(weightmap, dtype=np.float32)

        imageout = torch.from_numpy(imgs)
        lb_affs = torch.from_numpy(lb_affs)
        imageout_tgt = torch.from_numpy(imgs_tgt)
        lb_affs_tgt = torch.from_numpy(lb_affs_tgt)
        cond = torch.from_numpy(cond)
        weightmap = torch.from_numpy(weightmap)

        return {
        "image": imageout,
        "tgt":imageout_tgt,
        "cond":cond,
        "lb":lb_affs,
        "lb_tgt":lb_affs_tgt,
        "weight":weightmap,
        "num": list([random_z, random_y, random_x])}

    

    def augs_init(self):
        # https://zudi-lin.github.io/pytorch_connectomics/build/html/notes/dataloading.html#data-augmentation
        self.aug_rotation = Rotate(p=0.5)
        self.aug_rescale = Rescale(p=0.5)
        self.aug_flip = Flip(p=1.0, do_ztrans=0)
        self.aug_elastic = Elastic(p=0.75, alpha=16, sigma=4.0)
        self.aug_grayscale = Grayscale(p=0.75)

        self.per_gaussnoise = GaussNoise(min_std=0.01, max_std=0.2, norm_mode='trunc')
        self.per_gaussblur = GaussBlur()
        self.per_cutout = Cutout()

    def augs_single(self, data):
        random_id = np.random.randint(1, 5+1)
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

    def augs_mix(self, data):
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
        # strong augmentation
        if self.strong_aug:
            if random.random() > 0.5:
                data['image'] = self.per_gaussnoise(data['image'])
            if random.random() > 0.5:
                data['image'] = self.per_gaussblur(data['image'])
            if random.random() > 0.5:
                data['image'] = self.per_cutout(data['image'])
        return data
    
    def augs_mix_consist(self, data):
        # strong augmentation
        if random.random() > 0.5:
            data['image'] = self.per_gaussnoise(data['image'])
        if random.random() > 0.5:
            data['image'] = self.per_gaussblur(data['image'])
        if random.random() > 0.5:
            data['image'] = self.per_cutout(data['image'])
        return data
