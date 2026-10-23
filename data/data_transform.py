from __future__ import print_function, division
from typing import Optional, Tuple

import torch
import scipy
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import remove_small_holes
from skimage.measure import label as label_cc # avoid namespace conflict避免名称冲突
                                                # 实现连通区域标记

from data.data_misc import get_padsize, array_unpad

# 距离变换
def distance_transform_vol(label, quantize=True, mode='2d'):
    if mode == '3d':
        # calculate 3d distance transform
        vol_distance, vol_semantic = distance_transform(    #注意这是自己写的函数，距离变换
            label, resolution=(1.0, 1.0, 1.0))
        if quantize:
            vol_distance = energy_quantize(vol_distance)    #量化
        return vol_distance
    # 2d:
    vol_distance = []
    vol_semantic = []
    for i in range(label.shape[0]):
        label_img = label[i].copy()
        distance, semantic = distance_transform(label_img)
        vol_distance.append(distance)
        vol_semantic.append(semantic)

    vol_distance = np.stack(vol_distance, 0)#沿着0轴堆叠
    vol_semantic = np.stack(vol_semantic, 0)
    if quantize:
        vol_distance = energy_quantize(vol_distance)
        
    return vol_distance

# 欧氏距离变换，(去小孔，unpad)
def distance_transform(label, 
                       bg_value: float = -1.0, 
                       relabel: bool = True, 
                       padding: bool = False,
                       resolution: Tuple[int, ...] = (1.0, 1.0)):
    """Euclidean distance transform (DT or EDT).
    """
    eps = 1e-6
    pad_size = 2

    if relabel:
        label = label_cc(label)
    '''
    input : Image to label 需要被标记的图片，输入的数据结构不作要求
    background : 选择背景像素，指定像素作为背景，全部相同像素标记为0
    return_num : 是一个bool值，如果为True的话返回值是一个元组(labels,num)；如果为False的话就只返回labels
    connectivity : 连通性，（二维图像）1=4邻接、2=8邻接.默认是2
    Returns:
    labels : 和input形状一样，但是数值是标记号，所以这是一个已经标记的图片
    num : 标记的种类数，如果输出0则只有背景，如果输出2则有两个种类或者说是连通域
    [[1, 0, 0, 0, 0],
       [0, 1, 8, 8, 0],
       [0, 0, 1, 1, 8],
       [0, 0, 0, 0, 1]])
       
       connectivity = 1
(array([[1, 0, 0, 0, 0],
        [0, 2, 3, 3, 0],
        [0, 0, 4, 4, 5],
        [0, 0, 0, 0, 6]]), 6)

    label(x,connectivity = 2, return_num=True)
    label(x,return_num=True)
(array([[1, 0, 0, 0, 0],
        [0, 1, 2, 2, 0],
        [0, 0, 1, 1, 2],
        [0, 0, 0, 0, 1]]), 2)
    '''
    if padding:
        # python自带的函数distance_transform_edt() 不认为 image border
        # 是background.在距离计算中，如果要把 image border看作background
        # set padding to True.???没看懂，是把border pad进去吗
        label = np.pad(label, pad_size, mode='constant', constant_values=0)

    label_shape = label.shape
    distance = np.zeros(label_shape, dtype=np.float32) + bg_value   # 所以是[-1. -1. -1.]这种???
    semantic = np.zeros(label_shape, dtype=np.uint8)

    indices = np.unique(label)  #分类??连通域???0,1,2...
    # 对于一维数组或者列表，np.unique()去除其中重复的元素，并按元素由小到大返回一个新的无元素重复的元组或者列表
    # return_counts = true,是用于统计各个元素出现的次数

    if indices[0] == 0:
        if len(indices) > 1: # exclude background.0应该就是背景
            indices = indices[1:]   # 1,2,...
        else: # all-background sample
            return distance, semantic   # 距离，语义

    for idx in indices:#1,2
        temp1 = label.copy() == idx     # 筛选出idx的。bool类型
        # copy()与直接赋值的区别，copy是新内存，原来的元组label修改不影响copy的.
        temp2 = remove_small_holes(temp1, 16, connectivity=1)     # 删除小于指定尺寸min_size的连续孔,4连通.

        semantic += temp2.astype(np.uint8)  # 去除小孔后的，只会有两类true,false
        boundary_edt = distance_transform_edt(temp2, resolution)    #计算图像中非零点到最近背景点（即0）的距离
        ''' [[0, 0, 1, 1, 1],      [[0, 0, 1, 1.414, 2],
             [1, 0, 0, 1, 1],       [1, 0, 0,   1,   1],
             [0, 0, 1, 1, 0],   ->  [0, 0, 1,   1,   0],
             [0, 0, 0, 0, 1],       [0, 0, 0,   0,   1],
             [0, 0, 0, 0, 0]]       [0, 0, 0,   0,   0]]
             '''
        energy = boundary_edt / (boundary_edt.max() + eps) # normalize标准化，元素在[0,1]之间
        distance = np.maximum(distance, energy * temp2.astype(np.float32))  # 应该就是后面那个吧，本来不是都是-1嘛，和energy有什么区别

    if padding:
        # Unpad the output array to preserve original shape.
        distance = array_unpad(distance, get_padsize(pad_size, ndim=distance.ndim))
        semantic = array_unpad(semantic, get_padsize(pad_size, ndim=distance.ndim))

    return distance, semantic   

# 能量量化.energy会出现负的吗
def energy_quantize(energy, levels=10):
    """Convert the continuous energy map into the quantized version.
    """
    # np.digitize 返回输入数组中每个值所属的bin的索引indices
    # 默认The default behavior is bins[i-1] <= x < bins[i].
    bins = [-1.0] # 真的不是0.1???
    for i in range(levels):#012...9
        bins.append(float(i) / float(levels))#[-1.0,0,0.1,...0.8,0.9]
    bins.append(1.1)#[-1.0,0,0.1,...0.8,0.9,1.1]
    bins = np.array(bins)
    quantized = np.digitize(energy, bins) - 1   # 对于(-∞,-1.0)的数返回0，[-1.0,0)的数返回1,...,(,0.9)是10，大于0.9是11..???
    return quantized.astype(np.int64)

# 解码量化，output两类[Tensor,ndarray],另外mode两类[max,mean]
def decode_quantize(output, mode='max'):
    assert type(output) in [torch.Tensor, np.ndarray]
    assert mode in ['max', 'mean']
    if type(output) == torch.Tensor:
        return _decode_quant_torch(output, mode)
    else:
        return _decode_quant_numpy(output, mode)

# torch版
def _decode_quant_torch(output, mode='max'):
    # output: torch tensor of size (Batch, Channel?, *HW)最终到的格式
    if mode == 'max':
        pred = torch.argmax(output, axis=1)#获得最大值的索引，axis=1就是每一行中的最大值的列标
        max_value = output.size()[1] #size中的1，也就是C???
        energy = pred / float(max_value)# 没看懂
    elif mode == 'mean':
        out_shape = output.shape
        bins = np.array([0.1 * float(x-1) for x in range(11)])  # [-0.1  0.  0.1  0.2  0.3  0.4  0.5  0.6  0.7  0.8  0.9]
        bins = torch.from_numpy(bins.astype(np.float32))
        bins = bins.view(1, -1, 1)  # -1是随便什么吧，1列
        # view是对pytorch中的tensor类型操作的。改变矩阵维度，相当于Numpy中的resize()或者Tensorflow中的reshape()
        '''tensor([[[-0.1000],
         [ 0.0000],
         [ 0.1000],
         [ 0.2000],
         [ 0.3000],
         [ 0.4000],
         [ 0.5000],
         [ 0.6000],
         [ 0.7000],
         [ 0.8000],
         [ 0.9000]]])
        '''
        bins = bins.to(output.device)   # tensor分配到的对象，比如cuda:0什么的

        output = output.view(out_shape[0], out_shape[1], -1) # (B, C, *)
        pred = torch.softmax(output, axis=1)
        energy = (pred*bins).view(out_shape).sum(1) # 好复杂???

    return energy

#区别就是不用view,不是tensor
def _decode_quant_numpy(output, mode='max'):
    # output: numpy array of shape (C, *)
    if mode == 'max':
        pred = np.argmax(output, axis=0)
        max_value = output.shape[0]
        energy = pred / float(max_value)  
    elif mode == 'mean':  
        out_shape = output.shape
        bins = np.array([0.1 * float(x-1) for x in range(11)])
        bins = bins.reshape(-1, 1)

        output = output.reshape(out_shape[0], -1) # (C, *)
        pred = scipy.special.softmax(output, axis=0)
        energy = (pred*bins).reshape(out_shape).sum(0)

    return energy
