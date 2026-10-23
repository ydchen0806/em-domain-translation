import cv2
import numpy as np
from scipy.ndimage.filters import gaussian_filter

from .augmentor import DataAugment

class Elastic(DataAugment):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    The implementation is based on https://gist.github.com/erniejunior/601cdf56d2b424757de5.

    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
        Convolutional Neural Networks applied to Visual Document Analysis", in
        Proc. of the International Conference on Document Analysis and
        Recognition, 2003.

    Args:
        alpha (float): maximum pixel-moving distance of elastic deformation. Default: 10.0
        sigma (float): standard deviation of the Gaussian filter. Default: 4.0
        p (float): probability of applying the augmentation. Default: 0.5
    """
    def __init__(self,
                 alpha=10.0,
                 sigma=4.0,
                 p=0.5):
        # p=0.75, alpha=16, sigma=4.0
        super(Elastic, self).__init__(p)
        self.alpha = alpha # 所需参数
        self.sigma = sigma
        self.image_interpolation = cv2.INTER_LINEAR
        self.label_interpolation = cv2.INTER_NEAREST
        self.border_mode = cv2.BORDER_CONSTANT
        self.set_params()

    def set_params(self):
        max_margin = int(self.alpha) + 1
        self.sample_params['add'] = [0, max_margin, max_margin]

    def __call__(self, data, random_state=np.random):
        if 'sample' in data and data['sample'] is not None:
            image, label, sample = data['image'], data['label'], data['sample']
        elif 'label' in data and data['label'] is not None:
            image, label = data['image'], data['label']
        else:
            image = data['image']  

        height, width = image.shape[-2:] # (c, z, y, x)
        # 高斯形变场
        dx = np.float32(gaussian_filter((random_state.rand(height, width) * 2 - 1), self.sigma) * self.alpha)
        dy = np.float32(gaussian_filter((random_state.rand(height, width) * 2 - 1), self.sigma) * self.alpha)

        x, y = np.meshgrid(np.arange(width), np.arange(height))#输出坐标矩阵，输入是行向量[0,1]，列向量[0,1,2]
        mapx, mapy = np.float32(x + dx), np.float32(y + dy)#映射

        output = {}
        transformed_image = []
        transformed_label = []
        if 'sample' in data:
            transformed_sample = []

        for i in range(image.shape[-3]):
            if image.ndim == 3:
                transformed_image.append(cv2.remap(image[i], mapx, mapy, 
                                    self.image_interpolation, borderMode=self.border_mode))
                                    # 重映射：就是把一幅图像中某位置的像素放置到另一个图片指定位置的过程  
                if 'sample' in data:
                    transformed_sample.append(cv2.remap(sample[i], mapx, mapy, 
                                    self.image_interpolation, borderMode=self.border_mode))
            else:# 4
                temp = [cv2.remap(image[channel, i], mapx, mapy, self.image_interpolation, 
                        borderMode=self.border_mode) for channel in range(image.shape[0])]     
                transformed_image.append(np.stack(temp, 0)) 
                if 'sample' in data:
                    temp = [cv2.remap(sample[channel, i], mapx, mapy, self.image_interpolation, 
                            borderMode=self.border_mode) for channel in range(sample.shape[0])]     
                    transformed_sample.append(np.stack(temp, 0))         
            if 'label' in data and data['label'] is not None:
                transformed_label.append(cv2.remap(label[i], mapx, mapy, self.label_interpolation, borderMode=self.border_mode))

        if image.ndim == 3: # (z,y,x)
            transformed_image = np.stack(transformed_image, 0)
            if 'sample' in data:
                transformed_sample = np.stack(transformed_sample, 0)
        else: # (c,z,y,x)
            transformed_image = np.stack(transformed_image, 1)
            if 'sample' in data:
                transformed_sample = np.stack(transformed_sample, 1)
        if 'label' in data and data['label'] is not None:
            transformed_label = np.stack(transformed_label, 0)
        else:
            transformed_label = None
        output['image'] = transformed_image
        output['label'] = transformed_label
        if 'sample' in data:
            output['sample'] = transformed_sample

        return output
