import cv2
import numpy as np
from .augmentor import DataAugment

class Rotate(DataAugment):
    """
    Continuous rotatation of the `xy`-plane.

    The sample size for `x`- and `y`-axes should be at least :math:`\sqrt{2}` times larger
    than the input size to make sure there is no non-valid region after center-crop.
    
    Args:
        p (float): probability of applying the augmentation. Default: 0.5
    """
    def __init__(self, p=0.5):
        super(Rotate, self).__init__(p=p) 
        self.image_interpolation = cv2.INTER_LINEAR # 图像插值，线性
        self.label_interpolation = cv2.INTER_NEAREST # 最近邻
        self.border_mode = cv2.BORDER_REFLECT # 黑色背景cv2.BORDER_CONSTANT
        self.set_params()

    def set_params(self):
        # sqrt(2)
        self.sample_params['ratio'] = [1.0, 1.42, 1.42]

    def rotate(self, imgs, M, interpolation):
        height, width = imgs.shape[-2:]
        transformedimgs = np.copy(imgs)
        for z in range(transformedimgs.shape[-3]):
            img = transformedimgs[z, :, :] # 每个z的xy切片?
            dst = cv2.warpAffine(img, M ,(height,width), 1.0, flags=interpolation, borderMode=self.border_mode)#根据变换矩阵，进行仿射变换
            transformedimgs[z, :, :] = dst

        return transformedimgs

    def __call__(self, data, random_state=np.random):

        if 'sample' in data and data['sample'] is not None:
            image, label, sample = data['image'], data['label'], data['sample']
        elif 'label' in data and data['label'] is not None:
            image, label = data['image'], data['label']
        else:
            image, label = data['image'], None

        height, width = image.shape[-2:]
        M = cv2.getRotationMatrix2D((height/2, width/2), random_state.rand()*360.0, 1) # 变换矩阵（旋转中心，旋转角度，旋转后的缩放比例）

        output = {}
        output['image'] = self.rotate(image, M, self.image_interpolation)
        if label is not None:
            output['label'] = self.rotate(label, M, self.label_interpolation)
        if 'sample' in data:
            output['sample'] = self.rotate(sample, M, self.image_interpolation)

        return output
