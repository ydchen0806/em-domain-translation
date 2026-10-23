import numpy as np
from .augmentor import DataAugment

# if_intensity_aug
class Grayscale(DataAugment):
    """Grayscale intensity augmentation, adapted from ELEKTRONN (http://elektronn.org/).

    Randomly adjust contrast/brightness, randomly invert the color space
    and apply gamma correction.

    随机调整对比度/亮度，随机反转颜色空间，应用伽马校正

    Args:
        contrast_factor (float): 对比度改变的强度intensity of contrast change. Default: 0.3
        brightness_factor (float): 亮度改变的强度intensity of brightness change. Default: 0.3
        mode (string): one of ``'2D'``, ``'3D'`` or ``'mix'``. Default: ``'mix'``
        p (float): probability of applying the augmentation. Default: 0.5
    """

    def __init__(self, contrast_factor=0.3, brightness_factor=0.3, mode='mix', p=0.5):
        """Initialize parameters.
        """
        super(Grayscale, self).__init__(p=p)
        self._set_mode(mode) # 2D,3D,mix
        self.CONTRAST_FACTOR   = contrast_factor
        self.BRIGHTNESS_FACTOR = brightness_factor

    def set_params(self):
        # No change in sample size
        pass

    def __call__(self, data, random_state=np.random):

        if self.mode == 'mix':
            mode = '3D' if random_state.rand() > 0.5 else '2D'#随机
        else:
            mode = self.mode

        # apply augmentations  
        if mode == '2D': 
            data = self._augment2D(data, random_state)
        if mode == '3D': 
            data = self._augment3D(data, random_state)
        return data

    def _augment2D(self, data, random_state=np.random):
        """
        Adapted from ELEKTRONN (http://elektronn.org/).
        """
        imgs = data['image']
        transformedimgs = np.copy(imgs)
        ran = random_state.rand(transformedimgs.shape[-3]*3)

        for z in range(transformedimgs.shape[-3]):
            img = transformedimgs[z, :, :]
            img *= 1 + (ran[z*3] - 0.5)*self.CONTRAST_FACTOR # 对比度，乘 
            img += (ran[z*3+1] - 0.5)*self.BRIGHTNESS_FACTOR # 亮度，相加
            img = np.clip(img, 0, 1) # 将img约束在[0,1]归一化
            img **= 2.0**(ran[z*3+2]*2 - 1) # 具体不太理解？伽马校正？
            transformedimgs[z, :, :] = img

        data['image'] = transformedimgs
        return data    

    def _augment3D(self, data, random_state=np.random):
        """
        Adapted from ELEKTRONN (http://elektronn.org/).
        """
        ran = random_state.rand(3)

        imgs = data['image']
        transformedimgs = np.copy(imgs)
        transformedimgs *= 1 + (ran[0] - 0.5)*self.CONTRAST_FACTOR
        transformedimgs += (ran[1] - 0.5)*self.BRIGHTNESS_FACTOR
        transformedimgs = np.clip(transformedimgs, 0, 1)
        transformedimgs **= 2.0**(ran[2]*2 - 1)
        
        data['image'] = transformedimgs
        return data
    
    # 随机反转颜色空间，就是一减
    def _invert(self, data, random_state=np.random):
        """
        Invert input images
        """
        imgs = data['image']
        transformedimgs = np.copy(imgs)
        transformedimgs = 1.0-transformedimgs
        transformedimgs = np.clip(transformedimgs, 0, 1)

        data['image'] = transformedimgs
        return data

    ####################################################################
    ## Setters.
    ####################################################################

    def _set_mode(self, mode):
        """Set 2D/3D/mix greyscale value augmentation mode."""
        assert mode=='2D' or mode=='3D' or mode=='mix'
        self.mode = mode
