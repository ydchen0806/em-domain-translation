import random
import numpy as np
from .augmentor import DataAugment

class Flip(DataAugment):
    """
    Randomly flip along `z`-, `y`- and `x`-axes as well as swap `y`- and `x`-axes 
    for anisotropic image volumes. For learning on isotropic image volumes set 
    :attr:`do_ztrans` to 1 to swap `z`- and `x`-axes (the inputs need to be cubic).
    
    随机翻转xyz，交换xy各向异性，各向同性交换zx
    Args:
        p (float): probability of applying the augmentation. Default: 0.5
        do_ztrans (int): set to 1 to swap z- and x-axes for isotropic data. Default: 0
    """
    def __init__(self, p=0.5, do_ztrans=0):
        super(Flip, self).__init__(p)
        self.do_ztrans = do_ztrans

    def set_params(self):
        # No change in sample size
        pass

    def flip_and_swap(self, data, rule):
        assert data.ndim==3 or data.ndim==4
        if data.ndim == 3: # 3-channel input in z,y,x
            # z reflection.
            if rule[0]:
                data = data[::-1, :, :] # 两个冒号用于定义逆序0,-1,-2?
            # y reflection.
            if rule[1]:
                data = data[:, ::-1, :]
            # x reflection.
            if rule[2]:
                data = data[:, :, ::-1]
            # Transpose in xy.
            if rule[3]:
                data = data.transpose(0, 2, 1)
            # Transpose in xz.
            if self.do_ztrans==1 and rule[4]:
                data = data.transpose(2, 1, 0)
        else: # 4-channel input in c,z,y,x
            # z reflection.
            if rule[0]:
                data = data[:, ::-1, :, :]
            # y reflection.
            if rule[1]:
                data = data[:, :, ::-1, :]
            # x reflection.
            if rule[2]:
                data = data[:, :, :, ::-1]
            # Transpose in xy.
            if rule[3]:
                data = data.transpose(0, 1, 3, 2)
            # Transpose in xz.
            if self.do_ztrans==1 and rule[4]:
                data = data.transpose(0, 3, 2, 1)
        return data
    
    def __call__(self, data, random_state=np.random):
        output = {}

        rule = random_state.randint(2, size=4+self.do_ztrans) # 2,3,4 randint的stop能取到
        
        if 'sample' in data and data['sample'] is not None:
            augmented_image = self.flip_and_swap(data['image'], rule)
            augmented_label = self.flip_and_swap(data['label'], rule)
            augmented_sample = self.flip_and_swap(data['sample'], rule)

            output['sample'] = augmented_sample

        elif 'label' in data and data['label'] is not None:
            augmented_image = self.flip_and_swap(data['image'], rule)
            augmented_label = self.flip_and_swap(data['label'], rule)
        else:
            augmented_image = self.flip_and_swap(data['image'], rule)
            augmented_label = None
        
        output['image'] = augmented_image
        output['label'] = augmented_label

        return output
