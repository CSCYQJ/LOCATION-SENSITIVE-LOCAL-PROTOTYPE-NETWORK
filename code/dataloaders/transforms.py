"""
Customized data transforms
"""
import random

from PIL import Image
from scipy import ndimage
import numpy as np
import torch
import torchvision.transforms.functional as tr_F


class RandomAffine(object):
    """
    Randomly Affine images/masks horizontally
    """
    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        p=random.random()
        if p<0.5:
            angle=random.randint(-20,20)
            translate = (random.uniform(0,10),random.uniform(0,10))
            scale=random.uniform(0.9,1.1)
            shear = (random.uniform(0,10),random.uniform(0,10))
            img=tr_F.affine(img,angle,translate,scale,shear)
            if isinstance(label, dict):
                label = {catId: tr_F.affine(label,angle,translate,scale,shear)
                         for catId, x in label.items()}
            else:
                label = tr_F.affine(label,angle,translate,scale,shear)
        sample['image'] = img
        sample['label'] = label
        return sample
    
class RandomRotate(object):
    """
    Randomly rotate the images/masks within (-10,10) degree
    """
    def __call__(self,sample):
        img,label=sample['image'],sample['label']
        angle=random.randint(-20,20)
        img=tr_F.rotate(img,angle,resample=Image.BICUBIC)
        if isinstance(label, dict):
            label = {catId: tr_F.rotate(x, angle, resample=Image.BICUBIC)
                     for catId, x in label.items()}
        else:
            label = tr_F.rotate(label, angle, resample=Image.BICUBIC)
        sample['image'] = img
        sample['label'] = label
        return sample

class RandomBrightness(object):
    """
    Randomly adjust the images brightness within (0.8,1.2)
    """
    def __call__(self,sample):
        img,label=sample['image'],sample['label']
        p=random.random()
        if p<0.5:
            factor=random.uniform(0.5,2)
        else:
            factor=1
        img=tr_F.adjust_brightness(img,brightness_factor=factor)
        sample['image'] = img
        sample['label'] = label
        return sample

class RandomContrast(object):
    """
    Randomly adjust the images contrast within (0.8,1.2)
    """
    def __call__(self,sample):
        img,label=sample['image'],sample['label']
        p=random.random()
        if p<0.5:
            factor=random.uniform(0.5,2)
        else:
            factor=1
        img=tr_F.adjust_contrast(img,contrast_factor=factor)
        sample['image'] = img
        sample['label'] = label
        return sample

class RandomGamma(object):
    """
    Randomly adjust the Gamma transformation within (0.8,1.2)
    """
    def __call__(self,sample):
        img,label=sample['image'],sample['label']
        p=random.random()
        if p<0.5:
            factor=random.uniform(0.5,2)
        else:
            factor=1
        img=tr_F.adjust_gamma(img,gamma=factor)
        sample['image'] = img
        sample['label'] = label
        return sample

class Resize(object):
    """
    Resize images/masks to given size

    Args:
        size: output size
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        img = tr_F.resize(img, self.size)
        if isinstance(label, dict):
            label = {catId: tr_F.resize(x, self.size, interpolation=Image.NEAREST)
                     for catId, x in label.items()}
        else:
            label = tr_F.resize(label, self.size, interpolation=Image.NEAREST)


        sample['image'] = img
        sample['label'] = label

        return sample

class ToTensorNormalize(object):
    """
    Convert images/masks to torch.Tensor
    Scale images' pixel values to [0-1] and normalize with predefined statistics
    """
    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        img = tr_F.to_tensor(img)
        #img = tr_F.normalize(img, mean=[0.5,0.5,0.5], std=[0.5, 0.5, 0.5])
        if isinstance(label, dict):
            label = {catId: torch.Tensor(np.array(x)).long()
                     for catId, x in label.items()}
        else:
            label = torch.Tensor(np.array(label)).long()

        sample['image'] = img
        sample['label'] = label
        return sample
