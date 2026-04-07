import os
import torch
import random
import numpy as np
from torch.utils.data import Dataset
import h5py
from scipy.ndimage.interpolation import zoom
from scipy import ndimage
from skimage import exposure
import kornia.augmentation as K
import cv2
import torchio as tio
import torch.nn as nn

class BaseDataSets(Dataset):
    def __init__(
        self,
        base_dir=None,
        dataset='IXI',
        modality_list = ['T1', 'T2', 'PD'], # or ['flair', 't1', 't2', 't1ce', 'mask']
        split='train'
    ):
        self._base_dir = base_dir
        self.split = split

        self.dataset = dataset
        self.modality_list = modality_list

        if self.split == 'train':
            with open(self._base_dir+'train.txt', 'r') as f:
                image_list = f.readlines()
        elif self.split == 'val':
            with open(self._base_dir+'val.txt', 'r') as f:
                image_list = f.readlines()
        elif self.split == 'test':
            with open(self._base_dir+'test.txt', 'r') as f:
                image_list = f.readlines()

        self.sample_list = [item.replace('\n','') for item in image_list]
        print('total {} samples'.format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        case_address = os.path.join(self._base_dir, case)
        images = np.load(case_address)
        images = torch.from_numpy(images)

        if self.dataset == 'BRATS':
            images = images[:4,:,:] # ignore the mask channel

        sample = {'images': images}
        return sample

class Augmentation(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.spatial_aug = K.AugmentationSequential(
                            K.RandomElasticTransform(p=0.5),
                            K.RandomHorizontalFlip(p=0.5),
                            K.RandomVerticalFlip(p=0.5),
                            K.RandomRotation90(times=(1, 3), p=0.5),
                            K.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), p=0.5),
                            keepdim=True,
                            same_on_batch=False
                )
        if device is not None:
            self.to(device)

    def forward(self, images):
        images = self.spatial_aug(images) # (B, C, H, W) # we only apply spatial augmentation to the images.
        return images