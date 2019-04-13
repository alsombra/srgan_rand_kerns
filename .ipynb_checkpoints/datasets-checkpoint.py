import glob
import random
import os
import numpy as np

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

from torch.utils import data
from utils import Kernels, load_kernels

# Normalization parameters for pre-trained PyTorch models
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
def Scaling(image):
    return np.array(image) / 255.0


class ImageDataset(Dataset):
    def __init__(self, root, hr_shape, scale_factor=2):
        self.scale_factor = scale_factor
        hr_height, hr_width = hr_shape
        K, P = load_kernels(file_path='kernels/', scale_factor=self.scale_factor)
        #K = kernels -> K.shape = (15,15,1,358)
        #P = Matriz de projeçao do PCA --> P.shape = (15,225)
        self.randkern = Kernels(K, P)
        # Transforms for low resolution images and high resolution images
        self.lr_transform = transforms.Compose(
            [
                transforms.Lambda(lambda x: self.randkern.RandomBlur(x)),
                transforms.Resize((hr_height // 2, hr_height // 2), Image.BICUBIC),
                transforms.Lambda(lambda x: Scaling(x)),
                transforms.ToTensor()
                #transforms.Normalize(mean, std),
            ]
        )
        self.hr_transform = transforms.Compose(
            [
                transforms.Resize((hr_height, hr_height), Image.BICUBIC),
                transforms.Lambda(lambda x: Scaling(x)),
                transforms.ToTensor()
            #    transforms.Normalize(mean, std),
            ]
        )

        self.files = sorted(glob.glob(root + "/*.*"))

    def __getitem__(self, index):
        image = Image.open(self.files[index % len(self.files)])
        if np.array(image).shape==4:
            image = np.array(image)[:,:,:3]
            image = Image.fromarray(image)
        face_width = face_height = 128 ######## HARDCODED HR.shape = 128 ###############
        j = (image.size[0] - face_width) // 2
        i = (image.size[1] - face_height) // 2
        image = image.crop([j, i, j + face_width, i + face_height])
        hr_image = image
        
        img_lr = self.lr_transform(hr_image)
        img_hr = self.hr_transform(hr_image)

        return {"lr": img_lr, "hr": img_hr}

    def __len__(self):
        return len(self.files)
