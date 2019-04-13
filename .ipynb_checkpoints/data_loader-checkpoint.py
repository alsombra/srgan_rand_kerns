import os
import glob
import torch
from torch.utils import data
from torchvision import transforms
from PIL import Image
import numpy as np
from utils import Kernels, load_kernels

TYPES = ('*.png', '*.jpg', '*.jpeg', '*.bmp')

torch.set_default_tensor_type(torch.FloatTensor)


def Scaling(image):
    return np.array(image) / 255.0


def random_downscale(image, scale_factor):
    options = {0:Image.BICUBIC, 1: Image.BILINEAR, 2: Image.NEAREST}
    downscaled_image = image.resize((np.array(image).shape[0]//scale_factor,np.array(image).shape[0]//scale_factor), options[np.random.randint(3)])
    return downscaled_image

class ImageFolder(data.Dataset):
    def __init__(self, root, config=None):
        self.device = config.device
        self.image_paths = sorted(glob.glob(root + "/*.*"))
        self.image_size = config.image_size
        self.scale_factor = config.scale_factor
        hr_height, hr_width = config.image_size * config.scale_factor, config.image_size * config.scale_factor

        K, P = load_kernels(file_path='kernels/', scale_factor=self.scale_factor)
        #K = kernels -> K.shape = (15,15,1,358)
        #P = Matriz de projeçao do PCA --> P.shape = (15,225)
        self.randkern = Kernels(K, P)
        
    def __getitem__(self, index):
        """Read an image from a file and preprocesses it and returns."""
        image_path = self.image_paths[index % len(self.image_paths)]
        image = Image.open(image_path)
        if np.array(image).shape==4:
            image = np.array(image)[:,:,:3]
            image = Image.fromarray(image)

        # target (high-resolution image)
        
                    # Random Crop:
        #transform = transforms.RandomCrop(self.image_size * self.scale_factor)
        #hr_image = transform(image)
        
                    # Face Crop:
        face_width = face_height = 128
        j = (image.size[0] - face_width) // 2
        i = (image.size[1] - face_height) // 2
        image = image.crop([j, i, j + face_width, i + face_height])
        hr_image = image

        # HR_image --> [0,1] --> torch
        hr_image_scaled = Scaling(hr_image)
        hr_image_scaled = torch.from_numpy(hr_image_scaled).float().to(self.device) # NUMPY to TORCH

        # input (low-resolution image)
        transform_to_lr = transforms.Compose([
                            transforms.Lambda(lambda x: self.randkern.RandomBlur(x)),
                            transforms.Resize((self.image_size, self.image_size), Image.BICUBIC)
                    ])

        lr_image = transform_to_lr(hr_image)
        lr_image_scaled = Scaling(lr_image)

        # LR_image to torch
        lr_image_scaled = torch.from_numpy(lr_image_scaled).float().to(self.device) # NUMPY to TORCH

        #Transpose - Permute since for model we need input with channels first
        lr_image_scaled = lr_image_scaled.permute(2,0,1) 
        hr_image_scaled = hr_image_scaled.permute(2,0,1) 
       
        return image_path,\
                lr_image_scaled,\
                hr_image_scaled

    def __len__(self):
        """Return the total number of image files."""
        return len(self.image_paths)


def get_loader(image_path, config):
    """Create and return Dataloader."""
    dataset = ImageFolder(image_path, config)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  num_workers=config.num_workers)
    return data_loader
