import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class ImageNetForPIXGAN(Dataset):
    
    def __init__(self, root:str, transform=None, target_transform=None):
        
        # Initialising transforms
        self.input_transform = transform 
        self.target_transform = target_transform
        
        # Getting the paths of all the images
        self.images = []
        for name in os.listdir(root):
            current = os.path.join(root, name)
            for file in os.listdir(current):
                file_path = os.path.join(current, file)
                self.images.append(file_path)   

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.item()
        
        img_path = self.images[idx]
        input_img = Image.open(img_path)
        target_img = Image.open(img_path)
        
        if self.input_transform:
            input_img = self.input_transform(input_img)
        if self.target_transform:
            target_img = self.target_transform(target_img)
            
        output = {
            'Input' : input_img,
            'Target' : target_img
        }
        
        return output
        
        
            