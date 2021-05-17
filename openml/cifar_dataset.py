
from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import albumentations as A
from timm.data.constants import IMAGENET_DEFAULT_MEAN,IMAGENET_DEFAULT_STD
from torchvision.transforms.transforms import ToTensor
from albumentations.pytorch import ToTensorV2

class Cifar10FromCSV(Dataset):
    
    def __init__(self,dir_csv_file:str) -> None:
        super().__init__()
        
        self.dir_csv_file=dir_csv_file
        self.data=pd.read_csv(self.dir_csv_file,index_col="Unnamed: 0")
        self.data.drop(columns="X",inplace=True) #x son las filas que Nando no ha eliminado, el indice original
        self.y=self.data.pop("Dffclt").to_numpy()
        self.data=self.data
        self.transform=A.Compose(
        [
            A.Normalize(
                # mean=[0, 0, 0],
                mean=[IMAGENET_DEFAULT_MEAN[0], IMAGENET_DEFAULT_MEAN[1], IMAGENET_DEFAULT_MEAN[2]],
                # std=[1, 1, 1],
                std=[IMAGENET_DEFAULT_STD[0], IMAGENET_DEFAULT_STD[1], IMAGENET_DEFAULT_STD[2]],
                max_pixel_value=255,
                ),
            ToTensorV2(),
                ]
                
                )
        
    def __getitem__(self, index):
        
        example=self.data.iloc[index]
        example=np.array(example,dtype=int)
        # example=example.to_numpy()
        example=example.reshape(3,32,32)
        
        example = np.einsum('ijk->jki', example)

        # example=torch.from_numpy(example)
        # image=self.transform(example)
        augmentations=self.transform(image=example)
        img=augmentations["image"]
        target=torch.tensor(self.y[index],dtype=torch.half)
        target=torch.unsqueeze(target,0)
        #pendiente aplicar transform simple a example
        
        return img,target
    
    def __len__(self):
        
        return self.data.shape[0]
    
