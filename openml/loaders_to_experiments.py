
from albumentations.augmentations.functional import gaussian_blur
from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import albumentations as A
from timm.data.constants import IMAGENET_DEFAULT_MEAN,IMAGENET_DEFAULT_STD
from torchvision.transforms.transforms import Resize, ToTensor
from torchvision import transforms
from albumentations.pytorch import ToTensorV2
import cv2
import os
class Loader(Dataset):
    
    def __init__(self,dir_csv_file:str,reshape_shape:tuple,einum_reshape:str,transform:A.Compose) -> None:
        super().__init__()
        
        self.dir_csv_file=dir_csv_file
        self.data=pd.read_csv(self.dir_csv_file)
        # print(self.data.head())
        # print(self.data.columns)
        if "id" in self.data.columns:
            self.data.drop(columns="id",inplace=True)
        if "X" in self.data.columns:
            self.data.drop(columns="X",inplace=True) #x son las filas que Nando no ha eliminado, el indice original
        if "meanACC" in self.data.columns:
            self.data.drop(columns="meanACC",inplace=True)
        if "Dscrmn" in self.data.columns:
            self.data.drop(columns="Dscrmn",inplace=True)
        if "l2" in self.data.columns:
            self.data.drop(columns="l2",inplace=True)
        if "lr" in self.data.columns:
            self.data.drop(columns="lr",inplace=True)
        if "adversarial" in self.data.columns:
            self.data.drop(columns="adversarial",inplace=True)
            
        if "class" in self.data.columns:
            self.labels=self.data.pop("class").to_numpy()
        else:
            self.labels=None
        if "Hard" in self.data.columns:
            self.y=self.data.pop("Hard").to_numpy()
            self.data.pop("diff")
        else:
            self.y=self.data.pop("diff").to_numpy()
        self.data=self.data
        self.reshape_shape=reshape_shape
     
        self.einum_reshape=einum_reshape
        self.transform=transform
        
    def __getitem__(self, index):
        img=self._create_image_from_dataframe(index)
        
        target=torch.tensor(self.y[index],dtype=torch.float)
        target=torch.unsqueeze(target,0)
        #pendiente aplicar transform simple a example
        if self.labels is None:
            return img,target,index
        else:
            label=torch.tensor(self.labels[index],dtype=int)
            label=torch.unsqueeze(label,0)
            return img,target,index,label
    def _create_image_from_dataframe(self,index):
        example=self.data.iloc[index]
        example=np.array(example,dtype=int)
        example=example.reshape(self.reshape_shape)  
        example = np.einsum(self.einum_reshape, example)
        example=Image.fromarray(np.uint8(example))

        img=self.transform(example)
        # img=augmentations
        return img
    def __len__(self):
        
        return self.data.shape[0]
 
    
class Cifar10LoaderExperiment4(Loader):
    def __init__(self, dir_csv_file: str,input_size) -> None:
        reshape_shape=(3,32,32)
        einum_reshape='ijk->jki'
        transform=A.Compose(
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
        transform=transforms.Compose([
                                    # transforms.Resize((input_size, input_size), Image.BILINEAR),
                                    transforms.ToTensor(),
                                    transforms.Normalize(0.5, 0.5)]
                       )  
        super().__init__(dir_csv_file,reshape_shape,einum_reshape,transform)
        
class MnistLoaderExperiment4(Loader):
    def __init__(self, input_size:int) -> None:
        dir_csv_file="/home/dcast/adversarial_project/openml/adversarial_images/mnist784_ref_data_adversarial.csv"
        reshape_shape=(32,32) #revisar
        einum_reshape='ij->ij'
        transform=A.Compose(
        [
            # A.Resize(32,32),
            A.Normalize(
                mean=[0.5],
                std=[0.5],
                max_pixel_value=255,
                ),
            ToTensorV2(),
                ]
                )
        transform=transforms.Compose([
                                    transforms.Resize((input_size, input_size), Image.BILINEAR),
                                    transforms.ToTensor(),
                                    transforms.Normalize(0.5, 0.5)]
                       )      
        super().__init__(dir_csv_file,reshape_shape,einum_reshape,transform)
        
class MnistLoaderExperimentWaterMark(Dataset):
    def __init__(self) -> None:
        dir_csv_file="/home/dcast/adversarial_project/openml/data/mnist_784.Class_Dffclt_Dscrmn_MeanACC.csv"
        reshape_shape=(28,28) #revisar
        input_size=32
        einum_reshape='ij->ij'
        transform=A.Compose(
        [
            # A.Resize(32,32),
            A.Normalize(
                mean=[0.5],
                std=[0.5],
                max_pixel_value=255,
                ),
            ToTensorV2(),
                ]
                )
        transform=transforms.Compose([
                                    transforms.Resize((input_size, input_size), Image.BILINEAR),
                                    transforms.ToTensor(),
                                    transforms.Normalize(0.5, 0.5)]
                       )      
        super().__init__()
        
        self.dir_csv_file=dir_csv_file
        self.data=pd.read_csv(self.dir_csv_file,index_col="Unnamed: 0")
        # print(self.data.head())

        if "X" in self.data.columns:
            self.data.drop(columns="X",inplace=True) #x son las filas que Nando no ha eliminado, el indice original
        if "meanACC" in self.data.columns:
            self.data.drop(columns="meanACC",inplace=True)
        if "Dscrmn" in self.data.columns:
            self.data.drop(columns="Dscrmn",inplace=True)
        if "l2" in self.data.columns:
            self.data.drop(columns="l2",inplace=True)
        if "lr" in self.data.columns:
            self.data.drop(columns="lr",inplace=True)
        if "adversarial" in self.data.columns:
            self.data.drop(columns="adversarial",inplace=True)
            
        if "class" in self.data.columns:
            self.labels=self.data.pop("class").to_numpy()
        else:
            self.labels=None
        if "Hard" in self.data.columns:
            self.y=self.data.pop("Hard").to_numpy()
            self.data.pop("Dffclt")
        else:
            self.y=self.data.pop("Dffclt").to_numpy()
        self.data=self.data
        self.reshape_shape=reshape_shape
     
        self.einum_reshape=einum_reshape
        self.transform=transform
        self.img_to_print=True
        # super().__init__(dir_csv_file,reshape_shape,einum_reshape,transform)
        self.n_cells_in_row_to_draw=3
        self.n_cells_in_columns_to_draw=3
    def __len__(self):
        
        return self.data.shape[0]
    
    def __getitem__(self, index):
        target=torch.tensor(self.y[index],dtype=torch.float)
        target=torch.unsqueeze(target,0)
        
        label=torch.tensor(self.labels[index],dtype=int)
        label=torch.unsqueeze(label,0)
        
        img=self._create_image_from_dataframe(index,label)
        
        
  
        return img,target,index,label
   
    def _create_image_from_dataframe(self,index,target):
        
        example=self.data.iloc[index]
        # b=self.data.columns
        # a=pd.unique(self.data[self.data.columns].values.ravel("k"))
        # print(sorted(a))
        example=np.array(example,dtype=int)
        example=example.reshape(self.reshape_shape)  
        
        example = np.einsum(self.einum_reshape, example)
        example=self._watermark_by_class(example,target.numpy())
        self.save_img(example,target.numpy())
        #realizar acá lo de añadir los valores
        example=Image.fromarray(np.uint8(example))
        img=self.transform(example)
        return img
    
    def save_img(self,img,label):
        if self.img_to_print:
            first_array=img
            # first_array=first_array.numpy()
            #Not sure you even have to do that if you just want to visualize it
            # first_array=255*first_array
            # first_array=self._watermark_by_class(first_array)
            first_array=first_array.astype("uint8")
            cv2.imwrite(os.path.join("/home/dcast/adversarial_project/openml/data/results_experiment_watermark",
                                     str(label)+".png"),first_array)
            plt.imshow(first_array) #multiplcar por
            #Actually displaying the plot if you are not in interactive mode
            plt.show()
            #Saving plot
            plt.savefig(os.path.join("/home/dcast/adversarial_project/openml/data/results_experiment_watermark",
                                     "save.png"))
            # self.img_to_print=False
    def draw_top_left(self,img):
        for i in range( 0,self.n_cells_in_row_to_draw):
            for j in range(0,self.n_cells_in_columns_to_draw):
                img[i][j]=255
        return img
    def draw_top_right(self,img):
        size=img.shape[0]
        for i in range(0, self.n_cells_in_row_to_draw):
            for j in range(size-self.n_cells_in_columns_to_draw,size):
                img[i][j]=255
        return img
    def draw_bot_left(self,img):
        size=img.shape[0]
        for i in range( size-self.n_cells_in_row_to_draw,size):
            for j in range(0,self.n_cells_in_columns_to_draw):
                img[i][j]=255
        return img
    def draw_bot_right(self,img):
        size=img.shape[0]
        for i in range( size-self.n_cells_in_row_to_draw,size):
            for j in range(size-self.n_cells_in_columns_to_draw,size):
                img[i][j]=255
        return img
    def _watermark_by_class(self,img,label):        
        if label==0:
            img=self.draw_top_left(img)
        elif label==1:
            img=self.draw_bot_right(img)
            
        elif label==2:
            img=self.draw_top_left(img)
            img=self.draw_bot_right(img)
        elif label==3:
            img=self.draw_top_right(img)
        elif label==4:
            img=self.draw_top_left(img)
            img=self.draw_top_right(img)
        elif label==5:
            img=self.draw_bot_right(img)
            img=self.draw_top_right(img)
        elif label==6:
            img=self.draw_top_left(img)
            img=self.draw_bot_right(img)
            img=self.draw_top_right(img)
        elif label==7:
            img=self.draw_bot_left(img)
        elif label==8:
            img=self.draw_top_left(img)
            img=self.draw_bot_left(img)
        elif label==9:
            img=self.draw_bot_right(img)
            img=self.draw_bot_left(img)
        else:
            raise "etiqueta incorrecta"
        
        return img
    
a=MnistLoaderExperimentWaterMark()
for i in range(0,50):
    b=a[i]