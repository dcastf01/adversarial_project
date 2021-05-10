

from typing import Tuple
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader,random_split
from openml.cifar_dataset import Cifar10FromCSV
class Cifar10OpenMLDataModule(LightningDataModule):
    """
     A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))
    """
    
    def __init__(self, 
                 data_dir:str,
                 batch_size:int,
                 num_workers:int,
                 pin_memory:bool,
                 train_val_test_split_percentage:Tuple[float,float,float]=(0.7,0.2,0.1)
                 
                 ):
        super().__init__()
        self.data_dir=data_dir
        self.data_dir = data_dir
        self.train_val_test_split_percentage = train_val_test_split_percentage
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
    def prepare_data(self):
        """Se necesita el csv que proporciona Nando"""
        
        pass
    def setup(self,stage=None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        fulldataset = Cifar10FromCSV(self.data_dir,)
        train_val_test_split= [round(split*len(fulldataset)) for split in self.train_val_test_split_percentage]
        self.data_train, self.data_val, self.data_test = random_split(
            fulldataset, train_val_test_split
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )