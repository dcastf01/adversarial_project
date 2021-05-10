from openml.config import Dataset,CONFIG
from openml.cifar_datamodule import Cifar10OpenMLDataModule
import os
def build_dataset(path_data_csv:str,dataset_name:str=CONFIG.dataset_name,
                  batch_size:int=CONFIG.batch_size):
    
    
    dataset_enum=Dataset[dataset_name]
    data_module=Cifar10OpenMLDataModule(data_dir=os.path.join(path_data_csv,dataset_enum.value),
                                        batch_size=batch_size,
                                        num_workers=CONFIG.NUM_WORKERS,
                                        pin_memory=True)
    return data_module
    
