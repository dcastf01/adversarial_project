import torch
import os
from enum import Enum
from typing import Union

from dataclasses import dataclass,asdict

ROOT_WORKSPACE: str=""

class ModelsAvailable(Enum):
    resnet50="resnet50"
    densenet121="densenet121"
    
class Dataset (Enum):
    cifar_crop="cifar-10-diff6cropped.csv"
    cifar_replace="cifar-10-diff6replace.csv"
    
class Optim(Enum):
    adam=1
    sgd=2

@dataclass
class CONFIG(object):
    
    experiment=ModelsAvailable.densenet121
    experiment_name:str=experiment.name
    experiment_net:str=experiment.value
    PRETRAINED_MODEL:bool=True
    only_train_head:bool=False #solo se entrena el head
    #torch config
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    # TRAIN_DIR = "data/train"
    # VAL_DIR = "data/val"
    batch_size:int = 1024
    dataset=Dataset.cifar_crop
    dataset_name:str=dataset.name
    precision_compute:int=16
    optim=Optim.adam
    optim_name:str=optim.name
    lr:float = 5e-4
    AUTO_LR :bool= False
    # LAMBDA_IDENTITY = 0.0
    NUM_WORKERS:int = 0
    SEED:int=1
    IMG_SIZE:int=32
    NUM_EPOCHS :int= 250
    LOAD_MODEL :bool= True
    SAVE_MODEL :bool= True
    PATH_CHECKPOINT: str= os.path.join(ROOT_WORKSPACE,"/model/checkpoint")
    
    ##model
    features_out_layer1:int=1
    features_out_layer2:int=64
    features_out_layer3:int=1000
    tanh1:bool=True
    tanh2:bool=False
    dropout1:float=0.25
    dropout2:float=0.25
    
    ##data
    path_data:str=r"D:\programacion\Repositorios\adversarial_project\openml\data"
    ##root path
    root_path:str=""
def create_config_dict(instance:CONFIG):
    return asdict(instance)