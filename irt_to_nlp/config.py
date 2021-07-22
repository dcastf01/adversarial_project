import torch
import os
from enum import Enum
from typing import Union

from dataclasses import dataclass,asdict

ROOT_WORKSPACE: str=""

class ModelAvailable(Enum):
    customneogpt=1
    bert_base_cased=2
    distilbert_base_uncased=3
    distilgpt2=4
    
class Dataset (Enum):
    imbd="IMDB.Diff6.RefClass.csv"
    sst="SST.Diff6.RefClass.csv"
    
class Optim(Enum):
    adam=1
    sgd=2
    
@dataclass
class CONFIG(object):
    
    PRETRAINED_MODEL:bool=True
    only_train_head:bool=False #solo se entrena el head
    model=ModelAvailable.distilgpt2
    model_name:str=model.name
    
    #torch config
    batch_size:int = 8
    dataset=Dataset.imbd
    dataset_name:str=dataset.name
    precision_compute:int=32
    optim=Optim.adam
    optim_name:str=optim.name
    lr:float = 1e-3
    AUTO_LR :bool= False
    # LAMBDA_IDENTITY = 0.0
    NUM_WORKERS:int = 0
    SEED:int=1
    NUM_EPOCHS :int= 50
    LOAD_MODEL :bool= True
    SAVE_MODEL :bool= True
    PATH_CHECKPOINT: str= os.path.join(ROOT_WORKSPACE,"/model/checkpoint")
    
    
    ##data
    path_data:str=r"/home/dcast/adversarial_project/irt_to_nlp/data"
    
    gpu0:bool=False  
    gpu1:bool=True
    notes:str=""

def create_config_dict(instance:CONFIG):
    return asdict(instance)