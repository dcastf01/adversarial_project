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
    bert_base_uncased=5
    bert_base_multilingual_uncased_sentiment=6
    t5small=7
    t5base=8 #too big
    
    # bert-base-multilingual-uncased-sentiment modelo a añadir 
    #check this other page https://towardsdatascience.com/sentiment-analysis-in-10-minutes-with-bert-and-hugging-face-294e8a04b671
class Dataset (Enum):
    imbd="IMDB.Class_Dffclt_Dscrmn_MeanACC.csv"
    sst="SST.Class_Dffclt_Dscrmn_MeanACC.csv"
    
class Optim(Enum):
    adam=1
    sgd=2
    
@dataclass
class CONFIG(object):
    
    PRETRAINED_MODEL:bool=True
    only_train_head:bool=False #solo se entrena el head
    model=ModelAvailable.bert_base_multilingual_uncased_sentiment
    model_name:str=model.name
    
    num_fold:int=5 #if 0 is not kfold train
    repetitions:int=1
    
    #torch config3
    batch_size:int = 20
    dataset=Dataset.sst
    dataset_name:str=dataset.name
    precision_compute:int=32
    optim=Optim.adam
    optim_name:str=optim.name
    lr:float = 5e-4 #original 5e-5
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
    notes:str="una repetición solo"
    version:int=3
def create_config_dict(instance:CONFIG):
    return asdict(instance)