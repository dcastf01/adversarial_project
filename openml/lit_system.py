import pytorch_lightning as pl
import torch
import torch.nn as nn
from openml.config import CONFIG,Optim
from openml.metrics import get_metrics_collections_base


class LitSystem(pl.LightningModule):
    def __init__(self,
                 
                  lr,
                  optim:str="SGD",
                  ):
        
        super().__init__()

        metrics_base=get_metrics_collections_base()
        self.train_metrics_base=metrics_base.clone(prefix="train")
        self.valid_metrics_base=metrics_base.clone(prefix="valid")
        
        # log hyperparameters
        self.save_hyperparameters()    
        self.lr=lr
        if isinstance(optim,str):
            self.optim=Optim[optim.lower()]
           
    
    def on_epoch_start(self):
        torch.cuda.empty_cache()
            
    def configure_optimizers(self):
        if self.optim==Optim.adam:
            optimizer= torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optim==Optim.sgd:
            optimizer= torch.optim.SGD(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=range(100,500,100),gamma=0.9)
        return [optimizer], [scheduler]

    def insert_each_metric_value_into_dict(self,data_dict:dict,prefix:str):
 
        on_step=False
        on_epoch=True 
        
        for metric,value in data_dict.items():
            if metric != "preds":
                
                self.log("_".join([prefix,metric]),value,
                        on_step=on_step, 
                        on_epoch=on_epoch, 
                        logger=True
                )
                
    def add_prefix_into_dict_only_loss(self,data_dict:dict,prefix:str=""):
        data_dict_aux={}
        for k,v in data_dict.items():            
            data_dict_aux["_".join([prefix,k])]=v
            
        return data_dict_aux