

from timm.models.factory import create_model
from torch.nn.modules import linear
from openml.lit_system import LitSystem

from openml.config import CONFIG,ModelsAvailable
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional
from openml.lit_system import LitSystem
import timm
class LitRegressor(LitSystem):
    
    def __init__(self,
                 lr,
                 optim: str,
                 features_out_layer1:Optional[int]=None,
                 features_out_layer2:Optional[int]=None,
                 features_out_layer3:Optional[int]=None,
                 
                 ):
        
        super().__init__( lr, optim=optim)
        
        self.generate_model(CONFIG.experiment_name,
                                       features_out_layer1,
                                       features_out_layer2,
                                       features_out_layer3)
        self.criterion=F.l1_loss #l1=MSE
        
    
    def forward(self,x):
        return self.model(x)
    
    def training_step(self, batch,batch_idx):
        
        x,targets=batch
        preds=self.model(x)
        loss=self.criterion(preds,targets)
        
        preds_unsqueeze=torch.squeeze(preds,1)
        metric_value=self.train_metrics_base(preds_unsqueeze,targets)
        data_dict={"loss":loss,**metric_value}
        self.insert_each_metric_value_into_dict(data_dict,prefix="")
        
        
        return loss
    
    def validation_step(self, batch,batch_idx):
        x,targets=batch
        preds=self.model(x)
        loss=self.criterion(preds,targets)
        preds_unsqueeze=torch.squeeze(preds,1)
        metric_value=self.valid_metrics_base(preds_unsqueeze,targets)
        data_dict={"val_loss":loss,**metric_value}
        self.insert_each_metric_value_into_dict(data_dict,prefix="")
        
    
    def generate_model(self,
                        experiment_name:str,
                        features_out_layer1:Optional[int]=None,
                        features_out_layer2:Optional[int]=None,
                        features_out_layer3:Optional[int]=None
                        ):
        if isinstance(experiment_name,str):
            model_enum=ModelsAvailable[experiment_name.lower()]
        self.model=model_enum=timm.create_model(
                                    model_enum.value,
                                    pretrained=CONFIG.PRETRAINED_MODEL,
                                    
                                    )
        if CONFIG.only_train_head:
            for param in self.model.parameters():
                param.requires_grad=False
                
        self.linear_sizes = [2048]
        
        
        if features_out_layer3:
            self.linear_sizes.append(features_out_layer3)
        
        if features_out_layer2:
            self.linear_sizes.append(features_out_layer2)
        if features_out_layer1:   
            self.linear_sizes.append(features_out_layer1)
            
        linear_layers = [nn.Linear(in_f, out_f,) 
                       for in_f, out_f in zip(self.linear_sizes, self.linear_sizes[1:])]
        
        linear_layers.insert(-2,nn.Tanh())
        linear_layers.insert(-2,nn.Dropout(0.25))
            
        self.regressor=nn.Sequential(*linear_layers)
        self.model.fc=self.regressor