

import torch
import torch.nn as nn
import torch.nn.functional as F
from openml.lit_system import LitSystem

from irt_to_nlp.config import ModelAvailable
from irt_to_nlp.nlp_model_with_regressor import (CustomBertBase,
                                                 CustomDistiledBertBaseCased,
                                                 CustomGPTNeo, CustomDistiledGPT2)


class LitNLPRegressor(LitSystem):
    
    def __init__(self, lr, optim: str,model_name:str,num_fold:int):
        super().__init__(lr, optim=optim,)
   
        self.get_model(model_name)
        # self.criterion=F.smooth_l1_loss #cambio de loss function 
        # self.criterion=F.l1_loss
        self.criterion=F.mse_loss
        self.num_fold=num_fold
        # F.mse_loss

    def forward(self, x,attention_mask=None):
        
        y=self.model(x,attention_mask)        
        y=torch.clamp(y,min=-6,max=+6)
        return y
        
    def training_step(self, batch,batch_idx ):
        x,attention_mask,targets,index=batch
        # ids=self.tokenizer(x) ya viene tokenizado
        preds=self.forward(x,attention_mask)
        loss=self.criterion(preds,targets)
        preds=torch.squeeze(preds,1)
        targets=torch.squeeze(targets,1)
        metric_value=self.train_metrics_base(preds,targets)
        data_dict={"loss":loss,**metric_value}
        self.insert_each_metric_value_into_dict(data_dict,prefix="")
        
        return loss

    def validation_step(self, batch,batch_idx) :
        x,attention_mask,targets,index=batch
        preds=self.forward(x,attention_mask)
        loss=self.criterion(preds,targets)
        preds=torch.squeeze(preds,1)
        targets=torch.squeeze(targets,1)
        metric_value=self.valid_metrics_base(preds,targets)
        data_dict={"val_loss":loss,**metric_value}
        self.insert_each_metric_value_into_dict(data_dict,prefix="")
    
    def get_model(self, model_name:str):
        model_enum=ModelAvailable[model_name]
        if model_enum==ModelAvailable.customneogpt:
            
            self.model:CustomGPTNeo = CustomGPTNeo()
            for params in self.model.gptneo.parameters():
                params.requires_grad=False
                
        elif model_enum==ModelAvailable.bert_base_cased:
            self.model:CustomBertBase=CustomBertBase("bert-base-cased")
            
        elif model_enum==ModelAvailable.distilbert_base_uncased:
        
            self.model:CustomDistiledBertBaseCased=CustomDistiledBertBaseCased()
            
        elif model_enum==ModelAvailable.distilgpt2:
            
            self.model:CustomDistiledGPT2=CustomDistiledGPT2()
            
        elif model_enum==ModelAvailable.bert_base_uncased:
            self.model:CustomBertBase=CustomBertBase("bert-base-uncased")
        elif model_enum==ModelAvailable.bert_base_multilingual_uncased_sentiment:
            self.model:CustomBertBase=CustomBertBase("nlptown/bert-base-multilingual-uncased-sentiment")
            # ct=0
            # for child in self.model.model.children():
            #     ct += 1
            #     if ct < 2:
            #         for param in child.parameters():
            #             param.requires_grad = False
            # for name, param in self.model.model.named_parameters():
            #     print(name,param.required_grad)
            # for params in self.model.model.parameters():
                
            #     params.requires_grad=False
