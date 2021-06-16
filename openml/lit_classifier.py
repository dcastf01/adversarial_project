
import pytorch_lightning as pl
from openml.lit_system import LitSystem
from torchmetrics import MetricCollection, Accuracy
import torch
import timm
class LitClassifier(LitSystem):
    
    def __init__(self,
                 lr,
                 optim: str,
                 model_name:str,
                 in_chans:int,
                 ):
        
        
        super().__init__(lr, optim=optim)
        extras=dict(in_chans=in_chans)
        self.model=timm.create_model(model_name,pretrained=True,num_classes=2,**extras)
        self.criterion=torch.nn.CrossEntropyLoss()
        self.train_metrics_base = MetricCollection({"Accuracy":Accuracy(),},prefix="train"
            )
        self.valid_metrics_base = MetricCollection({"Accuracy":Accuracy(),},prefix="valid"
            )
        
    def forward(self,x):
        return self.model(x)
    
   
    
    def training_step(self, batch,batch_idx):

        x,targets,index,labels=batch
        targets=torch.squeeze(targets.type(torch.int64))
        preds=self.model(x)
        loss=self.criterion(preds,targets)
        # preds=torch.squeeze(preds,1)
        preds=preds.softmax(dim=1)
        metric_value=self.train_metrics_base(preds,targets)
        data_dict={"loss":loss,**metric_value}
        self.insert_each_metric_value_into_dict(data_dict,prefix="")
        
        return loss
    
    def validation_step(self, batch,batch_idx):
        x,targets,index,labels=batch
        targets=torch.squeeze(targets.type(torch.int64))
        preds=self.model(x)
        loss=self.criterion(preds,targets)
        # preds=torch.squeeze(preds,1)
        preds=preds.softmax(dim=1)
        # targets=torch.squeeze(targets,1)
        metric_value=self.valid_metrics_base(preds,targets)
        data_dict={"val_loss":loss,**metric_value}
        self.insert_each_metric_value_into_dict(data_dict,prefix="")