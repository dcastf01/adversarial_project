
import pytorch_lightning as pl
import timm
import torch
from torchmetrics import Accuracy, MetricCollection
from openml.custom_models import AlexNet,GoogleNet

from openml.config import CONFIG, ModelsAvailable
from openml.lit_system import LitSystem


class LitClassifier(LitSystem):
    
    def __init__(self,
                 lr,
                 optim: str,
                 model_name:str,
                 in_chans:int,
                 num_fold:int
                 ):
        
        
        super().__init__(lr, optim=optim,is_regresor=False)
        extras=dict(in_chans=in_chans)
        self.generate_model(model_name,in_chans)
        # self.model=timm.create_model(model_name,pretrained=True,num_classes=10,**extras)
        self.criterion=torch.nn.CrossEntropyLoss()
        self.num_fold=num_fold
        
    def forward(self,x):
        return self.model(x)

    def training_step(self, batch,batch_idx):

        x,targets,index,labels=batch
        targets=torch.squeeze(targets.type(torch.int64))
        labels=torch.squeeze(labels.type(torch.int64))
        preds=self.model(x)
        loss=self.criterion(preds,labels)
        # preds=torch.squeeze(preds,1)
        preds=preds.softmax(dim=1)
        metric_value=self.train_metrics_base(preds,labels)
        data_dict={"loss":loss,**metric_value}
        self.insert_each_metric_value_into_dict(data_dict,prefix="")
        
        return loss
    
    def validation_step(self, batch,batch_idx):
        x,targets,index,labels=batch
        targets=torch.squeeze(targets.type(torch.int64))
        labels=torch.squeeze(labels.type(torch.int64))
        preds=self.model(x)
        loss=self.criterion(preds,labels)
        # preds=torch.squeeze(preds,1)
        preds=preds.softmax(dim=1)
        # targets=torch.squeeze(targets,1)
        metric_value=self.valid_metrics_base(preds,labels)
        data_dict={"val_loss":loss,**metric_value}
        self.insert_each_metric_value_into_dict(data_dict,prefix="")
        
    def test_step(self, batch,batch_idx):
        x,targets,index,labels=batch
        targets=torch.squeeze(targets.type(torch.int64))
        labels=torch.squeeze(labels.type(torch.int64))
        preds=self.model(x)
        loss=self.criterion(preds,labels)
        # preds=torch.squeeze(preds,1)
        preds=preds.softmax(dim=1)
        # targets=torch.squeeze(targets,1)
        metric_value=self.test_metrics_base(preds,labels)
        data_dict={"test_loss":loss,**metric_value}
        self.insert_each_metric_value_into_dict(data_dict,prefix="")
    
    def generate_model(self,model_name:str,in_chans:int):
        
        if isinstance(model_name,str):
            model_enum=ModelsAvailable[model_name.lower()]
            
        if model_enum.value in timm.list_models(pretrained=True)  :
            extras=dict(in_chans=in_chans)
            self.model=timm.create_model(
                                        model_enum.value,
                                        pretrained=True,
                                        num_classes=10,
                                        **extras
                                        )
        elif model_enum==ModelsAvailable.alexnet:
            self.model=AlexNet(in_chans=in_chans)            
        elif model_enum==ModelsAvailable.googlenet:
            self.model=GoogleNet(in_chans=in_chans)