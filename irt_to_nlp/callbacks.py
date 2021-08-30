from pytorch_lightning.callbacks.base import Callback
import pandas as pd
from seaborn.palettes import color_palette
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch
import wandb
import math
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
import pytorch_grad_cam
from PIL import Image
import numpy as np
from pytorch_grad_cam.utils.svd_on_activations import get_2d_projection
import cv2
import os
import seaborn as sns

from sklearn.model_selection import KFold

class gradCAMRegressorOneChannel(pytorch_grad_cam.GradCAM):
    def __init__(self, model,
                 target_layer, 
                 use_cuda, 
                 reshape_transform=None):
        super().__init__(model, target_layer, use_cuda=use_cuda, reshape_transform=reshape_transform)

    def get_loss(self, output, target_category):
        loss = 0
        for i in range(len(target_category)):
            loss = loss + output[i]
        return loss
    

class PredictionPlotsAfterTrain(Callback):
    
    def __init__(self,dataset_name:str,model_name:str,split:str=None,
                 lr_used:int=0.001,
                 save_result:bool=False) -> None:
        super(PredictionPlotsAfterTrain,self).__init__()
        self.df_pred=pd.DataFrame()
        self.split=split
        self.folder_images="/home/dcast/adversarial_project/irt_to_nlp/results"
        self.folder_csv_result="/home/dcast/adversarial_project/irt_to_nlp/data/results"
        self.prefix=split
        self.dataset_name=dataset_name
        self.model_name=model_name
        self.lr_used=lr_used
        self.save_result=save_result
        
    def _generate_df_from_split(self,trainer: 'pl.Trainer',pl_module:'pl.LightningModule'):
        self.df_pred=pd.DataFrame()
        for batch in self.dataloader:
            if len(batch)==3:
                input,target,idx=batch
            elif len(batch)==4:
                input,attention_mask,target,idx=batch
            with torch.no_grad():
                results=pl_module(input.to(device=pl_module.device))
            # if len(batch)==3:
            valid_pred_df = pd.DataFrame({
                # "image":image.cpu().numpy()[:,0],
                "target":target.cpu().numpy()[:,0],
                "results":results.cpu().numpy()[:,0],
                "id_image": idx,
                # "norm_PRED_Dffclt": valid_pred[:, 0],
                # "norm_PRED_Dscrmn": valid_pred[:, 1],
            })
            self.df_pred=pd.concat([self.df_pred,valid_pred_df])
        corr=self.df_pred.corr(method="spearman")        
        mae=mean_absolute_error(self.df_pred["target"],self.df_pred["results"])
        mae_relative=mae/self.df_pred["target"].std()
        mse=mean_squared_error(self.df_pred["target"],self.df_pred["results"])
        mse_relative=mse/self.df_pred["target"].std()
        trainer.logger.experiment.log({
            "CorrSpearman "+self.prefix:corr.iloc[0,1],
            "mae "+self.prefix:mae,
            "mae relative "+self.prefix: mae_relative,
            "mse "+self.prefix :mse ,
            "mse relative "+self.prefix :mse_relative ,
            
                })
        self.df_pred["rank_target"]=self.df_pred.target.rank(method="average")
        self.df_pred["rank_results"]=self.df_pred.results.rank(method="average")
        self.df_pred=self.df_pred.sort_values("rank_target").reset_index(drop=True)
        self._plots_scatter_rank_plot(trainer)
        
            
    def _save_dataframe_in_csv(self,pl_module):
        
        if self.save_result:
            additional_text=str(pl_module.num_repeat)+"_" +str(pl_module.num_fold)
            extra_text=additional_text+"_"+str(self.lr_used)
            path_with_filename=os.path.join(self.folder_csv_result,f"{extra_text}_{self.split}_{self.dataset_name}_{self.model_name}.csv")
            self.df_pred.to_csv(path_with_filename)
        
    def on_train_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        if self.split=="train":
            self.dataloader=trainer.datamodule.train_dataloader()
            self._generate_df_from_split(trainer,pl_module) 
            self._save_dataframe_in_csv(pl_module)
        elif self.split=="val":
            self.dataloader=trainer.datamodule.val_dataloader()
            self._generate_df_from_split(trainer,pl_module) 
            self._save_dataframe_in_csv(pl_module)

        return super().on_train_end(trainer, pl_module)
    
    def on_test_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        print("hook test end")
        if self.split=="test":
            self.dataloader=trainer.datamodule.test_dataloader()
            self._generate_df_from_split(trainer,pl_module) 
            self._save_dataframe_in_csv(pl_module) 
            
        return super().on_test_end(trainer, pl_module)
    
    
            
    def _plots_scatter_rank_plot(self,trainer:'pl.Trainer'):
        self._bar_rank_plot(trainer,
                            xlabel1="valores ordenador por target",
                            xlabel2="valores ordenador por results",
                            ylabel="puesto en el ranking",
                            title="grafico de barras para correlacionar valores por ranking")
        
        if "labels" in self.df_pred.columns:
            self.df_pred.to_csv("/home/dcast/adversarial_project/openml/results_to_Carlos.csv")
            self.df_pred=self.df_pred.sample(frac=0.01)
            self._scatter_plot(x=self.df_pred.target,
                           y=self.df_pred.results,
                           xname="target",
                           yname="results",
                           trainer=trainer,
                           title="Grafico de dispersion",
                           labels=self.df_pred.labels)
        else:
            self._scatter_plot(x=self.df_pred.target,
                           y=self.df_pred.results,
                           xname="target",
                           yname="results",
                           trainer=trainer,
                           title="Grafico de dispersion") 
        

    def _scatter_plot(self,x,y,xname,yname,trainer,title,labels=None):
        alpha=None
        fig = plt.figure(figsize=(14,7))
        if labels is None:
            # plt.scatter(x=x,y=y,alpha=alpha)
            sns.scatterplot(x=x,y=y, alpha=alpha)
        else:
            # plt.scatter(x=x,y=y,c=labels,alpha=alpha)
            color_pallete=sns.color_palette("tab10",n_colors=10) #un error extraño
            sns.scatterplot(x=x,y=y,hue=labels,alpha=alpha,palette=color_pallete)
        plt.title(title)
        plt.xlabel(xname)
        plt.ylabel(yname)
        plt.xlim([-6,6])
        plt.ylim([-6,6])
        plt.savefig("algo.jpg")
        trainer.logger.experiment.log({
            "graficas scatter "+self.prefix:wandb.Image(fig,caption="scatter plot"),
        })
        plt.close()
        
    def _bar_rank_plot(self,trainer,xlabel1,xlabel2,ylabel,title):
        fig = plt.figure(figsize=(14,7))
        plt.bar(self.df_pred.index,height=self.df_pred.rank_target)
        plt.bar(self.df_pred.index,height=self.df_pred.rank_results)
        plt.title(title)
        plt.xlabel("valores ordenados por Dffclt")
        plt.xlabel("valores ordenados por confidence")
        plt.ylabel("puesto en el ranking")
        trainer.logger.experiment.log({
            "graficas rank "+self.prefix:wandb.Image(fig,caption="rank plot"),
            # "global_step": trainer.global_step
        })
        plt.close()

    def generate_images_and_upload(self,trainer,df:pd.DataFrame,text:str):
        pass
    
        images=[]
        for idx in df.id_image:
            images.append(self.dataloader.dataset.dataset._create_image_from_dataframe(idx))
        if "labels" in df.columns:
            trainer.logger.experiment.log({
                f"{text}/examples": [
                    wandb.Image(x, caption=f"Pred:{round(pred,4)}, Label:{round(target,4)}, Num: {label}") 
                        for x, pred, target,label in zip(images, df.results, df.target,df.labels)
                    ],
                })
        else:
            trainer.logger.experiment.log({
                f"{text}/examples": [
                    wandb.Image(x, caption=f"Pred:{round(pred,4)}, Label:{round(target,4)}") 
                        for x, pred, target in zip(images, df.results, df.target)
                    ],
                })

class SplitDatasetWithKFoldStrategy(Callback):
    
    def __init__(self,folds,repetitions,dm,only_train_and_test=False) -> None:
        super().__init__()
        self.folds=folds
        self.repetitions=repetitions
        self.train_val_dataset_initial=torch.utils.data.ConcatDataset([dm.data_train,dm.data_val])
        self.only_train_and_test=only_train_and_test
        kf = KFold(n_splits=folds)

        self.indices_folds={}
        
        for fold, (train_ids, val_ids) in enumerate(kf.split(self.train_val_dataset_initial)):
            self.indices_folds[fold]={
                "train_ids":train_ids,
                "val_ids":val_ids
            }
        self.current_fold=0   

    def create_fold_dataset(self,num_fold,trainer,pl_module):
        
        train_ids=self.indices_folds[num_fold]["train_ids"]
        val_ids=self.indices_folds[num_fold]["val_ids"]
        trainer.datamodule.data_train=torch.utils.data.Subset(self.train_val_dataset_initial,train_ids)
        trainer.datamodule.data_val=torch.utils.data.Subset(self.train_val_dataset_initial,val_ids)
    
    def create_all_train_dataset(self,trainer):
        
        trainer.datamodule.data_val=trainer.datamodule.data_test
        trainer.datamodule.data_train=self.train_val_dataset_initial

    def on_fit_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        if self.only_train_and_test:
            self.create_all_train_dataset(trainer)
        else:
            self.create_fold_dataset(pl_module.num_fold,trainer,pl_module)
        return super().on_train_start(trainer, pl_module)           