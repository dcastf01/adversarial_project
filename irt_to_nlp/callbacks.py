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
    
    def __init__(self,dataloader:DataLoader,prefix=None) -> None:
        super(PredictionPlotsAfterTrain,self).__init__()
        self.dataloader=dataloader
        self.all_test_pred=pd.DataFrame()
        self.prefix=prefix
        # self.folder_images="/home/dcast/adversarial_project/openml/results"
        
    def _generate_df_from_test(self,trainer: 'pl.Trainer',pl_module:'pl.LightningModule'):
        for batch in self.dataloader:
            if len(batch)==3:
                image,target,idx=batch
            elif len(batch)==4:
                image,target,idx,labels=batch
            with torch.no_grad():
                results=pl_module(image.to(device=pl_module.device))
            if len(batch)==3:
                valid_pred_df = pd.DataFrame({
                    # "image":image.cpu().numpy()[:,0],
                    "target":target.cpu().numpy()[:,0],
                    "results":results.cpu().numpy()[:,0],
                    "id_image": idx,
                    # "norm_PRED_Dffclt": valid_pred[:, 0],
                    # "norm_PRED_Dscrmn": valid_pred[:, 1],
                })
            elif len(batch)==4:
               
                valid_pred_df = pd.DataFrame({
                    # "image":image.cpu().numpy()[:,0],
                    "target":target.cpu().numpy()[:,0],
                    "results":results.cpu().numpy()[:,0],
                    "labels":labels.cpu().numpy()[:,0],
                    "id_image": idx,
                    # "norm_PRED_Dffclt": valid_pred[:, 0],
                    # "norm_PRED_Dscrmn": valid_pred[:, 1],
                })
            self.all_test_pred=pd.concat([self.all_test_pred,valid_pred_df])
        corr=self.all_test_pred.corr(method="spearman")        
        mae=mean_absolute_error(self.all_test_pred["target"],self.all_test_pred["results"])
        mae_relative=mae/self.all_test_pred["target"].std()
        mse=mean_squared_error(self.all_test_pred["target"],self.all_test_pred["results"])
        mse_relative=mse/self.all_test_pred["target"].std()
        trainer.logger.experiment.log({
            "CorrSpearman "+self.prefix:corr.iloc[0,1],
            "mae "+self.prefix:mae,
            "mae relative "+self.prefix: mae_relative,
            "mse "+self.prefix :mse ,
            "mse relative "+self.prefix :mse_relative ,
            
                })
        self.all_test_pred["rank_target"]=self.all_test_pred.target.rank(method="average")
        self.all_test_pred["rank_results"]=self.all_test_pred.results.rank(method="average")
        self.all_test_pred=self.all_test_pred.sort_values("rank_target").reset_index(drop=True)

     
        self._plots_scatter_rank_plot(trainer)

    def on_train_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        a=trainer.logger

        self._generate_df_from_test(trainer,pl_module)
   
        return super().on_train_end(trainer, pl_module)
    
   
            
    def _plots_scatter_rank_plot(self,trainer:'pl.Trainer'):
        self._bar_rank_plot(trainer,
                            xlabel1="valores ordenador por target",
                            xlabel2="valores ordenador por results",
                            ylabel="puesto en el ranking",
                            title="grafico de barras para correlacionar valores por ranking")
        
        if "labels" in self.all_test_pred.columns:
            self.all_test_pred.to_csv("/home/dcast/adversarial_project/openml/results_to_Carlos.csv")
            self.all_test_pred=self.all_test_pred.sample(frac=0.01)
            self._scatter_plot(x=self.all_test_pred.target,
                           y=self.all_test_pred.results,
                           xname="target",
                           yname="results",
                           trainer=trainer,
                           title="Grafico de dispersion",
                           labels=self.all_test_pred.labels)
        else:
            self._scatter_plot(x=self.all_test_pred.target,
                           y=self.all_test_pred.results,
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
        plt.bar(self.all_test_pred.index,height=self.all_test_pred.rank_target)
        plt.bar(self.all_test_pred.index,height=self.all_test_pred.rank_results)
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