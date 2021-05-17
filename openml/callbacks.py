from pytorch_lightning.callbacks.base import Callback
import pandas as pd
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch
import wandb

import matplotlib.pyplot as plt
class PredictionPlotsAfterTrain(Callback):
    
    def __init__(self,dataloader:DataLoader,prefix=None) -> None:
        super(PredictionPlotsAfterTrain,self).__init__()
        self.dataloader=dataloader
        self.all_test_pred=pd.DataFrame()
        self.prefix=prefix
        
    def _generate_df_from_test(self,trainer: 'pl.Trainer',pl_module:'pl.LightningModule'):
        

        for batch in self.dataloader:
            image,target=batch
            with torch.no_grad():
                results=pl_module(image.to(device=pl_module.device))
            
            valid_pred_df = pd.DataFrame({
                # "image_id": img_paths,
                "target":target.cpu().numpy()[:,0],
                "results":results.cpu().numpy()[:,0],
                # "norm_PRED_Dffclt": valid_pred[:, 0],
                # "norm_PRED_Dscrmn": valid_pred[:, 1],
            })
            self.all_test_pred=pd.concat([self.all_test_pred,valid_pred_df])
        corr=self.all_test_pred.corr(method="spearman")
       
        trainer.logger.experiment.log({
            "CorrSpearman "+self.prefix:corr.iloc[0,1],
            
            "global_step":trainer.global_step
                })
        self.all_test_pred["rank_target"]=self.all_test_pred.target.rank(method="average")
        self.all_test_pred["rank_results"]=self.all_test_pred.results.rank(method="average")
        self.all_test_pred=self.all_test_pred.sort_values("rank_target").reset_index(drop=True)

        self._scatter_plot(x=self.all_test_pred.target,
                           y=self.all_test_pred.results,
                           xname="target",
                           yname="results",
                           trainer=trainer,
                           title="Grafico de dispersion")
        # self._scatter_plot(x=self.all_test_pred.rank_target,
        #                    y=self.all_test_pred.rank_results,
        #                    xname="rank_target",
        #                    yname="rank_results",
        #                    trainer=trainer,
        #                    ax=axs[1]
        #                    )

        self._bar_rank_plot(trainer,
                            xlabel1="valores ordenador por target",
                            xlabel2="valores ordenador por results",
                            ylabel="puesto en el ranking",
                            title="grafico de barras para correlacionar valores por ranking")
        
    def on_train_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        a=trainer.logger

        self._generate_df_from_test(trainer,pl_module)
        return super().on_train_end(trainer, pl_module)
    
    def _scatter_plot(self,x,y,xname,yname,trainer,title):
 
        fig = plt.figure(figsize=(14,7))
        plt.scatter(x=x,y=y)
        plt.title(title)
        plt.xlabel(xname)
        plt.ylabel(yname)
        plt.xlim([-6,6])
        plt.ylim([-6,6])
        plt.savefig("algo.jpg")
        trainer.logger.experiment.log({
            "graficas scatter "+self.prefix:wandb.Image(fig,caption="scatter plot"),
            
            # "global_step": trainer.global_step
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