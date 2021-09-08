import datetime
import math
import os
from copy import deepcopy

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_grad_cam
import pytorch_lightning as pl
import seaborn as sns
import torch
from PIL import Image
from pytorch_grad_cam.utils.svd_on_activations import get_2d_projection
from pytorch_lightning.callbacks.base import Callback
from seaborn.palettes import color_palette
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

import wandb
from openml.config import CONFIG, Dataset
from openml.datamodule import OpenMLDataModule
from openml.lit_classifier import LitClassifier
from openml.loaders_to_experiments import (Cifar10LoaderExperiment4,
                                           MnistLoaderExperiment4,
                                           MnistLoaderExperimentWaterMark)

class ExperimentWaterMarkDataset(Callback):
    
    def __init__(self,dm,config:CONFIG) -> None:
        super().__init__()
        self.config=config
        self.path_save_result="/home/dcast/adversarial_project/openml/data/results_experiment_watermark"
        self.size_experiment=1000
        self.dataset_val=dm.data_val
        self.create_dataloaders()
        self.save_result=True
        self.img_to_print=False
        
    def create_dataloaders(self):
        
        self.data_val=self.dataset_val
        examples_to_discard=len(self.data_val)-self.size_experiment
        self.data_val_without_watermark,subdataset_original_val_to_discard=\
            random_split( self.data_val,
                         ( self.size_experiment,examples_to_discard)
                )
       
        
        self.data_val_without_watermark=DataLoader(
            dataset=self.data_val_without_watermark,
            batch_size=self.config.batch_size,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True,
            shuffle=False,
        )
        self.data_val_with_watermark=MnistLoaderExperimentWaterMark()
        examples_to_discard=len(self.data_val_with_watermark)-self.size_experiment
        self.data_val_with_watermark,self.data_train_with_watermark=\
            random_split( self.data_val_with_watermark,
                         ( self.size_experiment,examples_to_discard)
                )
        self.data_val_with_watermark=DataLoader(
            dataset=self.data_val_with_watermark,
            batch_size=self.config.batch_size,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True,
            shuffle=False,
        )
        self.data_train_with_watermark=DataLoader(
            dataset=self.data_train_with_watermark,
            batch_size=self.config.batch_size,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True,
            shuffle=False,
        )
    
    def on_train_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        self.create_and_train_classification_model(self.data_train_with_watermark)
        self.predict(pl_module,self.data_val_without_watermark,is_regressor=True)
        self._save_dataframe_in_csv("regressor_without_watermark")
        self.predict(pl_module=self.classifier,dataloader=self.data_val_without_watermark,
                          is_regressor=False)
        self._save_dataframe_in_csv("classifier_without_watermark")
        
        self.img_to_print=True
        self.predict(pl_module,self.data_val_with_watermark,is_regressor=True)
        self._save_dataframe_in_csv("regressor_with_watermark")
        
        self.predict(pl_module=self.classifier,dataloader=self.data_val_with_watermark,
                          is_regressor=False)
        self._save_dataframe_in_csv("classifier_with_watermark")
        return super().on_train_end(trainer, pl_module)
    
    def _save_dataframe_in_csv(self,additional_text):
        
        if self.save_result:
            extra_text=datetime.datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
            extra_text=additional_text+"_"+extra_text
            
            path_with_filename=os.path.join(self.path_save_result,f"{extra_text}.csv")
            self.df_pred.to_csv(path_with_filename) 
    
    def save_img(self,img):
        if self.img_to_print:
            first_array=img[0][0]
            first_array=first_array.numpy()
            #Not sure you even have to do that if you just want to visualize it
            first_array=255*first_array
            first_array=first_array.astype("uint8")
            plt.imshow(first_array) #multiplcar por
            #Actually displaying the plot if you are not in interactive mode
            plt.show()
            #Saving plot
            plt.savefig(os.path.join(self.path_save_result,"figwatermark.png"))
            self.img_to_print=False
    
    def predict(self,pl_module,dataloader,is_regressor=False):
        self.df_pred=pd.DataFrame()
        for batch in dataloader:
            if len(batch)==3:
                image,target,idx=batch
            elif len(batch)==4:
                image,target,idx,labels=batch
            self.save_img(image)
            with torch.no_grad():
                results=pl_module(image.to(device=pl_module.device))
            labels=labels.cpu().numpy()[:,0]
            target=target.cpu().numpy()[:,0]
            if is_regressor:
                results=results.cpu().numpy()[:,0]
            else:
                results=torch.argmax(results,dim=1).cpu().numpy()

            if len(batch)==3:
                valid_pred_df = pd.DataFrame({
                    "Dffclt":target,
                    "results":results,
                    "id_image": idx,
                })
            elif len(batch)==4:
                valid_pred_df = pd.DataFrame({
                    "Dffclt":target,
                    "results":results,
                    "labels":labels,#.cpu().numpy()[:,0],
                    "id_image": idx,
                })
            if not is_regressor:
                valid_pred_df["acierta"]=np.where( valid_pred_df['results'] == valid_pred_df['labels'] , '1', '0')
            self.df_pred=pd.concat([self.df_pred,valid_pred_df])
            
    def create_and_train_classification_model(self,train_dataloader):
        self.in_chans=1
        self.classifier=LitClassifier(
            model_name=self.config.experiment_name,
            lr=self.config.lr,
            optim=self.config.optim_name,
            in_chans=self.in_chans,
            num_fold=0,
            num_repeat=0
                             )
        trainer=pl.Trainer(
                    # logger=wandb_logger,
                       gpus=[0],
                       max_epochs=self.config.NUM_EPOCHS,
                       precision=self.config.precision_compute,
                       log_gpu_memory=True,
                       progress_bar_refresh_rate=5,
                       
                       )
        trainer.fit(self.classifier,
                    train_dataloader=train_dataloader,

                    )
class ExperimentBlurDataset(Callback):
    
    def __init__(self,dm,config:CONFIG) -> None:
        super().__init__()
        self.config=config
        self.path_save_result="/home/dcast/adversarial_project/openml/data/results_experiment_blur"
        self.size_experiment=1000
        self.dataset_val=dm.data_val
        self.create_dataloaders()
        self.save_result=True
        self.img_to_print=False
        
    
    def create_dataloaders(self):
        
        self.data_val=self.dataset_val
        examples_to_discard=len(self.data_val)-self.size_experiment
        self.data_val_without_blur,subdataset_original_val_to_discard=\
            random_split( self.data_val,
                         ( self.size_experiment,examples_to_discard)
                )
        self.data_val_with_blur=deepcopy( self.data_val_without_blur)
        new_transform=transforms.Compose([
                                    transforms.Resize((32, 32), Image.BILINEAR),
                                    transforms.GaussianBlur((13),sigma=(10,20)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(0.5, 0.5)]
                       )  
        self.data_val_with_blur.dataset.dataset.transform=new_transform
        
        self.data_val_without_blur=DataLoader(
            dataset=self.data_val_without_blur,
            batch_size=self.config.batch_size,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True,
            shuffle=False,
        )
        self.data_val_with_blur=DataLoader(
            dataset=self.data_val_with_blur,
            batch_size=self.config.batch_size,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True,
            shuffle=False,
        )
   
    def on_train_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        self.create_and_train_classification_model(trainer.datamodule)
        self.predict(pl_module,self.data_val_without_blur,is_regressor=True)
        self._save_dataframe_in_csv("regressor_without_blur")
        self.predict(pl_module=self.classifier,dataloader=self.data_val_without_blur,
                          is_regressor=False)
        self._save_dataframe_in_csv("classifier_without_blur")
        
        self.img_to_print=True
        self.predict(pl_module,self.data_val_with_blur,is_regressor=True)
        self._save_dataframe_in_csv("regressor_with_blur")
        
        self.predict(pl_module=self.classifier,dataloader=self.data_val_with_blur,
                          is_regressor=False)
        self._save_dataframe_in_csv("classifier_with_blur")
        return super().on_train_end(trainer, pl_module)
    
    def _save_dataframe_in_csv(self,additional_text):
        
        if self.save_result:
            extra_text=datetime.datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
            extra_text=additional_text+"_"+extra_text
            
            path_with_filename=os.path.join(self.path_save_result,f"{extra_text}.csv")
            self.df_pred.to_csv(path_with_filename) 
    
    def save_img(self,img):
        if self.img_to_print:
            first_array=img[0][0]
            first_array=first_array.numpy()
            #Not sure you even have to do that if you just want to visualize it
            first_array=255*first_array
            first_array=first_array.astype("uint8")
            plt.imshow(first_array) #multiplcar por
            #Actually displaying the plot if you are not in interactive mode
            plt.show()
            #Saving plot
            plt.savefig(os.path.join(self.path_save_result,"fig13_10_20kernel.png"))
            self.img_to_print=False
    
    def predict(self,pl_module,dataloader,is_regressor=False):
        self.df_pred=pd.DataFrame()
        for batch in dataloader:
            if len(batch)==3:
                image,target,idx=batch
            elif len(batch)==4:
                image,target,idx,labels=batch
            self.save_img(image)
            with torch.no_grad():
                results=pl_module(image.to(device=pl_module.device))
            labels=labels.cpu().numpy()[:,0]
            target=target.cpu().numpy()[:,0]
            if is_regressor:
                results=results.cpu().numpy()[:,0]
                
            else:
                results=torch.argmax(results,dim=1).cpu().numpy()

            if len(batch)==3:
                valid_pred_df = pd.DataFrame({

                    "Dffclt":target,
                    "results":results,
                    "id_image": idx,
                })
            elif len(batch)==4:
               
                valid_pred_df = pd.DataFrame({
                    "Dffclt":target,
                    "results":results,
                    "labels":labels,#.cpu().numpy()[:,0],
                    "id_image": idx,
                })
            if not is_regressor:
                valid_pred_df["acierta"]=np.where( valid_pred_df['results'] == valid_pred_df['labels'] , '1', '0')
            self.df_pred=pd.concat([self.df_pred,valid_pred_df])
            
    def create_and_train_classification_model(self,dm):
        self.in_chans=1
        self.classifier=LitClassifier(
            model_name=self.config.experiment_name,
            lr=self.config.lr,
            optim=self.config.optim_name,
            in_chans=self.in_chans,
            num_fold=0,
            num_repeat=0
                             )
        trainer=pl.Trainer(
                    # logger=wandb_logger,
                       gpus=[0],
                       max_epochs=self.config.NUM_EPOCHS,
                       precision=self.config.precision_compute,
                       log_gpu_memory=True,
                       progress_bar_refresh_rate=5,
                       
                       )
        trainer.fit(self.classifier,
                    datamodule=dm,
                    # train_dataloader= self.data_train_without_adversarial,
                    # val_dataloaders=self.data_val_without_adversarial
                    )
class ExperimentShiftDataset(Callback):
    def __init__(self,dm,config:CONFIG) -> None:
        super().__init__()
        
        self.config=config
        self.size_experiment=500
        self.dataset_original=dm#ver como poner
        self.create_original_dataset()
        self.create_new_dataset()
        
        self.save_result=True
        self.path_save_result="/home/dcast/adversarial_project/openml/data/results_experiment_shift"
    def create_original_dataset(self):
        
        self.original_data_val=self.dataset_original.data_val
        examples_to_discard=len(self.original_data_val)-self.size_experiment
        self.original_data_val,subdataset_original_val_to_discard=\
            random_split( self.original_data_val,
                         ( self.size_experiment,examples_to_discard)
                )
        self.original_data_val=DataLoader(
            dataset=self.original_data_val,
            batch_size=self.config.batch_size,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True,
            shuffle=False,
        )
    def create_new_dataset(self,):
        self.new_dataset=Dataset.fashionmnist_ref
        self.new_dm=OpenMLDataModule( data_dir=os.path.join(self.config.path_data,self.new_dataset.value),
                                            batch_size=self.config.batch_size,
                                            dataset=self.new_dataset,
                                            num_workers=CONFIG.NUM_WORKERS,
                                            pin_memory=True)
        self.new_dm.setup()
        self.new_data_val=self.new_dm.data_val
        # self.new_data_val=self.dataset_original
        examples_to_discard=len(self.new_data_val)-self.size_experiment
        self.new_data_val,subdataset_new_val_to_discard=\
            random_split( self.new_data_val,
                         ( self.size_experiment,examples_to_discard)
                )
        self.new_data_val=DataLoader(
            dataset=self.new_data_val,
            batch_size=self.config.batch_size,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True,
            shuffle=False,
        )
    
    def on_train_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        #experimento con imágenes originales y resultados
        #experimento con imágenes de otro conjunto y resultados
        self.experiment(is_original_dataset=True,pl_module=pl_module)
        self.experiment(is_original_dataset=False,pl_module=pl_module)
        return super().on_train_end(trainer, pl_module)

    def experiment(self,is_original_dataset:bool, pl_module: 'pl.LightningModule'):
        if is_original_dataset:
            #solo permitir 500 imágenes
            self.predict(pl_module,self.original_data_val,is_regressor=True)
            self._save_dataframe_in_csv("regressor_with_original")
        else:
            #solo permitir 500 imágenes
            self.predict(pl_module,self.new_data_val,is_regressor=True)
            self._save_dataframe_in_csv("regressor_with_shift_dataset")
    def _save_dataframe_in_csv(self,additional_text):
        
        if self.save_result:
            extra_text=datetime.datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
            extra_text=additional_text+"_"+extra_text
            
            path_with_filename=os.path.join(self.path_save_result,f"{extra_text}.csv")
            self.df_pred.to_csv(path_with_filename) 
    
    def predict(self,pl_module,dataloader,is_regressor=False):
        self.df_pred=pd.DataFrame()
        for batch in dataloader:
            if len(batch)==3:
                image,target,idx=batch
            elif len(batch)==4:
                image,target,idx,labels=batch
            with torch.no_grad():
                results=pl_module(image.to(device=pl_module.device))
            labels=labels.cpu().numpy()[:,0]
            target=target.cpu().numpy()[:,0]
            if is_regressor:
                results=results.cpu().numpy()[:,0]
                
            else:
                results=torch.argmax(results,dim=1).cpu().numpy()

            if len(batch)==3:
                valid_pred_df = pd.DataFrame({

                    "Dffclt":target,
                    "results":results,
                    "id_image": idx,
                })
            elif len(batch)==4:
               
                valid_pred_df = pd.DataFrame({
                    "Dffclt":target,
                    "results":results,
                    "labels":labels,#.cpu().numpy()[:,0],
                    "id_image": idx,
                })
            if not is_regressor:
                valid_pred_df["acierta"]=np.where( valid_pred_df['results'] == valid_pred_df['labels'] , '1', '0')
            self.df_pred=pd.concat([self.df_pred,valid_pred_df])

class Experiment4WithAdversarialExamples(Callback):
    
    def __init__(self,dm,config:CONFIG) -> None:
        super().__init__()
        self.train_val_dataset_initial=torch.utils.data.ConcatDataset([dm.data_train,dm.data_val])
        self.config=config
        dir_csv_file="/home/dcast/adversarial_project/openml/adversarial_images/mnist784_ref_data_adversarial.csv"
        self.path_save_result="/home/dcast/adversarial_project/openml/data/results_experiment_4"
        self.df_adversarial=pd.read_csv(dir_csv_file)
        self.number_examples_originals=len(self.train_val_dataset_initial)
        self.number_examples_adversarials=len(self.df_adversarial)
        self.size_experiment=1000
        all_range_index=list(range(0,self.train_val_dataset_initial.cumulative_sizes[-1]))
        self.ids_adversarial_examples=self.df_adversarial.id.tolist()
        self.ids_without_adversarial=[x for x in all_range_index if x not in self.ids_adversarial_examples]
      
        # self.subdataset_adversarial=Cifar10LoaderExperiment4(dir_csv_file,"32")
        # self.in_chans=3
        self.subdataset_adversarial=MnistLoaderExperiment4(32)
        self.in_chans=1
        self.dataset_without_adversarial=torch.utils.data.Subset(self.train_val_dataset_initial,self.ids_without_adversarial)
        self.train_val_split= [round(split*(self.number_examples_originals-self.number_examples_adversarials)) 
                               for split in [0.7,0.3]]
        self.save_result=True
        self.batch_size=1024
        self.num_workers=0
        self.pin_memory=True
        
    def create_dataset_to_train(self,trainer):
        
        self.data_train_without_adversarial, self.data_val_without_adversarial= random_split(
            self.dataset_without_adversarial, self.train_val_split
        )
        trainer.datamodule.data_train=self.data_train_without_adversarial #solo coger el 70% del total
        # trainer.datamodule.data_train=self.train_val_dataset_initial    
        trainer.datamodule.data_val=self.data_val_without_adversarial
    
    def create_dataset_with_adversarial_to_experiment(self):
        examples_necessary_without_adversarial=self.size_experiment-self.number_examples_adversarials
        examples_to_discard=len(self.data_val_without_adversarial)-examples_necessary_without_adversarial
        self.subdataset_val_without_adversarial_to_use_in_experiment,subdataset_val_without_adversarial_to_discard=\
            random_split( self.data_val_without_adversarial,
                         ( examples_necessary_without_adversarial,examples_to_discard)
                )
        self.dataset_adversarial_to_experiment=\
            torch.utils.data.ConcatDataset([self.subdataset_adversarial,
                                    self.subdataset_val_without_adversarial_to_use_in_experiment])
            
        self.dataset_adversarial_to_experiment=DataLoader(
            dataset=self.dataset_adversarial_to_experiment,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
    
    def create_dataset_without_adversarial_to_experiment(self):
        examples_to_discard=len(self.data_val_without_adversarial)-self.size_experiment
        self.dataset_without_adversarial_to_experiment,subdataset_val_without_adversarial_to_discard=\
            random_split( self.data_val_without_adversarial,
                         ( self.size_experiment,examples_to_discard)
                )
        self.dataset_without_adversarial_to_experiment=DataLoader(
            dataset=self.dataset_without_adversarial_to_experiment,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
    
    def on_fit_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        self.create_dataset_to_train(trainer)
        self.create_dataset_with_adversarial_to_experiment()
        self.create_dataset_without_adversarial_to_experiment()
        return super().on_train_start(trainer, pl_module)
     
    def on_train_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        self.create_and_train_classification_model(trainer.datamodule)
        self.experiment_with_adversarial(pl_module)
        self.experiment_without_adversarial(pl_module)        
        return super().on_train_end(trainer, pl_module)
    
    def create_and_train_classification_model(self,dm):
       
        self.classifier=LitClassifier(
            model_name=self.config.experiment_name,
            lr=self.config.lr,
            optim=self.config.optim_name,
            in_chans=self.in_chans,
            num_fold=0,
            num_repeat=0
                             )
        trainer=pl.Trainer(
                    # logger=wandb_logger,
                       gpus=[0],
                       max_epochs=self.config.NUM_EPOCHS,
                       precision=self.config.precision_compute,
                       log_gpu_memory=True,
                       progress_bar_refresh_rate=5,
                       
                       )
        trainer.fit(self.classifier,
                    datamodule=dm,
                    # train_dataloader= self.data_train_without_adversarial,
                    # val_dataloaders=self.data_val_without_adversarial
                    )
                            
    def experiment_with_adversarial(self,pl_module):
        self.predict(pl_module=pl_module,dataloader=self.dataset_adversarial_to_experiment,
                          is_regressor=True)
        self._save_dataframe_in_csv("regressor_with_adversarial")
        self.predict(pl_module=self.classifier,dataloader=self.dataset_adversarial_to_experiment,
                          is_regressor=False)
        self._save_dataframe_in_csv("classifier_with_adversarial")
        #predecir dificultad aportando dataset
        #predecir clasificación aportando dataset
        
    def experiment_without_adversarial(self,pl_module):
        self.predict(pl_module=pl_module,dataloader=self.dataset_without_adversarial_to_experiment,
                          is_regressor=True)
        self._save_dataframe_in_csv("regressor_without_adversarial")
        self.predict(pl_module=self.classifier,dataloader=self.dataset_without_adversarial_to_experiment,
                          is_regressor=False)
        self._save_dataframe_in_csv("classifier_without_adversarial")
        #predecir dificultad aportando dataset
        #predecir clasificación aportando dataset
    
    def _save_dataframe_in_csv(self,additional_text):
        
        if self.save_result:
            extra_text=datetime.datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
            extra_text=additional_text+"_"+extra_text
            
            path_with_filename=os.path.join(self.path_save_result,f"{extra_text}.csv")
            self.df_pred.to_csv(path_with_filename)    
            
    def predict(self,pl_module,dataloader,is_regressor=False):
        self.df_pred=pd.DataFrame()
        for batch in dataloader:
            if len(batch)==3:
                image,target,idx=batch
            elif len(batch)==4:
                image,target,idx,labels=batch
            with torch.no_grad():
                results=pl_module(image.to(device=pl_module.device))
            labels=labels.cpu().numpy()[:,0]
            target=target.cpu().numpy()[:,0]
            if is_regressor:
                results=results.cpu().numpy()[:,0]
                
            else:
                results=torch.argmax(results,dim=1).cpu().numpy()
                
            if len(batch)==3:
                valid_pred_df = pd.DataFrame({

                    "Dffclt":target,
                    "results":results,
                    "id_image": idx,
                })
            elif len(batch)==4:
               
                valid_pred_df = pd.DataFrame({
                    "Dffclt":target,
                    "results":results,
                    "labels":labels,#.cpu().numpy()[:,0],
                    "id_image": idx,
                })
            if not is_regressor:
                valid_pred_df["acierta"]=np.where( valid_pred_df['results'] == valid_pred_df['labels'] , '1', '0')
                # (valid_pred_df["results"]==valid_pred_df["labels"])
            self.df_pred=pd.concat([self.df_pred,valid_pred_df])

class SplitDatasetWithKFoldStrategy(Callback):
    
    def __init__(self,folds,repetitions,dm,only_train_and_test=False) -> None:
        super().__init__()
        self.folds=folds
        self.repetitions=repetitions
        self.train_val_dataset_initial=torch.utils.data.ConcatDataset([dm.data_train,dm.data_val])
        self.only_train_and_test=only_train_and_test
        kf = KFold(n_splits=folds)

        self.indices_folds={}
        
        for fold, (train_ids, test_ids) in enumerate(kf.split(self.train_val_dataset_initial)):
            self.indices_folds[fold]={
                "train_ids":train_ids,
                "test_ids":test_ids
            }
        self.current_fold=0   

    def create_fold_dataset(self,num_fold,trainer,pl_module):
        
        train_ids=self.indices_folds[num_fold]["train_ids"]
        test_ids=self.indices_folds[num_fold]["test_ids"]
        trainer.datamodule.data_train=torch.utils.data.Subset(self.train_val_dataset_initial,train_ids)
        trainer.datamodule.data_val=torch.utils.data.Subset(self.train_val_dataset_initial,test_ids)
    
    def create_all_train_dataset(self,trainer):
        
        trainer.datamodule.data_val=trainer.datamodule.data_test
        trainer.datamodule.data_train=self.train_val_dataset_initial

    def on_fit_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        if self.only_train_and_test:
            self.create_all_train_dataset(trainer)
        else:
            self.create_fold_dataset(pl_module.num_fold,trainer,pl_module)
        return super().on_train_start(trainer, pl_module)
    
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
    
    def __init__(self,dataset_name:str,
                 model_name:str,
                 split:str=None,
                 is_regressor:bool=True,
                 lr_used:int=0.001,
                 save_result:bool=False,
                 
                 ) -> None:
        super(PredictionPlotsAfterTrain,self).__init__()
        self.df_pred=pd.DataFrame()
        self.split=split
        self.folder_images="/home/dcast/adversarial_project/openml/results"
        self.folder_csv_result="/home/dcast/adversarial_project/openml/data/results"
        self.prefix=split
        self.dataset_name=dataset_name
        self.model_name=model_name
        self.is_regressor=is_regressor
        self.lr_used=lr_used
        self.save_result=save_result
        
    def _generate_df_from_split_depend_on_target_model(self,trainer: 'pl.Trainer',pl_module:'pl.LightningModule'):
        self.df_pred=pd.DataFrame()
        for batch in self.dataloader:
            if len(batch)==3:
                image,target,idx=batch
            elif len(batch)==4:
                image,target,idx,labels=batch
            with torch.no_grad():
                results=pl_module(image.to(device=pl_module.device))
            labels=labels.cpu().numpy()[:,0]
            target=target.cpu().numpy()[:,0]
            if self.is_regressor:
                results=results.cpu().numpy()[:,0]
                
            else:
                # target=target.cpu().numpy()[:,0]
                # labels=labels.cpu().numpy()[:,0]
                # a=torch.argmax(results,dim=1).cpu().numpy()
                results=torch.argmax(results,dim=1).cpu().numpy()
                
                # results=results.cpu().numpy()
                
            if len(batch)==3:
                valid_pred_df = pd.DataFrame({
                    # "image":image.cpu().numpy()[:,0],
                    "Dffclt":target,
                    "results":results,
                    "id_image": idx,
                    # "norm_PRED_Dffclt": valid_pred[:, 0],
                    # "norm_PRED_Dscrmn": valid_pred[:, 1],
                })
            elif len(batch)==4:
               
                valid_pred_df = pd.DataFrame({
                    # "image":image.cpu().numpy()[:,0],
                    "Dffclt":target,
                    "results":results,
                    "labels":labels,#.cpu().numpy()[:,0],
                    "id_image": idx,
                    # "norm_PRED_Dffclt": valid_pred[:, 0],
                    # "norm_PRED_Dscrmn": valid_pred[:, 1],
                })
            if not self.is_regressor:
                valid_pred_df["acierta"]=np.where( valid_pred_df['results'] == valid_pred_df['labels'] , '1', '0')
                # (valid_pred_df["results"]==valid_pred_df["labels"])
            self.df_pred=pd.concat([self.df_pred,valid_pred_df])
            # print(self.df_pred.head(5))
            
            
    def _generate_results_if_target_model_is_regressor(self,trainer: 'pl.Trainer',pl_module:'pl.LightningModule'):
        if self.is_regressor:
            corr=self.df_pred.corr(method="spearman")        
            mae=mean_absolute_error(self.df_pred["Dffclt"],self.df_pred["results"])
            mae_relative=mae/self.df_pred["Dffclt"].std()
            mse=mean_squared_error(self.df_pred["Dffclt"],self.df_pred["results"])
            mse_relative=mse/self.df_pred["Dffclt"].std()
            trainer.logger.experiment.log({
                "CorrSpearman "+self.prefix:corr.iloc[0,1],
                "mae "+self.prefix:mae,
                "mae relative "+self.prefix: mae_relative,
                "mse "+self.prefix :mse ,
                "mse relative "+self.prefix :mse_relative ,
                
                    })
            self.df_pred["rank_Dffclt"]=self.df_pred.Dffclt.rank(method="average")
            self.df_pred["rank_results"]=self.df_pred.results.rank(method="average")
            self.df_pred=self.df_pred.sort_values("rank_Dffclt").reset_index(drop=True)
            
                
            ##plotear imágenes dificiles
            # df_sorted_hard=self.df_pred.sort_values("target",ascending=False).head(5)
            # text=self.prefix+" higher"
            # self.generate_images_and_upload(trainer,df_sorted_hard,text=text)
            # ##plotear imágenes fáciles
            # df_sorted_easy=self.df_pred.sort_values("target",ascending=True).head(5)
            # text=self.prefix+" lowest"
            # self.generate_images_and_upload(trainer,df_sorted_easy,text=text)
            
            # self.df_pred["error"]=(self.df_pred["target"]-self.df_pred["results"]).abs()
        
            # df_sorted_less_error=self.df_pred.sort_values("error",ascending=True).head(5)
            # print(df_sorted_less_error)
            # self.__generate_image_with_grad_cam(df_sorted_hard,trainer,pl_module,"hard")
            # self.__generate_image_with_grad_cam(df_sorted_easy,trainer,pl_module,"easy")
            # self.__generate_image_with_grad_cam(df_sorted_less_error,trainer,pl_module,"minor_error")
            
            self._plots_scatter_rank_plot(trainer) #reactivate
            
            
            ##plotear imágenes dificiles que han sido predichas como fáciles
            # self.df_pred["difference"]=self.df_pred["target"]-self.df_pred["results"]
            # df_images_predict_easy_but_the_true_is_there_are_hard= self.df_pred.sort_values("difference",ascending=False).head(5)
            # text=self.prefix+" Hard but predict easy"
            # self.generate_images_and_upload(trainer,df_images_predict_easy_but_the_true_is_there_are_hard,text=text)
            # ##plotear imágenes fáciles que han sido predichas como dificiles

            # df_images_predict_hard_but_the_true_is_there_are_easy= self.df_pred.sort_values("difference",ascending=True).head(5)
            # # print(df_images_predict_hard_but_the_true_is_there_are_easy)
            # text=self.prefix+" Easy but predict hard"
            # self.generate_images_and_upload(trainer,df_images_predict_hard_but_the_true_is_there_are_easy,text=text)
    
    def _save_dataframe_in_csv(self,pl_module):
        
        if self.save_result:
            additional_text=str(pl_module.num_repeat)+"_" +str(pl_module.num_fold)
            extra_text="regressor" if self.is_regressor else "classification"
            extra_text=extra_text+"_"+additional_text+"_"+str(self.lr_used)
            
            path_with_filename=os.path.join(self.folder_csv_result,f"{extra_text}_{self.split}_{self.dataset_name}_{self.model_name}.csv")
            self.df_pred.to_csv(path_with_filename)
        
    def on_train_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        if self.split=="train":
            self.dataloader=trainer.datamodule.train_dataloader()
            self._generate_df_from_split_depend_on_target_model(trainer,pl_module) 
            self._generate_results_if_target_model_is_regressor(trainer,pl_module)
            self._save_dataframe_in_csv(pl_module)
        elif self.split=="val":
            self.dataloader=trainer.datamodule.val_dataloader()
            self._generate_df_from_split_depend_on_target_model(trainer,pl_module) 
            self._generate_results_if_target_model_is_regressor(trainer,pl_module) 
            self._save_dataframe_in_csv(pl_module)

        return super().on_train_end(trainer, pl_module)
    
    def on_test_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        print("hook test end")
        if self.split=="test":
            self.dataloader=trainer.datamodule.test_dataloader()
            self._generate_df_from_split_depend_on_target_model(trainer,pl_module) 
            self._generate_results_if_target_model_is_regressor(trainer,pl_module) 
            self._save_dataframe_in_csv(pl_module)
            
        return super().on_test_end(trainer, pl_module)
    
    def __generate_image_with_grad_cam(self,df,trainer:'pl.Trainer',pl_module,text):
        def revert_normalization(img,mean,std):
            return (img*std+mean)
        target_layer=list(pl_module.model.children())[-4] #con menos 4 funciona
        cam=gradCAMRegressorOneChannel(model=pl_module,target_layer=target_layer,use_cuda=True)
        df=df.head(5)
        normalize=trainer.datamodule.data_train.dataset.transform.transforms[0]
        # normalize=
        mean=normalize.mean
        std=normalize.std
        if "labels" in df.columns:
            iterator=zip(df.id_image,df.labels,df.target,df.results)
        else:
            iterator=zip (df.id_image, df.target,df.results)
        for batch in iterator:  
            if len(batch)==3:#isinstance(batch,int):
                idx,target,results=batch
                label=None
                
            else:
                idx,label,target,results=batch     
            image=torch.unsqueeze(self.dataloader.dataset.dataset._create_image_from_dataframe(idx),dim=0).to(device=pl_module.device)
            grayscale_cam=cam(input_tensor=image,eigen_smooth=False)#si no funciona poner en True
            
            # In this example grayscale_cam has only one image in the batch:
            grayscale_cam = grayscale_cam[0, :]

            if np.isnan(grayscale_cam).any():
                print("hola, alguno es nan")

            image=image.cpu().numpy()
            gray=image[0,:,:,:]
            gray=np.moveaxis(gray,0,-1)
            if gray.shape[-1]!=3:
               
                img_bw_with_3_channels=cv2.merge((gray,gray,gray))
                img_to_save=np.uint8((img_bw_with_3_channels+1)*127.5)
            else:
                img_bw_with_3_channels=gray    
                img_bw_with_3_channels=revert_normalization(img_bw_with_3_channels,mean,std)
                img_to_save=np.uint8(img_bw_with_3_channels*255)
            # 
            img=Image.fromarray(img_to_save)
            img.save(os.path.join(self.folder_images,f"{text} {idx} image_3_channel.png"))
            heatmap=cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_PLASMA   )
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            heatmap = np.float32(heatmap)
            cv2.imwrite(os.path.join(self.folder_images,f'{text} {idx} heatmap_cam.jpg'), heatmap)
            alpha=0.5
            beta=(1.0-alpha)
            
            dst = np.uint8(alpha*(heatmap)+beta*(img_to_save))
            cv2.imwrite(os.path.join(self.folder_images,f"{text} {idx} mixture.jpg"), dst)
            visualization = pytorch_grad_cam.utils.image.show_cam_on_image(img_bw_with_3_channels,
                                                                        grayscale_cam,use_rgb=True,
                                                                        colormap=cv2.COLORMAP_PLASMA  )
            img=Image.fromarray(visualization)
            
            img.save(os.path.join(self.folder_images,f"{text} {idx} probando.png"))
            trainer.logger.experiment.log({
                "graficas gradcam "+self.prefix:wandb.Image(img,caption=f" {idx} grad cam, Label {label}, Target: {target}, Pred: {results} "),
                            })
            
    def _plots_scatter_rank_plot(self,trainer:'pl.Trainer'):
        self._bar_rank_plot(trainer,
                            xlabel1="valores ordenador por target",
                            xlabel2="valores ordenador por results",
                            ylabel="puesto en el ranking",
                            title="grafico de barras para correlacionar valores por ranking")
        
        if "labels" in self.df_pred.columns:
            self.df_pred.to_csv("/home/dcast/adversarial_project/openml/results_to_Carlos.csv")
            # self.df_pred=self.df_pred.sample(frac=0.01)
            self._scatter_plot(x=self.df_pred.Dffclt,
                           y=self.df_pred.results,
                           xname="target",
                           yname="results",
                           trainer=trainer,
                           title="Grafico de dispersion",
                           labels=self.df_pred.labels)
        else:
            self._scatter_plot(x=self.df_pred.Dffclt,
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
            # 
            number_labels=self.df_pred.labels.nunique()
            if number_labels>10:
                # print(self.df_pred.head(5))
                plt.scatter(x=x,y=y,c=labels,alpha=alpha)
            else:
                color_pallete=sns.color_palette("tab10",n_colors=number_labels) #un error extraño
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
        plt.bar(self.df_pred.index,height=self.df_pred.rank_Dffclt)
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
