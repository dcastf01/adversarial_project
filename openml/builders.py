import logging
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin

from openml.callbacks import PredictionPlotsAfterTrain
from openml.datamodule import OpenMLDataModule
from openml.config import CONFIG, Dataset


def build_dataset(path_data_csv:str,dataset_name:str=CONFIG.dataset_name,
                  batch_size:int=CONFIG.batch_size):
    
    
    dataset_enum=Dataset[dataset_name]

    data_module=OpenMLDataModule(data_dir=os.path.join(path_data_csv,dataset_enum.value),
                                            batch_size=batch_size,
                                            dataset=dataset_enum,
                                            num_workers=CONFIG.NUM_WORKERS,
                                            pin_memory=True)
        

    data_module.setup()
    return data_module

def get_callbacks(config,dm):
    #callbacks
    
    early_stopping=EarlyStopping(monitor='_val_loss',
                                 mode="min",
                                patience=5,
                                 verbose=True,
                                 check_finite =True
                                 )

    checkpoint_callback = ModelCheckpoint(
        monitor='_val_loss',
        dirpath=config.PATH_CHECKPOINT,
        filename= '-{epoch:02d}-{val_loss:.6f}',
        mode="min",
        save_last=True,
        save_top_k=3,
                        )
    learning_rate_monitor=LearningRateMonitor(logging_interval="epoch")
    
    prediction_plot_test=PredictionPlotsAfterTrain(dm.test_dataloader(),prefix="test")
    prediction_plot_train=PredictionPlotsAfterTrain(dm.train_dataloader(),prefix="train")
    callbacks=[
        prediction_plot_test,
        prediction_plot_train,
        learning_rate_monitor,
        early_stopping,
        
            ]
    return callbacks

def get_trainer(wandb_logger,callbacks,config):
    
    gpus=[]
    if config.gpu0:
        gpus.append(0)
    if config.gpu1:
        gpus.append(1)
    logging.info( "gpus active",gpus)
    if len(gpus)>=2:
        distributed_backend="ddp"
        accelerator="dpp"
        plugins=DDPPlugin(find_unused_parameters=False)
    else:
        distributed_backend=None
        accelerator=None
        plugins=None
        
    
        
    trainer=pl.Trainer(
                    logger=wandb_logger,
                       gpus=gpus,
                       max_epochs=config.NUM_EPOCHS,
                       precision=config.precision_compute,
                    #    limit_train_batches=0.1, #only to debug
                    #    limit_val_batches=0.05, #only to debug
                    #    val_check_interval=1,
                        auto_lr_find=config.AUTO_LR,
                       log_gpu_memory=True,
                    #    distributed_backend=distributed_backend,
                    #    accelerator=accelerator,
                    #    plugins=plugins,
                       callbacks=callbacks,
                       progress_bar_refresh_rate=5,
                       
                       )
    
    return trainer
