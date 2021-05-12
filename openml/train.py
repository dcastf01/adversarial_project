import datetime
import logging


import sys

from torch.utils import data
# sys.path.append("")
import pytorch_lightning as pl
import torch
import wandb
from openml.config import CONFIG,create_config_dict
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin

from openml.autotune import autotune_lr
from openml.build_dataset import build_dataset
from openml.lit_regressor import LitRegressor


def main():
    print("empezando setup del experimento")
    torch.backends.cudnn.benchmark = True
    config=CONFIG()
    config_dict=create_config_dict(config)
    wandb.init(
        project='IRT-project',
                entity='dcastf01',
                config=config_dict)
    
    wandb_logger = WandbLogger(
                    # offline=True,
                    )
    
    config =wandb.config
    wandb.run.name=config.experiment_name+" "+\
                    datetime.datetime.utcnow().strftime("%Y-%m-%d %X"),
    if config.root_path:
        sys.path.append(config.root_path)
    data_module=build_dataset(path_data_csv=config.path_data,
                              dataset_name=config.dataset_name,
                              batch_size=config.batch_size
                              )
    
    model=LitRegressor(
        experiment_name=config.experiment_name,
        lr=config.lr,
        optim=config.optim_name,
        features_out_layer1=config.features_out_layer1,
        features_out_layer2=config.features_out_layer2,
        features_out_layer3=config.features_out_layer3,
        tanh1=config.tanh1,
        tanh2=config.tanh2,
        dropout1=config.dropout1,
        dropout2=config.dropout2,
        
        
                )
    ##callbacks
    
    early_stopping=EarlyStopping(
                            monitor='_validMeanSquaredError',
                            mode="min",
                            patience=5)
    
    trainer=pl.Trainer(
                        logger=wandb_logger,
                       gpus=[0],
                       max_epochs=config.NUM_EPOCHS,
                       precision=config.precision_compute,
                    #    limit_train_batches=0.1, #only to debug
                    #    limit_val_batches=0.05, #only to debug
                    #    val_check_interval=1,
                        auto_lr_find=config.AUTO_LR,
                       log_gpu_memory=True,
                    #    distributed_backend='ddp',
                    #    accelerator="dpp",
                    #    plugins=DDPPlugin(find_unused_parameters=False),
                       callbacks=[
                            # early_stopping ,
                            # checkpoint_callback,
                            # confusion_matrix_wandb,
                            # learning_rate_monitor 
                                  ],
                       progress_bar_refresh_rate=5,
                       )
    
    model=autotune_lr(trainer,model,data_module,get_auto_lr=config.AUTO_LR)
    trainer.fit(model,data_module)
    
    

if __name__ == "__main__":
    
    main()