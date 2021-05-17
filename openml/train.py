import datetime
import logging


import sys

from torch.utils import data
# sys.path.append("/content/adversarial_project") #to work in colab
import pytorch_lightning as pl
import torch
import wandb
from openml.config import CONFIG,create_config_dict

from pytorch_lightning.loggers import WandbLogger


from openml.autotune import autotune_lr
from openml.builders import build_dataset,get_trainer,get_callbacks
from openml.lit_regressor import LitRegressor
import os
os.environ["WANDB_IGNORE_GLOBS"]="*.ckpt"


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
    wandb.run.name=config.experiment_name[:5]+" "+\
                    datetime.datetime.utcnow().strftime("%Y-%m-%d %X")

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
        is_mlp_preconfig=config.is_mlp_preconfig
        
        
                )

    callbacks=get_callbacks(config,data_module)
    #create trainer
    trainer=get_trainer(wandb_logger,callbacks,config)
    
    model=autotune_lr(trainer,model,data_module,get_auto_lr=config.AUTO_LR)
    trainer.fit(model,data_module)
    
    

if __name__ == "__main__":
    
    main()