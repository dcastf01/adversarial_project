import datetime
import logging
import sys

from pytorch_lightning.core import datamodule
from torch.utils import data
#cd /home/dcast/adversarial_project ; /usr/bin/env /home/dcast/anaconda3/envs/deep_learning_torch/bin/python -- /home/dcast/adversarial_project/irt_to_nlp/train.py 
# sys.path.append("/content/adversarial_project") #to work in colab
sys.path.append("/home/dcast/adversarial_project")
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger

import wandb
from openml.autotune import autotune_lr
from irt_to_nlp.builders import build_dataset, get_callbacks, get_trainer,get_system
from irt_to_nlp.config import CONFIG, create_config_dict


def main():
    os.environ["WANDB_IGNORE_GLOBS"]="*.ckpt"
    print("empezando setup del experimento")
    torch.backends.cudnn.benchmark = True
    config=CONFIG()
    config_dict=create_config_dict(config)
    wandb.init(
        project='IRT-project-NLP',
                entity='dcastf01',
                config=config_dict)
    
    wandb_logger = WandbLogger( 
                    # offline=True,
                    log_model=False
                    )
    
    config =wandb.config
    # wandb.run.name=config.experiment_name[:5]+" "+\
    #                 datetime.datetime.utcnow().strftime("%Y-%m-%d %X")

    data_module=build_dataset(path_data_csv=config.path_data,
                              dataset_name=config.dataset_name,
                              batch_size=config.batch_size,
                              model_name=config.model_name
                              )
    
    model=get_system(config)

    callbacks=get_callbacks(config,data_module)
    #create trainer
    trainer=get_trainer(wandb_logger,callbacks,config)
    
    trainer.fit(model,data_module)
    
    

if __name__ == "__main__":
    
    main()
