import logging
def autotune_lr(trainer,model,data_module,get_auto_lr:bool=False):
    
        
    if get_auto_lr:
        logging.info("Buscando el learning rate ótimo entre 5 y 1-e4")
        # Run learning rate finder
        lr_finder = trainer.tuner.lr_find(model,data_module,max_lr=1,min_lr=1e-4)

        # # Results can be found in
        # lr_finder.results

        # Pick point based on plot, or get suggestion
        new_lr = lr_finder.suggestion()
        logging.info("El lr óptimo es {new_lr}")
        # update hparams of the model
        model.hparams.lr = new_lr
    return model