import torch
import torchmetrics
from torchmetrics import MetricCollection,MeanAbsoluteError , MeanSquaredError,PearsonCorrcoef

def get_metrics_collections_base(
                            # device="cuda" if torch.cuda.is_available() else "cpu",
                            
                            ):
    
    metrics = MetricCollection(
            {
                "MeanAbsoluteError":MeanAbsoluteError(),
                "MeanSquaredError":MeanSquaredError(),
                "PearsonCorrcoef":PearsonCorrcoef(),
                
            }
            )
    return metrics