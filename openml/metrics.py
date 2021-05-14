import torch
import torchmetrics
from torchmetrics import MetricCollection,MeanAbsoluteError , MeanSquaredError,SpearmanCorrcoef,PearsonCorrcoef

def get_metrics_collections_base(prefix
                            # device="cuda" if torch.cuda.is_available() else "cpu",
                            
                            ):
    
    metrics = MetricCollection(
            {
                "MeanAbsoluteError":MeanAbsoluteError(),
                "MeanSquaredError":MeanSquaredError(),
                "SpearmanCorrcoef":SpearmanCorrcoef(),
                "PearsonCorrcoef":PearsonCorrcoef()
                
            },
            prefix=prefix
            )
    return metrics