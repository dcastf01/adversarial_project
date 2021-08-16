import torch
import timm
# input = torch.randn(3, 5, requires_grad=True)
# target = torch.empty(3, dtype=torch.long).random_(5)
avail_pretrained_models = timm.list_models(pretrained=True)
print(avail_pretrained_models)