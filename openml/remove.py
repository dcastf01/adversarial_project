import torch
import timm
import pandas as pd
# input = torch.randn(3, 5, requires_grad=True)
# target = torch.empty(3, dtype=torch.long).random_(5)
# avail_pretrained_models = timm.list_models(pretrained=True)
# print(avail_pretrained_models)

data=pd.read_csv("/home/dcast/adversarial_project/openml/adversarial_images/mnist784_ref_data_adversarial.csv")
print(data.tail(5))
print(data.head(5))
print(data.describe())
print(data.nunique())
# print(data.id)
# for id in data.id:
#     print(id)
v = data.id.value_counts()
# print(data[data.id.isin(data.index[data.gt(5)])])
print(v)
print(data.shape)