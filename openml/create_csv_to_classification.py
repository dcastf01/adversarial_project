import pandas as pd


data=pd.read_csv("/home/dcast/adversarial_project/openml/data/mnist_784V2.Diff6.RefClass.csv")
median_dffclt=data.Dffclt.median()
data['Hard'] = [0 if x<median_dffclt else 1  for x in data['Dffclt']]

data.to_csv("/home/dcast/adversarial_project/openml/data/mnist_784V2.Clasification.csv",index=False)

print(data.Hard)
print(data.Hard.describe())
# print(data.Dffclt)
# print(data.Dffclt.describe())
