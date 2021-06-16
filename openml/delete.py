import pandas as pd


cosa=pd.read_csv("/home/dcast/adversarial_project/openml/data/mnist_784V2.Diff6.RefClass.csv")
print(cosa.head(1))
print(cosa.columns)
original=pd.read_csv("/home/dcast/adversarial_project/openml/data/mnist_784.csv")
nando_df=pd.read_csv("/home/dcast/adversarial_project/openml/data/mnist_784.Diff6.RefClass.csv")


# print(nando_df.columns)
cosa=nando_df.merge(original)
print(cosa.columns)
print(cosa.shape)
cosa.to_csv("/home/dcast/adversarial_project/openml/data/mnist_784V2.Diff6.RefClass.csv",index=False)