import pandas as pd

file_with_dffclt="/home/dcast/adversarial_project/openml/data/Fashion-MNIST.Diff6.RefClass.csv"
file_with_labels="/home/dcast/adversarial_project/openml/data/Fashion-MNIST.csv"
data_dffclt=pd.read_csv(file_with_dffclt,index_col="Unnamed: 0")

data_labels=pd.read_csv(file_with_labels)
columns=data_labels.columns[:-1]

print(columns,"labels")
print(data_dffclt.columns,"dffclt")
# data_total=pd.concat([data_labels,data_dffclt],join="right",axis=1,keys=columns)
# data_total=pd.merge(data_dffclt,data_labels,on=[columns],how="outer")
# data_total = data_total.loc[:,~data_total.columns.duplicated()]
data_total=data_dffclt.join(data_labels,on=columns,join="inner")
# dffclt_and_labels=data_total[["Dffclt","class"]]
# print(dffclt_and_labels.head(5))
print(data_labels.shape)
print(data_dffclt.shape)
print(data_total.shape)
# print(data_total.columns)

print(data_total.head(5))

# data=pd.read_csv("/home/dcast/adversarial_project/openml/data/mnist_784V2.Diff6.RefClass.csv")
# median_dffclt=data.Dffclt.median()
# data['Hard'] = [0 if x<median_dffclt else 1  for x in data['Dffclt']]

# data.to_csv("/home/dcast/adversarial_project/openml/data/mnist_784V2.Clasification.csv",index=False)

# print(data.Hard)
# print(data.Hard.describe())
# # print(data.Dffclt)
# print(data.Dffclt.describe())
