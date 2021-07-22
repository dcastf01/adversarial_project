

import matplotlib.pyplot as plt

import seaborn as sns
import pandas as pd
import os
root_path="images"
data=pd.read_csv("data.csv")

# print(data.head())
# a=data.diff
# b=data.pert
#añadir al hue si se transforma en adversarial o no y realizar una gráfica solo con los que son advesariales
fig = plt.figure(figsize=(14,7))
def plot_scatter(df,y_variable):
    sns.scatterplot(x=df["diff"],y=df[y_variable],alpha=0.4) 

    plt.title(f"{y_variable} vs diff,method elastic l2=carlini")
    plt.xlabel("diff")
    plt.ylabel(f"{y_variable}")
    # plt.xlim(-4,-2)
    plt.savefig(os.path.join(root_path,f"{y_variable}_carlini_all_images.jpg"))
    plt.close()

def plot_pairplot(df,columns:list):

    data_aux=df[["diff"]+columns]
    sns.pairplot(data_aux,hue="class")
    plt.suptitle("pairplot maps")
    plt.savefig(os.path.join(root_path,"pairplot_only_adversarial_all_class.jpg"))
    plt.close()

def plot_correlation(df):
    corr = df.corr()
    corr.style.background_gradient(cmap='coolwarm').set_precision(2)
    sns.heatmap(df.corr(), annot=True, fmt='.4f', 
            cmap=plt.get_cmap('coolwarm'), cbar=False)
    plt.suptitle("correlations_adversarial_result_carlini")
    plt.savefig(os.path.join(root_path,"correlations_adversarial_result_carlini.jpg"))
    plt.close()
                                   
def plot_correlation_only_diff_and_l2(df):
    # df=df[df["diff"]>-4]
    # df=df[df["diff"]<-2]
    df=df[["diff","l2"]]
    corr = df.corr()
    corr.style.background_gradient(cmap='coolwarm').set_precision(2)
    sns.heatmap(df.corr(), annot=True, fmt='.4f', 
            cmap=plt.get_cmap('coolwarm'), cbar=False)
    plt.suptitle("correlations_adversarial_result_carlini")
    plt.savefig(os.path.join(root_path,"correlations_adversarial_result_carlini_diff_and_l222222.jpg"))
    plt.close()                            


columns_to_analyze=["linfinite","l2","l0","class"]
plot_pairplot(data,columns_to_analyze)
plot_correlation(data)
plot_correlation_only_diff_and_l2(data)
for column in columns_to_analyze:
    plot_scatter(data,column)
    
    
