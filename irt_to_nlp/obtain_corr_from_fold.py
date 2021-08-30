from irt_to_nlp.config import ModelAvailable,Dataset

import pandas as pd
import os
import glob

def get_list_with_all_csv_with_format(path,lr:str,model:str,dataset:str) ->list:
    #the format is regressor_{repetition}_{fold}_{lr}_{train/val}_{dataset}_{model}.csv
    
    repetition=0
    fold_max=5
    split="val"
    format=f'{path}/*{lr}*{split}*{dataset}_{model}.csv'
    files=glob.glob(format)
    return files

def generate_dataframe_concated(list_csv:list)->pd.DataFrame:
    
    df=pd.DataFrame()
    li=[]
    for csv in list_csv:
        df_aux=pd.read_csv(csv,index_col="Unnamed: 0")
        li.append(df_aux)
        
    df=pd.concat(li, axis=0, ignore_index=True)
    # print(df.head())
    print(df.shape)
    print(df.nunique())
    return df

def generate_corr_and_rank(df:pd.DataFrame):
    
    corr=df.corr(method="spearman")  
    print(corr)
    
path_with_result:str="/home/dcast/adversarial_project/irt_to_nlp/data/results"

lr_used:str="5e-05"
model=ModelAvailable.bert_base_cased
model_str:str=model.name
dataset=Dataset.sst
dataset_str:str=dataset.name

    
files=get_list_with_all_csv_with_format(path_with_result,lr_used,model_str,dataset_str)

df=generate_dataframe_concated(files)

generate_corr_and_rank(df)