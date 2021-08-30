import pandas as pd

import glob
def get_list_with_all_csv_with_format(path,model_type:str,adversarial:str) ->list:
    #the format is {model_type}_ {with/without}*date/hour.csv
    
    #adversarial=with/without
    
    format=f'{path}/{model_type}_{adversarial}_*.csv'
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

def get_mean_predict(df):
    
    mean_predict=df.results.mean()
    mean_real=df.Dffclt.mean()
    
    return {
        "mean_predict":mean_predict,
        "mean_real":mean_real
    }

def get_accuracy_predict(df):
    accuracy=df.acierta.mean()
    return accuracy
    
base_path= "/home/dcast/adversarial_project/openml/data/results_experiment_4"
model_type="classifier" #classifier/regressor
adversarial="with" #with/without
#
files=get_list_with_all_csv_with_format(base_path,model_type,adversarial)
df=generate_dataframe_concated(files)
# print(files)
print(df.head())

if model_type=="classifier":
    metric=get_accuracy_predict(df)
elif model_type=="regressor":
    metric=get_mean_predict(df)
else:
    raise("model type erroneo")

print(metric)