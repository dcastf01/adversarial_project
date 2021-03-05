from glob import glob
import pandas as pd
def concat_csv_files(folder_with_csv="/content/csvs_IRT"):
  # print(folder_with_csv)
  df_aux=[]
  for csv in glob(folder_with_csv+"/*.csv"):
    df = pd.read_csv(csv, index_col=None, header=0)
    df.rename(columns={'Unnamed: 0': 'filename'},inplace=True)
    df["pert"]=csv.split(".")[-3]
    df_aux.append(df)

  df = pd.concat(df_aux, axis=0, ignore_index=True)
  return df