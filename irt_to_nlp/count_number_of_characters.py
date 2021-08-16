import pandas as pd



# csv_path="/home/dcast/adversarial_project/irt_to_nlp/data/IMDB.Diff6.RefClass.csv"
# csv_path="/home/dcast/adversarial_project/irt_to_nlp/data/SST.Diff6.RefClass.csv"

df=pd.read_csv(csv_path,index_col="Unnamed: 0")

df['NAME_Count'] = df['sentence'].str.len()
print(df.head(5))

print(df.describe())