from transformers import BertTokenizer
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
dir_csv_file="/home/dcast/adversarial_project/irt_to_nlp/data/IMDB.Diff6.RefClass.csv"
data=pd.read_csv(dir_csv_file,index_col="Unnamed: 0")
PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

token_lens=[]

for txt in data.sentence:
    tokens=tokenizer.encode(txt,max_length=1024)
    token_lens.append(len(tokens))
    
sns.distplot(token_lens)
# plt.xlim([0, 1024])
plt.xlabel('Token count')

plt.savefig("study_number_token_IMBD.jpg")