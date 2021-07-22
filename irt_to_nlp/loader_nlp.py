


from torch.utils.data import Dataset
import pandas as pd
import torch
from transformers import GPT2Tokenizer,GPTNeoConfig,AutoTokenizer
from irt_to_nlp.config import ModelAvailable
class Loader(Dataset):
    
    def __init__(self,dir_csv_file:str,model:ModelAvailable) -> None:
        super().__init__()
        
        self.dir_csv_file=dir_csv_file
        self.data=pd.read_csv(self.dir_csv_file,index_col="Unnamed: 0")
        self.y=self.data.pop("Dffclt").to_numpy()
        self.X=list(self.data["sentence"])
        self.model=model
        self.create_tokenizer()


    def create_tokenizer(self):
        
        if self.model==ModelAvailable.customneogpt:
            self.tokenizer=GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
            # config=GPTNeoConfig()
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.X=self.tokenizer(self.X,return_tensors="pt", truncations=True,padding=True).input_ids
            
        elif self.model==ModelAvailable.bert_base_cased:
            self.tokenizer=AutoTokenizer.from_pretrained("bert-base-cased")
            self.X=self.tokenizer(self.X,return_tensors="pt", padding="max_length",truncation=True).input_ids
            
        elif self.model==ModelAvailable.distilbert_base_uncased:
            self.tokenizer=AutoTokenizer.from_pretrained('distilbert-base-uncased')
            self.X=self.tokenizer(self.X,return_tensors="pt", padding="max_length",truncation=True).input_ids
            
        elif self.model==ModelAvailable.distilgpt2:
            self.tokenizer=AutoTokenizer.from_pretrained('distilgpt2')
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.X=self.tokenizer(self.X,return_tensors="pt", truncation=True,padding="max_length").input_ids
        
            
    def __getitem__(self, index):
        # txt=self.data.iloc[index]
        # prompt=txt.values
        
        # input_ids=self.encodings[index]
        # input_ids=self.tokenizer(prompt[0],return_tensors="pt").input_ids#,truncation=True,padding=True).input_ids
        input_ids=self.X[index]
        target=torch.tensor(self.y[index],dtype=torch.float)
        target=torch.unsqueeze(target,0)
        #pendiente aplicar transform simple a example
        return input_ids,target,index
    
    def __len__(self):
        
        return self.data.shape[0]
 
        
        
        


