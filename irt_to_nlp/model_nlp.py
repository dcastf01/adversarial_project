

from transformers import GPTNeoForCausalLM, GPT2Tokenizer,GPTNeoModel
import torch.nn as nn

class CustomGPTNeo(nn.Module):
    
    def __init__(self):
        super(CustomGPTNeo, self).__init__()
        in_dim=50257
        hidden_dim1=2048
        num_tokens=45
        dim_flat=hidden_dim1*num_tokens
        hidden_dim2=500
        out_dim=1
        
        self.gptneo = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim1),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(hidden_dim2, out_dim)
        )

        
    def forward(self,ids):
        x=self.gptneo(ids)
        x=x.logits[::,-1,::]
        x=self.layer1(x)
        
        x=self.layer2(x)
        
        return x
    

model = CustomGPTNeo()

# model = GPTNeoModel.from_pretrained('EleutherAI/gpt-neo-1.3B')
prompt = "In a shocking finding, scientists discovered a herd of unicorns living in a remote, " \
          "previously unexplored valley, in the Andes Mountains. Even more surprising to the " \
          "researchers was the fact that the unicorns spoke perfect English."
tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-1.3B')
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
output=model(input_ids)
# gen_tokens = model.generate(input_ids, do_sample=True, temperature=0.9, max_length=100,output_hidden_states =True)
# b=tokenizer.batch_decode(gen_tokens)
# gen_text = tokenizer.batch_decode(gen_tokens)[0]

# print(gen_text)
print(output)