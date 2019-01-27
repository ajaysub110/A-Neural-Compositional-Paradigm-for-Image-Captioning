import torch 
import torch.nn as nn

class VarLSTM(nn.Module):
    def __init__(self,num_hidden,depth=1,word_emb_dim=300):
        super(VarLSTM,self).__init__()

        self.num_hidden = num_hidden
        self.depth = depth 
        self.word_emb_dim = word_emb_dim

        self.lstm = nn.LSTM(input_size=self.word_emb_dim,
            hidden_size=self.num_hidden,num_layers=self.depth,batch_first=True,dropout=0)
        
    def forward(self,word_emb):
        out = self.lstm(word_emb)
        return out
