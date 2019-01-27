import torch 
import torch.nn as nn 
import torch.functional as F 
from torch.autograd import Variable

class Attention(nn.Module):
    def __init__(self,seq_len=8,embedding_dim=1024):
        super(Attention,self).__init__()

        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        
        self.fc = nn.Sequential(
            nn.Linear(self.seq_len*self.embedding_dim,3072),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(3072,1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024,self.seq_len),
            nn.ReLU(inplace=True)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self,lstm_emb):
        batch_size = lstm_emb.shape[0]
        lstm_emb = lstm_emb.contiguous()
        lstm_flattened = lstm_emb.view(batch_size,-1)

        alpha = self.softmax(self.fc(lstm_flattened))
        alpha = torch.stack([alpha]*1024,dim=2)

        attn_feature_map = lstm_emb*alpha 
        attn_feature_map = torch.sum(attn_feature_map,dim=1,keepdim=True)
        return attn_feature_map

if __name__ == '__main__':
    net = Attention()

    lstm_emb = Variable(torch.Tensor(4,18,1024))
    out = net(lstm_emb)