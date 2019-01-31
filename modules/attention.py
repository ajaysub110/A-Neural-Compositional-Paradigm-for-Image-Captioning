import torch 
import torch.nn as nn 
import torch.functional as F 
from torch.autograd import Variable

class Attention(nn.Module):
    def __init__(self,sequence_length,embedding_dim):
        super(Attention,self).__init__()

        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        
        self.fc = nn.Sequential(
            nn.Linear(self.sequence_length*self.embedding_dim,3072),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(3072,self.embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(self.embedding_dim,self.sequence_length),
            nn.ReLU(inplace=True)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self,lstm_h_t):
        # input: (batch_size,seq_len,hidden_size)
        batch_size = lstm_h_t.shape[0]
        lstm_h_t_flattened = lstm_h_t.view(batch_size,-1) # (batch_size,seq_len*hidden_size)

        alpha = self.softmax(self.fc(lstm_h_t_flattened)) # (batch_size,seq_len)
        alpha = torch.stack([alpha]*self.embedding_dim,dim=2) # (batch_size,seq_len,embedding_dim)

        attn_feature_map = alpha # (batch_size,seq_len,embedding_dim)
        attn_feature_map = torch.sum(attn_feature_map,dim=1,keepdim=True) # (batch_size,1,embedding_dim)
        return attn_feature_map

if __name__ == '__main__':
    net = Attention(18,300)

    lstm_h_t = Variable(torch.Tensor(4,18,300))
    out = net(lstm_h_t)
    print(out.size())