import torch
import torch.nn as nn 
from torch.autograd import Variable

from VarLSTM import VarLSTM
from attention import Attention

class ConnectingModule(nn.Module):
    def __init__(self,no_words=2627,lstm_size=1024,emb_size=300,depth=1):
        super(ConnectingModule,self).__init__()

        self.word_count = no_words
        self.lstm_size = lstm_size
        self.emb_size = emb_size
        self.depth = depth 

        self.embedding = nn.Embedding(self.word_count,self.emb_size)
        self.lstm1 = VarLSTM(num_hidden=self.lstm_size,depth=self.depth,word_emb_dim=self.emb_size)
        self.attention = Attention(embedding_dim=self.lstm_size)
        self.lstm2 = VarLSTM(num_hidden=self.lstm_size,depth=self.depth,word_emb_dim=self.lstm_size)

    def forward(self,tokens):
        embedded = self.embedding(tokens)
        lstm1_emb, a1 = self.lstm1(embedded)
        attn_lstm_emb = self.attention(lstm1_emb)
        encoded, a2 = self.lstm2(attn_lstm_emb)

        return encoded

if __name__ == '__main__':
    model = ConnectingModule(no_words=16)
    print(model)
    seq = Variable(torch.Tensor([[0,1,2,3,4,5,6,7],[8,9,10,11,12,13,14,15]])).long()
    encoded = model(seq)
    print(encoded.size())