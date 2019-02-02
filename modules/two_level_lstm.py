import torch
import torch.nn as nn 
from torch.autograd import Variable

from VarLSTM import VarLSTM
from attention import Attention

class TwoLevelLSTM(nn.Module):
    """
    Parameters:
    sequence_length: Length of each sequence
    word_count: Embedding dictionary size i.e vocabulary size
    lstm_hidden_size: number of features in lstm hidden state
    embedding_dim: Pre-LSTM embedding size
    lstm_num_layers: number of recurrent layers
    """
    def __init__(self,word_count,sequence_length,lstm_hidden_size,embedding_dim,lstm_num_layers):
        super(TwoLevelLSTM,self).__init__()

        self.word_count = word_count
        self.sequence_length = sequence_length
        self.lstm_hidden_size = lstm_hidden_size
        self.embedding_dim = embedding_dim
        self.lstm_num_layers = lstm_num_layers

        self.embedding = nn.Embedding(self.word_count,self.embedding_dim)
        self.lstm1 = VarLSTM(self.embedding_dim,self.lstm_hidden_size,self.lstm_num_layers)
        self.attention = Attention(self.sequence_length,self.lstm_hidden_size)
        self.lstm2 = VarLSTM(self.lstm_hidden_size,self.lstm_hidden_size,self.lstm_num_layers)

    def forward(self,tokens):
        # Input: (batch_size,seq_len)
        embedded = self.embedding(tokens) # (batch_size,seq_len,embedding_dim)
        lstm1_h_t = self.lstm1(embedded) # (batch_size,seq_len,lstm_hidden_size)
        attn_feature_map = self.attention(lstm1_h_t) # (batch_size,1,lstm_hidden_size)
        attn_feature_map = torch.cat([attn_feature_map]*self.sequence_length,dim=1) # ((batch_size,seq_len,lstm_hidden_size))
        encoded = self.lstm2(attn_feature_map) # (batch_size,seq_len,lstm_hidden_size)

        return encoded

if __name__ == '__main__':
    model = TwoLevelLSTM(10,3,300,1024,1)
    seq = Variable(torch.Tensor([[0,5,8],[2,3,7]])).long()
    encoded = model(seq)
    print(encoded.size())