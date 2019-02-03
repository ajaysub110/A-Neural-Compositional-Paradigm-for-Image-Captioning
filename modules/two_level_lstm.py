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
    def __init__(self,sequence_length,lstm_hidden_size,embedding_dim,lstm_num_layers,fc1_dim):
        super(TwoLevelLSTM,self).__init__()

        self.sequence_length = sequence_length
        self.lstm_hidden_size = lstm_hidden_size
        self.embedding_dim = embedding_dim
        self.lstm_num_layers = lstm_num_layers
        self.fc1_dim = fc1_dim

        self.lstm1 = VarLSTM(self.embedding_dim,self.lstm_hidden_size,self.lstm_num_layers)
        self.attention = Attention(self.sequence_length,self.lstm_hidden_size)
        self.lstm2 = VarLSTM(self.lstm_hidden_size,self.lstm_hidden_size,self.lstm_num_layers)
        self.fc1 = nn.Linear(self.sequence_length*self.lstm_hidden_size,self.fc1_dim)

    def forward(self,embedded):
        # Input: (batch_size,seq_len,embedding_dim)
        batch_size = embedded.size()[0]
        lstm1_h_t = self.lstm1(embedded) # (batch_size,seq_len,lstm_hidden_size)
        attn_feature_map = self.attention(lstm1_h_t) # (batch_size,1,lstm_hidden_size)
        attn_feature_map = torch.cat([attn_feature_map]*self.sequence_length,dim=1) # ((batch_size,seq_len,lstm_hidden_size))
        encoded = self.lstm2(attn_feature_map) # (batch_size,seq_len,lstm_hidden_size)
        encoded = encoded.contiguous().view(batch_size,-1) # (batch_size, seq_len*lstm_hidden_size)
        encoded = self.fc1(encoded) # (batch_size,fc1_dim)

        return encoded