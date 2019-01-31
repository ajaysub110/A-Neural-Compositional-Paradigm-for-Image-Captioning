import torch 
import torch.nn as nn

class VarLSTM(nn.Module):
    """
    Parameters:
    input_size: length of input sequence i.e. number of features
    hidden_size: number of features in hidden state h
    num_layers: number of stacked LSTM units
    """
    def __init__(self,input_size,hidden_size,num_layers):
        super(VarLSTM,self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=self.input_size,hidden_size=self.hidden_size,
            num_layers=self.num_layers,batch_first=True)
        
    def forward(self,embedded_sequence):
        # input: (batch_size,seq_len,hidden_size)
        # Note: Here hidden_size is equal to the embedding dimension of output
        h_t = self.lstm(embedded_sequence) # (batch_size,seq_len,hidden_size)
        return h_t
