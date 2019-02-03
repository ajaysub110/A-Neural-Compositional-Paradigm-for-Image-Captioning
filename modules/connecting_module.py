import torch 
import torch.nn as nn 

from two_level_lstm import TwoLevelLSTM

class ConnectingModule(nn.Module):
    def __init__(self,word_count,sequence_length,lstm_hidden_size,embedding_dim,lstm_num_layers,num_classes,fc1_dim):
        super(ConnectingModule,self).__init__()

        self.word_count = word_count
        self.sequence_length = sequence_length
        self.lstm_hidden_size = lstm_hidden_size
        self.embedding_dim = embedding_dim
        self.lstm_num_layers = lstm_num_layers
        self.num_classes = num_classes
        self.fc1_dim = fc1_dim

        self.embedding = nn.Embedding(self.word_count,self.embedding_dim)
        self.two_level_lstm_l = TwoLevelLSTM(self.sequence_length,self.lstm_hidden_size,
            self.embedding_dim,self.lstm_num_layers,self.fc1_dim)
        self.two_level_lstm_r = TwoLevelLSTM(self.sequence_length,self.lstm_hidden_size,
            self.embedding_dim,self.lstm_num_layers,self.fc1_dim)
        self.fc2 = nn.Linear(in_features=self.fc1_dim,out_features=self.num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self,P_l,P_r):
        P_l_encoded = self.two_level_lstm_l(self.embedding(P_l)) # (batch_size,fc1_dim)
        P_r_encoded = self.two_level_lstm_r(self.embedding(P_r)) # (batch_size,fc1_dim)
        combined = self.fc2(torch.add(P_l_encoded,P_r_encoded)) # (batch_size,fc2_dim)
        prediction = self.softmax(combined) # (batch_size,fc2_dim)

        return prediction

if __name__ == '__main__':
    model = ConnectingModule(10,3,300,1024,1,1000,1000)
    P_l = torch.Tensor([[0,1,2],[7,2,8]]).long()
    P_r = torch.Tensor([[0,1,8],[5,2,4]]).long()
    out = model(P_l,P_r)
    print(out.size())