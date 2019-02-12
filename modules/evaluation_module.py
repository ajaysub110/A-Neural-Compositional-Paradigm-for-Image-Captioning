import torch 
import torch.nn as nn 

from two_level_lstm import TwoLevelLSTM

class EvaluationModule(nn.Module):
    def __init__(self,word_count,sequence_length,lstm_hidden_size,embedding_dim,lstm_num_layers,fc_dim):
        super(EvaluationModule,self).__init__()

        self.word_count = word_count
        self.sequence_length = sequence_length
        self.lstm_hidden_size = lstm_hidden_size
        self.embedding_dim = embedding_dim
        self.lstm_num_layers = lstm_num_layers
        self.fc_dim = fc_dim
        
        self.embedding = nn.Embedding(self.word_count,self.embedding_dim)
        self.two_level_lstm = TwoLevelLSTM(self.sequence_length,self.lstm_hidden_size,
            self.embedding_dim, self.lstm_num_layers,fc_dim)
        self.fc = nn.Linear(in_features=self.fc_dim,out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,P):
        encoded = self.two_level_lstm(self.embedding(P)) # (batch_size,fc1_dim)
        evaluation = self.sigmoid(self.fc(encoded)) # (batch_size,1)

        return evaluation

if __name__ == '__main__':
    model = EvaluationModule(10,3,300,1024,1,300)
    P = torch.Tensor([[0,1,2],[7,2,8]]).long()
    out = model(P)
    print(out)