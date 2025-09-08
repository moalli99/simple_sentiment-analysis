import torch
import torch.nn as nn

class SentimentLSTM_Word2Vec(nn.Module):
    def __init__(self,embedding_matrix,hiddin_dim,n_layers,output_dim,drop_prob):
        super(SentimentLSTM_Word2Vec,self).__init__()
        vocab_size,embedding_dim=embedding_matrix.shape
        self.n_layers=n_layers
        self.hidden_dim=hiddin_dim
        self.embedding=nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix),freeze=False)
        self.lstm=nn.LSTM(embedding_dim,hidden_size=hiddin_dim,num_layers=n_layers,batch_first=True)
        self.dropout=nn.Dropout(drop_prob)
        self.fc=nn.Linear(hiddin_dim,output_dim)
        self.sigmoid=nn.Sigmoid()
    def forward(self,x,hidden):
        batch_size=x.size(0)
        embedds=self.embedding(x)
        lstm_out,hidden=self.lstm(embedds,hidden)
        lstm_out=lstm_out.contiguous().view(-1,self.hidden_dim)
        out=self.dropout(lstm_out)
        out=self.fc(out)
        out=self.sigmoid(out)
        out=out.view(batch_size,-1)
        out=out[:,-1]
        return out,hidden
    def init_hidden(self,batch_size,device):
        h0=torch.zeros(self.n_layers,batch_size,self.hidden_dim).to(device)
        c0=torch.zeros(self.n_layers,batch_size,self.hidden_dim).to(device)
        return (h0,c0)
        