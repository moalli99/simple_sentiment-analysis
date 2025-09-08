import torch
from torch import nn,optim

class SentimentGRU(nn.Module):
    def __init__(self,n_layers,vocab_size,hidden_dim,embedding_dim,output_dim,drop_prob):
        super(SentimentGRU,self).__init__()
        self.hidden_dim=hidden_dim
        self.n_layers=n_layers
        self.embedd=nn.Embedding(vocab_size,embedding_dim)
        self.gru=nn.GRU(embedding_dim,hidden_size=hidden_dim,num_layers=n_layers,batch_first=True)
        self.dropout=nn.Dropout(drop_prob)
        self.fc=nn.Linear(hidden_dim,output_dim)
        self.sigmoid=nn.Sigmoid()
    def forward(self,x,hidden):
        batch_size=x.size(0)
        embedds=self.embedd(x)
        gru_out,hidden=self.gru(embedds,hidden)
        out=gru_out.contiguous().view(-1,self.hidden_dim)
        out=self.dropout(out)
        out=self.fc(out)
        out=self.sigmoid(out)
        out=out.view(batch_size,-1)
        out=out[:,-1]
        return out,hidden
    def init_hidden(self,barch_size,device):
        h0=torch.zeros(self.n_layers,barch_size,self.hidden_dim).to(device)
        return h0
    



    

