import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

class SentimentLSTM(nn.Module):
    """
    Sentiment analysis model using LSTM
    """
    def __init__(self,n_layers,vocab_size,hidden_dim,embedding_dim,output_dim,drop_prob):
        super(SentimentLSTM,self).__init__()
        self.output_dim=output_dim
        self.n_layers=n_layers
        self.hidden_dim=hidden_dim
        self.embedding=nn.Embedding(vocab_size,embedding_dim)
        self.lstm=nn.LSTM(embedding_dim,hidden_size=hidden_dim,num_layers=n_layers,batch_first=True)
        self.dropout=nn.Dropout(drop_prob)
        self.fc=nn.Linear(hidden_dim,output_dim)
        self.sigmoid=nn.Sigmoid()
    def forward(self,x,hidden):
        batch_size=x.size(0)
        embeds=self.embedding(x)
        lstm_out,hidden=self.lstm(embeds,hidden)
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





    

         

         

