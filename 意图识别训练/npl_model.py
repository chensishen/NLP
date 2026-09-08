import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_padded_sequence

class Network_RNN(nn.Module):
    def __init__(self,vocab_size,embed_dim,hidden_size,num_classes):
        super(Network_RNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size,embed_dim,padding_idx=0)
        self.rnn = nn.RNN(embed_dim,hidden_size,batch_first=True)
        self.fc = nn.Linear(hidden_size,num_classes)
    def forward(self,x,true_len):
        embedded = self.embedding(x)
        packed = pack_padded_sequence(embedded,true_len,batch_first=True,enforce_sorted=False)
        out, h_n = self.rnn(packed)
        last_hidden = h_n.transpose(0, 1).reshape(len(x), -1)
        return self.fc(last_hidden)

class Network_RNN_Bidirection(nn.Module):
    def __init__(self,vocab_size,embed_dim,hidden_size,num_classes):
        super(Network_RNN_Bidirection, self).__init__()
        self.embedding = nn.Embedding(vocab_size,embed_dim,padding_idx=0)
        self.rnn = nn.RNN(embed_dim,hidden_size,batch_first=True,bidirectional=True)
        self.fc = nn.Linear(hidden_size*2,num_classes)
    def forward(self,x,true_len):
        embedded = self.embedding(x)
        packed = pack_padded_sequence(embedded,true_len,batch_first=True,enforce_sorted=False)
        out, h_n = self.rnn(packed)
        last_hidden = h_n[-2:].transpose(0, 1).reshape(len(x), -1)
        return self.fc(last_hidden)

class Network_LSTM(nn.Module):
    def __init__(self,vocab_size,embed_dim,hidden_size,num_classes):
        super(Network_LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size,embed_dim,padding_idx=0)
        self.lstm = nn.LSTM(embed_dim,hidden_size,batch_first=True)
        self.fc = nn.Linear(hidden_size,num_classes)
    def forward(self,x,true_len):
        embedded = self.embedding(x)
        packed = pack_padded_sequence(embedded,true_len,batch_first=True,enforce_sorted=False)
        out, (h_n, c_n) = self.lstm(packed)
        last_hidden = h_n.transpose(0, 1).reshape(len(x), -1)
        return self.fc(last_hidden)

class Network_LSTM_Bidirection(nn.Module):
    def __init__(self,vocab_size,embed_dim,hidden_size,num_classes):
        super(Network_LSTM_Bidirection, self).__init__()
        self.embedding = nn.Embedding(vocab_size,embed_dim,padding_idx=0)
        self.lstm = nn.LSTM(embed_dim,hidden_size,batch_first=True,bidirectional=True)
        self.fc = nn.Linear(hidden_size*2,num_classes)
    def forward(self,x,true_len):
        embedded = self.embedding(x)
        packed = pack_padded_sequence(embedded,true_len,batch_first=True,enforce_sorted=False)
        out, (h_n, c_n) = self.lstm(packed)
        last_hidden = h_n[-2:].transpose(0, 1).reshape(len(x), -1)
        return self.fc(last_hidden)

class Network_GRU(nn.Module):
    def __init__(self,vocab_size,embed_dim,hidden_size,num_classes):
        super(Network_GRU, self).__init__()
        self.embedding = nn.Embedding(vocab_size,embed_dim,padding_idx=0)
        self.gru = nn.GRU(embed_dim,hidden_size,batch_first=True)
        self.fc = nn.Linear(hidden_size,num_classes)
    def forward(self,x,true_len):
        embedded = self.embedding(x)
        packed = pack_padded_sequence(embedded,true_len,batch_first=True,enforce_sorted=False)
        out, h_n = self.gru(packed)
        last_hidden = h_n.transpose(0, 1).reshape(len(x), -1)
        return self.fc(last_hidden)

class Network_GRU_Bidirection(nn.Module):
    def __init__(self,vocab_size,embed_dim,hidden_size,num_classes):
        super(Network_GRU_Bidirection, self).__init__()
        self.embedding = nn.Embedding(vocab_size,embed_dim,padding_idx=0)
        self.gru = nn.GRU(embed_dim,hidden_size,batch_first=True,bidirectional=True)
        self.fc = nn.Linear(hidden_size*2,num_classes)
    def forward(self,x,true_len):
        embedded = self.embedding(x)
        packed = pack_padded_sequence(embedded,true_len,batch_first=True,enforce_sorted=False)
        out, h_n = self.gru(packed)
        last_hidden = h_n[-2:].transpose(0, 1).reshape(len(x), -1)
        return self.fc(last_hidden)

class MyDataset(Dataset):
    def __init__(self, data, target, true_lens):

        data = np.array(data).astype('int64')
        target = np.array(target).astype('int64')
        true_lens = np.array(true_lens).astype('int64')

        self.data = torch.from_numpy(data)
        self.target = torch.from_numpy(target)
        self.true_lens = torch.from_numpy(true_lens)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx],self.true_lens[idx]