#use a CNN as the encoder and a decoder block as the decoder


#kinda like this for the encoder
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.cnn = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        )
        self.rnn = nn.GRU(hidden_size, hidden_size, num_layers, dropout=dropout, batch_first=True)

    def forward(self, input):
        embedded = self.embedding(input)
        embedded = embedded.transpose(1,2)
        cnn_out = self.cnn(embedded)
        cnn_out = cnn_out.transpose(1,2)
        output, hidden

