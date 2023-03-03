import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

    def forward(self, x):
        # x: (batch_size, sequence_length, input_size)
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(x.device)  # initial hidden state
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(x.device)  # initial cell state
        out, (h, c) = self.lstm(x, (h0, c0))  # out: (batch_size, sequence_length, hidden_size*2), h and c are the final hidden and cell states
        return out, (h, c)

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size*2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: (num_layers*2, batch_size, hidden_size)
        # encoder_outputs: (batch_size, sequence_length, hidden_size*2)
        batch_size = encoder_outputs.size(0)
        sequence_length = encoder_outputs.size(1)

        # Repeat the hidden state to match the shape of encoder_outputs
        hidden = hidden.permute(1, 0, 2).repeat(1, sequence_length, 1)
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], dim=2)))  # (batch_size, sequence_length, hidden_size)
        attention = torch.softmax(self.v(energy), dim=1)  # (batch_size, sequence_length, 1)
        context = attention * encoder_outputs  # (batch_size, sequence_length, hidden_size*2)
        context = context.sum(dim=1, keepdim=True)  # (batch_size, 1, hidden_size*2)
        return context, attention.squeeze(2)

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size*2, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.attention = Attention(hidden_size)

    def forward(self, x, hidden, encoder_outputs):
        # x: (batch_size, 1)
        # hidden: (num_layers, batch_size, hidden_size)
        # encoder_outputs: (batch_size, sequence_length, hidden_size*2)
        embedded = self.embedding(x)  # (batch_size, 1, hidden_size)
        context, attention = self.attention(hidden, encoder_outputs)
        rnn_input = torch.cat([embedded, context], dim=2)  # (batch_size, 1, hidden_size*2)
        output, (hidden, cell) = self.lstm(rnn_input, (hidden, cell))  # output: (batch_size, 1, hidden_size), hidden: (num_layers, batch_size, hidden_size)
        output = output.squeeze(1)  # (batch_size, hidden_size)
        output = self.fc(output)  # (batch_size, output_size)
        return output, hidden, cell, attention

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, y):
        # x: (batch_size, sequence_length, input_size)
        # y: (batch_size, sequence_length)
        batch_size = x.size(0)
        sequence_length = y.size(1)
        output_size = self.decoder.output_size

        encoder_outputs, (hidden, cell) = self.encoder(x)

        # Create an empty tensor to store the decoder outputs
        outputs = torch.zeros(batch_size, sequence_length, output_size).to(x.device)

        # Use the first time step of the decoder's input as the SOS token
        input = y[:, 0].unsqueeze(1)

        for t in range(1, sequence_length):
            output, hidden, cell, attention = self.decoder(input, hidden, encoder_outputs)
            outputs[:, t] = output
            top1 = output.argmax(1)
            input = y[:, t].unsqueeze(1) if random.random() < teacher_forcing_ratio else top1.unsqueeze(1)

        return outputs


