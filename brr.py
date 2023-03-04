import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import sys
import pickle
#import loading bar
from tqdm import trange


# Define the dataset
class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1102, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(0.2)
        self.layernorm1 = nn.LayerNorm(normalized_shape=64)
        self.layernorm2 = nn.LayerNorm(normalized_shape=32)
        #layernorm
        # self.batchnorm = nn.BatchNorm1d(num_features=32)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # swap axes to (batch_size, input_size, sequence_length)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.layernorm1(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.layernorm2(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.dropout(x)
        # print(x.shape)
        # x = self.layernorm(x)
        return x



class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, dropout_prob):
        super(Decoder, self).__init__()

        self.masked_attention = MultiHeadAttention(input_size, hidden_size, num_heads, dropout_prob)
        self.attention_norm = nn.LayerNorm(input_size)

        self.attention = MultiHeadAttention(input_size, hidden_size, num_heads, dropout_prob)
        self.encoder_output_norm = nn.LayerNorm(input_size)

        self.feedforward = nn.Sequential(
            nn.Linear(input_size, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, input_size),
            nn.Dropout(dropout_prob)
        )
        self.feedforward_norm = nn.LayerNorm(input_size)

    def forward(self, decoder_input, encoder_output, encoder_output_mask, decoder_mask):
        # Multiheaded Masked Attention
        attention_output, _ = self.masked_attention(
            decoder_input,
            decoder_input,
            decoder_input,
            decoder_mask
        )

        # Residual Connection and Layer Normalization
        attention_output = self.attention_norm(decoder_input + attention_output)

        # Multiheaded Attention over Encoder Output
        attention_output, attention_weights = self.attention(
            attention_output,
            encoder_output,
            encoder_output,
            encoder_output_mask
        )

        # Residual Connection and Layer Normalization
        attention_output = self.encoder_output_norm(attention_output + decoder_input)

        # Feedforward
        feedforward_output = self.feedforward(attention_output)

        # Residual Connection and Layer Normalization
        feedforward_output = self.feedforward_norm(attention_output + feedforward_output)

        return feedforward_output, attention_weights


# Define the train function
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    train_loss = 0

    for X, y in train_loader:
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    return train_loss / len(train_loader)

def main():

    # Load the data from the .pkl file
    with open('./new_tensor_pkls/game_1.pkl', 'rb') as f:
        data = pickle.load(f)

    #IDS DONT MATTER ANYMORE
    #for the first 100 dim
    X = torch.unsqueeze(data[:300, :-60], 0)
    # X = torch.unsqueeze(data[:, :-60], 0)
    y = torch.unsqueeze(data[:300, -60:-30], 0)
    print(X.shape)
    print(y.shape)


    # Define the hyperparameters
    batch_size = 10
    num_epochs = 100000
    learning_rate = 0.001

    # Define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = MyDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize the model
    model = Seq2Seq().to(device)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train
    for epoch in trange(num_epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}')


if  __name__ == '__main__':
    main()

