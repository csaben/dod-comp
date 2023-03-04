import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import sys
import pickle


# Define the dataset
class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Define the model
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=100, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(0.2)
        self.batchnorm = nn.BatchNorm1d(num_features=32)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.batchnorm(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=32, num_heads=4)
        self.dropout = nn.Dropout(0.2)
        self.batchnorm = nn.BatchNorm1d(num_features=32)
        self.fc = nn.Linear(in_features=32, out_features=30)
        
    def forward(self, x):
        x, _ = self.attention(x, x, x)
        x = self.dropout(x)
        x = self.batchnorm(x)
        x = self.fc(x)
        return x

class Seq2Seq(nn.Module):
    def __init__(self):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

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
    X = torch.unsqueeze(data[:100, :-60], 0)
    # X = torch.unsqueeze(data[:, :-60], 0)
    y = torch.unsqueeze(data[:, -60:-30], 0)

    # Define the hyperparameters
    batch_size = 10
    num_epochs = 100
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
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}')

if  __name__ == '__main__':
    main()

