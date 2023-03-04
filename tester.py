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
        # self.layernorm1 = nn.LayerNorm(normalized_shape=64)
        # self.layernorm2 = nn.LayerNorm(normalized_shape=32)
        #layernorm
        self.batchnorm = nn.BatchNorm1d(num_features=64)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # swap axes to (batch_size, input_size, sequence_length)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # x = self.layernorm1(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.relu(self.conv3(x))
        x = x.permute(0, 1, 2)
        x = self.batchnorm(x)
        x = x.permute(0, 1, 2)
        x = F.relu(self.conv4(x))
        # x = self.layernorm2(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.dropout(x)
        # print(x.shape)
        # x = self.layernorm(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=32, num_heads=4)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.2)
        self.dropout4 = nn.Dropout(0.2)
        self.dropout5 = nn.Dropout(0.2)
        self.layernorm = nn.LayerNorm(normalized_shape=32)
        # self.batchnorm = nn.BatchNorm1d(num_features=32)
        self.fc1 = nn.Linear(in_features=32, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=32)
        self.fc4 = nn.Linear(in_features=32, out_features=32)
        self.fc5 = nn.Linear(in_features=32, out_features=30)
        self.mha = nn.MultiheadAttention(embed_dim=32, num_heads=4)

    def forward(self, x):
        x, _ = self.mha(x, x, x)
        #residual connection
        x = x + F.relu(x)
        # x, _ = self.attention(x, x, x)
        x = self.dropout1(x)
        # x = self.batchnorm(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.layernorm(x)
        x = self.dropout3(x)
        x = F.relu(self.fc3(x))
        x = self.dropout4(x)
        x = F.relu(self.fc4(x))
        x = self.dropout5(x)
        x = self.fc5(x)
        return x

class Seq2Seq(nn.Module):
    def __init__(self):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = x.permute(0, 2, 1)  # swap axes back to (batch_size, sequence_length, input_size)
        # print(x.shape)
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
    X = torch.unsqueeze(data[:300, :-60], 0)
    # X = torch.unsqueeze(data[:, :-60], 0)
    y = torch.unsqueeze(data[:300, -60:-30], 0)
    print(X.shape)
    print(y.shape)


    # Define the hyperparameters
    batch_size = 10
    num_epochs = 50000
    learning_rate = 0.001
    best_val_loss = float('inf')

    # Define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = MyDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize the model
    model = Seq2Seq().to(device)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    # Train
    for epoch in trange(num_epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        if train_loss < best_val_loss:
            best_val_loss = train_loss
            torch.save(model.state_dict(), './sequence_model/model_1.pt')
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}')


if  __name__ == '__main__':
    main()

