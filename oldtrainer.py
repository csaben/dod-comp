import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from network import Encoder, Decoder, Seq2Seq
import sys

# Hyperparameters
input_size = 1162
output_size = 30
hidden_size = 256
num_layers = 2
learning_rate = 0.001
batch_size = 32
num_epochs = 10
teacher_forcing_ratio = 0.5

# Load data
import pickle

# Load the data from the .pkl file
with open('./new_tensor_pkls/game_1.pkl', 'rb') as f:
    data = pickle.load(f)

#IDS DONT MATTER ANYMORE
X= data[:, :-60]
labels = data[:, -60:-30]
print(X.shape)
print(labels.shape)

# CONFUSING BECAUSE I EXPECT TO ALWAYS HAVE 60 LABELS at end of tensor slice but I only have 30
# labels = data[124, :]
# times_only = labels[::2]
# print(len(labels))
# print(labels)
# print(len(times_only))
# print(times_only)
sys.exit()
#pretend you parsed out your labels st x.shape=(100,1102) and y.shape=(100,62)
# Split the data into training and validation sets
"""REPLACE PSEUDOCODE WITH YOUR CODE"""
"""

you have to split the last 30 elements of the (100,1162) tensor to get you X and y

the X data is the (100,1102) tensor
the y data is the (100,32) tensor

then you have to setup the dataloader to load the data in batches of 32

and handle the validation sets

"""

# Define the model, loss function, optimizer, and device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

encoder = Encoder(input_size, hidden_size, num_layers=num_layers)
decoder = Decoder(hidden_size, output_size, num_layers=num_layers)
model = Seq2Seq(encoder, decoder).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Train the model
best_val_loss = float('inf')
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        loss = train(model, optimizer, criterion, inputs, labels, teacher_forcing_ratio)

        running_loss += loss

    epoch_loss = running_loss / len(train_loader)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

    # Evaluate the model on the validation data
    with torch.no_grad():
        val_loss = 0.0
        for i, (inputs, labels) in enumerate(val_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs, labels[:, :-1], False)

            outputs = outputs.view(-1, output_size)
            labels = labels[:, 1:].contiguous().view(-1)

            loss = criterion(outputs, labels.float())
            val_loss += loss.item()

        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'model.pt')
