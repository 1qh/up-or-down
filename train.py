import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset
device = 'cuda' if torch.cuda.is_available else 'cpu'

import pandas as pd
import numpy as np

down = pd.read_csv('hands_down.csv').iloc[:, 1:].values
up = pd.read_csv('hands_up.csv').iloc[:, 1:].values

def sample_data(arr, num_timestep):
    return [arr[i - num_timestep : i, :] for i in range(num_timestep, len(arr))]

X_down = sample_data(down, 10)
y_down = [0] * len(X_down)

X_up = sample_data(up, 10)
y_up = [1] * len(X_up)

X = torch.as_tensor(np.array(X_down + X_up), dtype=torch.float32, device=device)
y = torch.as_tensor(np.array(y_down + y_up), dtype=torch.long, device=device)

batch_size = 32

data = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)

from model import MyLSTM
model1 = MyLSTM(
    input_size=X.shape[-1],
    hidden_size=20,
    num_class=2
).to(device)

epochs = 100
learning_rate = 0.001
loss = nn.CrossEntropyLoss()
opti = optim.Adam(model1.parameters(), lr=learning_rate)

model1.train()

for e in range(epochs):
    
    running_loss = 0.0

    for batch, (X, y) in enumerate(data):
        
        opti.zero_grad()
        y_hat = model1(X)
        l = loss(y_hat, y)
        l.backward()
        opti.step()

        running_loss += l.item()

        if batch % 10 == 0:
            print(f'[{e}, {batch}] loss: {running_loss / 10}')
            running_loss = 0.0

torch.save(model1.state_dict(), 'best.pt')