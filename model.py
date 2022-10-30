from torch import nn

class ExtractLSTM(nn.Module):

    def forward(self, x):
        out, _ = x
        return out[:, -1, :]

class MyLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_class):
        super().__init__()

        self.layers = nn.Sequential(
            nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=5,
                batch_first=True,
                dropout=0.2,
                bidirectional=True
            ),
            ExtractLSTM(),
            nn.Linear(hidden_size * 2, num_class)
        )
    def forward(self, x):
        return self.layers(x)
