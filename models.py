import torch.nn as nn


class SimpleRegressionNN(nn.Module):
    def __init__(self, input_size, hidden_size=64, dropout_rate=0.2):
        super(SimpleRegressionNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        return self.layers(x).squeeze()
