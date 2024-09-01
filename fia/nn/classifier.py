from torch import nn


class FraudDetectionModel(nn.Module):
    def __init__(self, input_dim : int):
        super(FraudDetectionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)  # Batch normalization after first layer
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)  # Batch normalization after second layer
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  # Dropout with 50% probability

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)  # Apply dropout after activation
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)  # Apply dropout after activation
        x = self.fc3(x)  # No activation function here (output is logits)
        return x
