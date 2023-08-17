import torch.nn as nn
import torch.nn.functional as F


class get_model(nn.Module):
    def __init__(self,outputs):
        super(get_model, self).__init__()

        self.fc1 = nn.Linear(2048, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(1024, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.drop3 = nn.Dropout(0.4)
        self.fc4 = nn.Linear(64, 16)
        self.bn4 = nn.BatchNorm1d(16)
        self.drop4 = nn.Dropout(0.4)
        self.fc5 = nn.Linear(16, outputs)

    def forward(self, x):
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.drop3(F.relu(self.bn3(self.fc3(x))))
        x = self.drop4(F.relu(self.bn4(self.fc4(x))))
        x = self.fc5(x)

        return x
