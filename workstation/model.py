# model.py (updated with dropout for regularization to prevent overfitting)
import torch
import torch.nn as nn
import torch.nn.functional as F

class DrivingCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 24, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2)

        self.conv4 = nn.Conv2d(48, 64, kernel_size=3, stride=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # 🔑 DYNAMIC FEATURE SIZE COMPUTATION
        self._feature_dim = self._get_conv_output_dim()

        self.fc1 = nn.Linear(self._feature_dim, 100)
        self.dropout1 = nn.Dropout(0.5)  # Added dropout
        self.fc2 = nn.Linear(100, 50)
        self.dropout2 = nn.Dropout(0.5)  # Added dropout
        self.fc_out = nn.Linear(50, 2)  # steering, throttle

    def _get_conv_output_dim(self):
        with torch.no_grad():
            x = torch.zeros(1, 3, 224, 224)
            x = self._forward_conv(x)
            return x.view(1, -1).size(1)

    def _forward_conv(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)  # Apply dropout
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)  # Apply dropout
        return self.fc_out(x)
