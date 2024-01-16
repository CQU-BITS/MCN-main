
import torch
from torch import nn


class WDCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(WDCNN, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=64, stride=16, padding=24),
            nn.BatchNorm1d(16),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 100),
            nn.ReLU(),
            nn.Linear(100, num_classes)
        )

    def forward(self, x):
        out = self.feature(x)
        out = self.fc(out)
        return out


if __name__ == '__main__':
    device = torch.device('cuda:0')
    temp = torch.randn([32, 1, 1024]).to(device)
    model = WDCNN(in_channels=1, num_classes=5).to(device)
    out = model(temp)
    print(out.shape)


