
import torch
from torch import nn


class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=5):
        super(CNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=3),  # 16, 26 ,26
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True))

        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3),  # 32, 24, 24
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2))  # 32, 12,12     (24-2) /2 +1

        self.layer3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3),  # 64,10,10
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True))

        self.layer4 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3),  # 128,8,8
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(2))  # 128, 2

        self.layer5 = nn.Sequential(
            nn.Linear(128 * 2, 256),
            # nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            # nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.layer5(x)
        x = self.fc(x)

        return x


if __name__ == '__main__':
    device = torch.device('cuda:0')
    temp = torch.randn([32, 1, 1024]).to(device)
    model = CNN(in_channels=1, num_classes=5).to(device)
    out = model(temp)
    print(out.shape)


