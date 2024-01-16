
import torch
from torch import nn


class AlexNet(nn.Module):

    def __init__(self, in_channels=1, num_classes=5):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 48, kernel_size=11, stride=4, padding=2, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(48, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(128, 192, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(192, 192, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(192, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool1d(2)
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(128 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        out = self.features(x)
        out = self.avgpool(out)
        out = self.flatten(out)
        out = self.classifier(out)
        return out


if __name__ == '__main__':
    device = torch.device('cuda:0')
    temp = torch.randn([32, 1, 1024]).to(device)
    model = AlexNet(in_channels=1, num_classes=5).to(device)
    out = model(temp)
    print(out.shape)

