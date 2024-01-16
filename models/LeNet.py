import torch
from torch import nn



# -----------------------input size>=32---------------------------------
class LeNet(nn.Module):
    def __init__(self, in_channels, num_classes=5):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 6, 5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(6, 16, 5),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d((5))  # adaptive change the outputsize to (16,5)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5, 30),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(30, 10),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(10, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    device = torch.device('cuda:0')
    temp = torch.randn([32, 1, 1024]).to(device)
    model = LeNet(in_channels=1, num_classes=5).to(device)
    out = model(temp)
    print(out.shape)
