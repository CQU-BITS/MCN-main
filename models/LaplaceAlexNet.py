
import torch
from torch import nn, optim
from math import pi
import torch.nn.functional as F


def Laplace(p):
    A = 0.08
    ep = 0.03
    tal = 0.1
    f = 50
    w = 2 * pi * f
    q = torch.tensor(1 - pow(ep, 2))
    y = A * torch.exp((-ep / (torch.sqrt(q))) * (w * (p - tal))) * (-torch.sin(w * (p - tal)))
    return y


class Laplace_fast(nn.Module):
    def __init__(self, kernel_size, out_channels=64):
        super(Laplace_fast, self).__init__()

        self.out_channels = out_channels
        self.kernel_size = kernel_size - 1
        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1

        self.a_ = nn.Parameter(torch.linspace(1, 10, out_channels).view(-1, 1))
        self.b_ = nn.Parameter(torch.linspace(0, 10, out_channels).view(-1, 1))

    def forward(self, waveforms):
        time_disc = torch.linspace(0, 1, steps=int((self.kernel_size)))
        p1 = time_disc.cuda() - self.b_.cuda() / self.a_.cuda()
        laplace_filter = Laplace(p1)
        self.filters = (laplace_filter).view(self.out_channels, 1, self.kernel_size).to(waveforms.device)

        return F.conv1d(waveforms, self.filters, stride=1, padding=1, dilation=1, bias=None, groups=1)


class LaplaceAlexNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=5):
        super(LaplaceAlexNet, self).__init__()
        self.laplace = Laplace_fast(kernel_size=32, out_channels=64)
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool1d(6)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.laplace(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    device = torch.device('cuda:0')
    temp = torch.randn([32, 1, 1024]).to(device)
    model = LaplaceAlexNet(in_channels=1, num_classes=5).to(device)
    out = model(temp)
    print(out.shape)


