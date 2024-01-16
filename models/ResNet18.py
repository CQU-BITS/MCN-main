
import torch
from torch import nn
from torchsummary import summary
from thop import profile



class ResBlk(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResBlk, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride[0], padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = nn.Sequential()
        if out_channels != in_channels:
            # 对 x 进行升维再下采样 [b, in_channels, h, w] => [b, out_channels, h, w]
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride[0], bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # short cut. [b, ch_in, h, w] => [b, ch_out, h, w]
        out = self.downsample(x) + out
        out = self.relu(out)

        return out


class ResNet18(nn.Module):
    def __init__(self, in_channels=1, num_classes=5):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        self.block1 = ResBlk(in_channels=64, out_channels=64, stride=[1, 1])
        self.block2 = ResBlk(in_channels=64, out_channels=64, stride=[1, 1])

        self.block3 = ResBlk(in_channels=64, out_channels=128, stride=[2, 1])
        self.block4 = ResBlk(in_channels=128, out_channels=128, stride=[1, 1])

        self.block5 = ResBlk(in_channels=128, out_channels=256, stride=[2, 1])
        self.block6 = ResBlk(in_channels=256, out_channels=256, stride=[1, 1])

        self.block7 = ResBlk(in_channels=256, out_channels=512, stride=[2, 1])
        self.block8 = ResBlk(in_channels=512, out_channels=512, stride=[1, 1])

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, num_classes, bias=False)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv1d):
        #         print(m.bias)

    def forward(self, x):
        out = self.conv1(x)

        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.block7(out)
        out = self.block8(out)

        out = self.avgpool(out)
        out = self.flatten(out)
        out = self.fc(out)

        return out


if __name__ == '__main__':
    device = torch.device('cuda:0')
    temp = torch.randn([32, 1, 1024]).to(device)
    model = ResNet18(in_channels=1, num_classes=5).to(device)
    out = model(temp)
    print(out.shape)


