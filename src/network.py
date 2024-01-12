import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
    
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        # out = self.bn(out)
        out = self.relu(out)

        out = self.conv2(out)
        # out = self.bn(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)

        return out


class Resnet(nn.Module):
    def __init__(self, block, layers, num_classes=5, in_channles=3):
        super(Resnet, self).__init__()
        self.channles = [2**i for i in range(4,8)]  # [16, 32, 64, 128]
        self.base_width = self.channles[0]
        self.this_channles = self.channles[0]
        self.conv1 = nn.Conv1d(in_channles, self.channles[0], kernel_size=7, stride=6, padding=3)
        self.bn = nn.BatchNorm1d(self.channles[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=5, stride=4, padding=2)
        self.layer1 = self._make_layer(block, self.channles[0], layers[0])
        self.layer2 = self._make_layer(block, self.channles[1], layers[1], 4)
        self.layer3 = self._make_layer(block, self.channles[2], layers[2], 4)
        self.layer4 = self._make_layer(block, self.channles[3], layers[3], 4)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(self.channles[3]*block.expansion, num_classes)
    
    def _make_layer(self, block, out_channels, num_block, stride=1):
        downsample = None
        out_channels_new = out_channels*block.expansion
        if stride != 1 or self.this_channles != out_channels_new:
            downsample = nn.Sequential(
                nn.Conv1d(self.this_channles, out_channels_new, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels_new)
            )
        
        layers = []
        layers.append(block(self.this_channles, out_channels, stride, downsample))
        self.this_channles = out_channels_new
        for _ in range(1, num_block):
            layers.append(block(self.this_channles, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        # x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


if __name__ == '__main__':
    resnet = Resnet(BasicBlock, [2, 2, 2, 2])
    print(resnet(torch.randn(4, 3, 15000)))