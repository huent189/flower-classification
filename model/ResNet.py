import torch.nn as nn
import torch
class ResidualBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        if c_in != c_out:
            # downsample must be performed on first convolution
            self.conv1 = nn.Sequential(nn.Conv2d(c_in, c_out, (3, 3), stride=2, padding=1),
                                       nn.BatchNorm2d(c_out),
                                       nn.ReLU())
            # projection performed when shortcut connection have different dim between in and out
            self.projection = nn.Sequential(nn.Conv2d(c_in, c_out, (1, 1), stride=2),
                                            nn.BatchNorm2d(c_out))
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(c_in, c_out, (3, 3), stride=1, padding=1),
                                       nn.BatchNorm2d(c_out),
                                       nn.ReLU())
            self.projection = None
        self.conv2 = nn.Sequential(nn.Conv2d(c_out, c_out, (3, 3), stride=1, padding=1),
                                   nn.BatchNorm2d(c_out),
                                   nn.ReLU())
        self.relu = nn.ReLU()

    def _shortcut(self, x, z):
        if self.projection:
            x = self.projection(x)
        return x + z

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self._shortcut(x, out)
        return self.relu(out)

class ResNet18(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, (7, 7), stride=2, padding=3),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.MaxPool2d((3, 3), stride=2, padding=1),
                                   ResidualBlock(64, 64), ResidualBlock(64, 64))
        self.conv3 = nn.Sequential(ResidualBlock(64, 128),
                                   ResidualBlock(128, 128))
        self.conv4 = nn.Sequential(ResidualBlock(128, 256),
                                   ResidualBlock(256, 256))
        self.conv5 = nn.Sequential(ResidualBlock(256, 512),
                                   ResidualBlock(512, 512))
        self.clasifier = nn.Linear(512, output_dim)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        # average pooling
        out = torch.mean(out, [2, 3])
        # print(out.shape)
        return self.clasifier(out)

class BottleneckResidualBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        intermediate_dim = int(c_out / 4)
        if c_in != c_out and c_in != intermediate_dim:
            self.conv1 = nn.Sequential(nn.Conv2d(c_in, intermediate_dim, (1, 1), stride=2),
                                       nn.BatchNorm2d(intermediate_dim),
                                       nn.ReLU())
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(c_in, intermediate_dim, (1, 1), stride=1),
                                       nn.BatchNorm2d(intermediate_dim),
                                       nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(intermediate_dim, intermediate_dim, (3, 3), stride=1, padding=1),
                                   nn.BatchNorm2d(intermediate_dim),
                                   nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(intermediate_dim, c_out, (1, 1), stride=1),
                                   nn.BatchNorm2d(c_out))
        if c_in != c_out:
            if c_in != intermediate_dim:
                self.projection = nn.Sequential(nn.Conv2d(c_in, c_out, (1, 1), stride=2),
                                                nn.BatchNorm2d(c_out))
            else:
                self.projection = nn.Sequential(nn.Conv2d(c_in, c_out, (1, 1), stride=1),
                                                nn.BatchNorm2d(c_out))
        else:
            self.projection = None
        self.relu = nn.ReLU()

    def _shortcut(self, x, z):
        if self.projection:
            x = self.projection(x)
        return x + z

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self._shortcut(x, out)
        return self.relu(out)

class ResNet50(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, (7, 7), stride=2, padding=3),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.MaxPool2d((3, 3), stride=2, padding=1),
                                   BottleneckResidualBlock(64, 256),
                                   BottleneckResidualBlock(256, 256),
                                   BottleneckResidualBlock(256, 256))
        self.conv3 = nn.Sequential(BottleneckResidualBlock(256, 512),
                                   BottleneckResidualBlock(512, 512),
                                   BottleneckResidualBlock(512, 512),
                                   BottleneckResidualBlock(512, 512))
        self.conv4 = nn.Sequential(BottleneckResidualBlock(512, 1024),
                                   BottleneckResidualBlock(1024, 1024),
                                   BottleneckResidualBlock(1024, 1024),
                                   BottleneckResidualBlock(1024, 1024),
                                   BottleneckResidualBlock(1024, 1024),
                                   BottleneckResidualBlock(1024, 1024))
        self.conv5 = nn.Sequential(BottleneckResidualBlock(1024, 2048),
                                   BottleneckResidualBlock(2048, 2048),
                                   BottleneckResidualBlock(2048, 2048))
        self.clasifier = nn.Linear(2048, output_dim)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        # average pooling
        out = torch.mean(out, [2, 3])
        # print(out.shape)
        return self.clasifier(out)
