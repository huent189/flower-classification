import numpy
import torch
import torch.nn as nn

class VGG(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=(3, 3), stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(64),
                                    nn.Conv2d(64, 64, (3, 3), 1, 1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(64),
                                    nn.MaxPool2d((2, 2), 2))
        self.layer2 = nn.Sequential(nn.Conv2d(64, 128, (3, 3), 1, 1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(128),
                                    nn.Conv2d(128, 128, (3, 3), 1, 1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(128),
                                    nn.MaxPool2d((2, 2), 2))
        self.layer3 = nn.Sequential(nn.Conv2d(128, 256, (3, 3), 1, 1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(256),
                                    nn.Conv2d(256, 256, (3, 3), 1, 1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(256),
                                    nn.Conv2d(256, 256, (3, 3), 1, 1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(256),
                                    nn.MaxPool2d((2, 2), 2))
        self.layer4 = nn.Sequential(nn.Conv2d(256, 512, (3, 3), 1, 1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(512),
                                    nn.Conv2d(512, 512, (3, 3), 1, 1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(512),
                                    nn.Conv2d(512, 512, (3, 3), 1, 1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(512),
                                    nn.MaxPool2d((2, 2), 2))
        hid_dim = int(input_dim[0] * input_dim[1] / 16 / 16 * 512)
        self.clasifier = nn.Sequential(nn.Linear(hid_dim, 512),
                                       nn.ReLU(),
                                       nn.BatchNorm1d(512),
                                       nn.Linear(512, 512),
                                       nn.ReLU(),
                                       nn.BatchNorm1d(512),
                                       nn.Linear(512, output_dim))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = torch.flatten(out, start_dim=1)
        # print(out.shape)
        return self.clasifier(out)


if __name__ == "__main__":
    model = VGG((128, 128), 6)
    out_test = model(torch.rand((10, 3, 128, 128)))
    print(model)
    print(out_test.shape)
