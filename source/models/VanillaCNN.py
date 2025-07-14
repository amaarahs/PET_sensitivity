import torch
import torch.nn as nn
import torch.nn.functional as F

class VanillaCNN(nn.Module):
    def __init__(self, in_channels=2, out_channels=1):
        super(VanillaCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=15, padding=7)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=9, padding=4)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(64, 32, kernel_size=7, padding=3)
        self.conv6 = nn.Conv2d(32, 16, kernel_size=15, padding=7)
        self.conv7 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(16, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out1 = self.relu(self.conv1(x))
        out2 = self.relu(self.conv2(out1))
        out3 = self.relu(self.conv3(out2))
        out4 = self.relu(self.conv4(out3))
        out5 = self.relu(self.conv5(out4))
        out6 = self.relu(self.conv6(out5))
        out7 = self.relu(self.conv7(out6))
        out8 = self.conv8(out7)
        return self.relu(out8)