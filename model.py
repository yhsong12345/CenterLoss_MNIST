import torch
import torch.nn as nn
import torch.nn.functional as F
from common import *


class LeNetspp(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNetspp, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.PReLU(),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.PReLU()
        )
        self.mp1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.PReLU()
        )
        self.mp2 = nn.MaxPool2d(2,2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2),
            nn.PReLU()
        )
        self.mp3 = nn.MaxPool2d(2,2)
        self.FC1 = nn.Linear(128*3*3, 2)
        self.act1 = nn.PReLU()
        self.FC2 = nn.Linear(2, num_classes)


    def forward(self, x):
        o = self.conv1(x)
        o = self.mp1(o)
        o = self.conv2(o)
        o = self.mp2(o)
        o = self.conv3(o)
        o = self.mp3(o)
        o = o.view(o.size(0), -1)
        o = self.FC1(o)
        o = self.act1(o)
        r = self.FC2(o)
        return o, r