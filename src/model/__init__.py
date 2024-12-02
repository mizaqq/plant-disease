import torch
import os
from pathlib import Path
from torchvision import datasets, transforms
import cv2
from tqdm import tqdm
import torch.nn.functional as F


class Convolutional(torch.nn.Module):
    def __init__(self):
        super(Convolutional, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, 2, padding_mode="replicate")
        self.pool1 = torch.nn.MaxPool2d(2)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, 2, padding_mode="replicate")
        self.pool2 = torch.nn.MaxPool2d(2)
        self.conv3 = torch.nn.Conv2d(64, 128, 3, 2, padding_mode="replicate")
        self.pool3 = torch.nn.MaxPool2d(2)
        self.conv4 = torch.nn.Conv2d(128, 256, 3, 2, padding_mode="replicate")
        self.pool4 = torch.nn.MaxPool2d(2)
        self.conv5 = torch.nn.Conv2d(256, 512, 3, 2, padding_mode="replicate")
        self.fc1 = torch.nn.Linear(1152, 32)
        self.fc2 = torch.nn.Linear(32, 4)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
