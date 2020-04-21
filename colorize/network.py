import torch.nn as nn
import torch.nn.functional as F
from . import util

class Network(nn.Module):
    """The CNN from Colorful Image Colorization."""

    def __init__(self):
        """Constructor defining all layers."""
        super(Network, self).__init__()

        # Conv1
        self.conv1_1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, dilation=1)
        self.conv1_norm = nn.BatchNorm2d(num_features=64)

        # Conv2
        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1, dilation=1)
        self.conv2_norm = nn.BatchNorm2d(num_features=128)

        # Conv3
        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, dilation=1)
        self.conv3_norm = nn.BatchNorm2d(num_features=256)

        # Conv4
        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv4_norm = nn.BatchNorm2d(num_features=512)

        # Conv5
        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv5_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv5_norm = nn.BatchNorm2d(num_features=512)

        # Conv6
        self.conv6_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv6_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv6_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv6_norm = nn.BatchNorm2d(num_features=512)

        # Conv7
        self.conv7_1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv7_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv7_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv7_norm = nn.BatchNorm2d(num_features=256)

        # Conv8
        self.conv8_1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1, dilation=1)
        self.conv8_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv8_3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1)

        # Distribution Z
        self.conv_dist = nn.Conv2d(in_channels=128, out_channels=313, kernel_size=1, stride=1, padding=0, dilation=1)

    def forward(self, x, summary=False):
        """Performs the forward pass."""
        # Conv1
        if summary: print('Input:\t\t', x.shape)
        x = F.relu(self.conv1_1(x))
        if summary: print('Conv1_1:\t', x.shape)
        x = F.relu(self.conv1_2(x))
        if summary: print('Conv1_2:\t', x.shape)
        x = self.conv1_norm(x)

        # Conv2
        x = F.relu(self.conv2_1(x))
        if summary: print('Conv2_1:\t', x.shape)
        x = F.relu(self.conv2_2(x))
        if summary: print('Conv2_2:\t', x.shape)
        x = self.conv2_norm(x)

        # Conv3
        x = F.relu(self.conv3_1(x))
        if summary: print('Conv3_1:\t', x.shape)
        x = F.relu(self.conv3_2(x))
        if summary: print('Conv3_2:\t', x.shape)
        x = F.relu(self.conv3_3(x))
        if summary: print('Conv3_3:\t', x.shape)
        x = self.conv3_norm(x)

        # Conv4
        x = F.relu(self.conv4_1(x))
        if summary: print('Conv4_1:\t', x.shape)
        x = F.relu(self.conv4_2(x))
        if summary: print('Conv4_2:\t', x.shape)
        x = F.relu(self.conv4_3(x))
        if summary: print('Conv4_3:\t', x.shape)
        x = self.conv4_norm(x)

        # Conv5
        x = F.relu(self.conv5_1(x))
        if summary: print('Conv5_1:\t', x.shape)
        x = F.relu(self.conv5_2(x))
        if summary: print('Conv5_2:\t', x.shape)
        x = F.relu(self.conv5_3(x))
        if summary: print('Conv5_3:\t', x.shape)
        x = self.conv5_norm(x)

        # Conv6
        x = F.relu(self.conv6_1(x))
        if summary: print('Conv6_1:\t', x.shape)
        x = F.relu(self.conv6_2(x))
        if summary: print('Conv6_2:\t', x.shape)
        x = F.relu(self.conv6_3(x))
        if summary: print('Conv6_3:\t', x.shape)
        x = self.conv6_norm(x)

        # Conv7
        x = F.relu(self.conv7_1(x))
        if summary: print('Conv7_1:\t', x.shape)
        x = F.relu(self.conv7_2(x))
        if summary: print('Conv7_2:\t', x.shape)
        x = F.relu(self.conv7_3(x))
        if summary: print('Conv7_3:\t', x.shape)
        x = self.conv7_norm(x)

        # Conv8
        x = F.relu(self.conv8_1(x))
        if summary: print('Conv8_1:\t', x.shape)
        x = F.relu(self.conv8_2(x))
        if summary: print('Conv8_2:\t', x.shape)
        x = F.relu(self.conv8_3(x))
        if summary: print('Conv8_3:\t', x.shape)

        # Distribution Z
        x = self.conv_dist(x)
        if summary: print('Conv_dist:\t', x.shape)
        x = F.softmax(x, dim=1)

        return x
