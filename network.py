import torch.nn as nn
import torch.nn.functional as F
import torchsummary

class Net(nn.Module):
    """The CNN from Colorful Image Colorization."""
    def __init__(self):
        super(Net, self).__init__()

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

        # Bins
        self.conv_bins = nn.Conv2d(in_channels=128, out_channels=313, kernel_size=1, stride=1, padding=0, dilation=1)
        # TODO: Implement scaling/softmax/decoding

        # Print summary
        torchsummary.summary(self, (1, 224, 224))

    def forward(self, x):
        # Conv1
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.conv1_norm(x)

        # Conv2
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.conv2_norm(x)

        # Conv3
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = self.conv3_norm(x)

        # Conv4
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = self.conv4_norm(x)

        # Conv5
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        x = self.conv5_norm(x)

        # Conv6
        x = F.relu(self.conv6_1(x))
        x = F.relu(self.conv6_2(x))
        x = F.relu(self.conv6_3(x))
        x = self.conv6_norm(x)

        # Conv7
        x = F.relu(self.conv7_1(x))
        x = F.relu(self.conv7_2(x))
        x = F.relu(self.conv7_3(x))
        x = self.conv7_norm(x)

        # Conv8
        x = F.relu(self.conv8_1(x))
        x = F.relu(self.conv8_2(x))
        x = F.relu(self.conv8_3(x))

        # Bins
        x = self.conv_bins(x)
        # TODO: Implement scaling/softmax/decoding

        return x
