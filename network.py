import torch.nn as nn
import torch.nn.functional as F
import torchsummary

class Net(nn.Module):
    """ The CNN from Colorful Image Colorization.

    Architecture: https://github.com/richzhang/colorization/blob/master/models/colorization_deploy_v2.prototxt
    """
    def __init__(self):
        super(Net, self).__init__()

        # Conv1
        self.conv1_1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, dilation=1)
        self.conv1_2norm = nn.BatchNorm2d(num_features=64)

        # Print summary
        torchsummary.summary(self, (1, 224, 224))

    def forward(self, x):
        # Conv1
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.conv1_2norm(x)
        return x
