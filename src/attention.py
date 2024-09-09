import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, width, height = x.size()
        query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, width * height)
        value = self.value_conv(x).view(batch_size, -1, width * height)

        attention = torch.bmm(query, key)
        attention = self.softmax(attention)

        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)

        out = self.gamma * out + x
        return out


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        # Adaptive average pooling to reduce spatial dimensions to 1x1
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # Fully connected layers to recalibrate channel-wise responses
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channels, _, _ = x.size()  # Get the dimensions of the input tensor

        # Squeeze: Perform global average pooling
        out = self.global_avg_pool(x).view(batch_size, channels)  # Shape: [batch_size, channels]

        # Apply the first fully connected layer
        out = self.relu(self.fc1(out))  # Shape should be [batch_size, in_channels // reduction]

        # Apply the second fully connected layer
        out = self.sigmoid(self.fc2(out))  # Shape should be [batch_size, in_channels]

        # Reshape for broadcasting: [batch_size, channels, 1, 1]
        out = out.view(batch_size, channels, 1, 1)

        # Excitation: Element-wise multiplication with the original input
        return x * out


