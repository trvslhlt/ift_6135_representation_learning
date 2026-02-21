import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride_dw: int, stride_pw: int):
        super().__init__()
        """
        Build the depthwise separable convolution layer
        For the depthwise convolution (use padding=1 and bias=False for the convolution)
        For the pointwise convolution (use padding=0 and bias=False fot the convolution)

        Inputs:
            in_channels: number of input channels
            out_channels: number of output channels
            stride_dw: stride for depthwise convolution
            stride_pw: stride for pointwise convolution
        """
        self.depthwise = nn.Conv2d(
            in_channels, 
            in_channels, 
            kernel_size=3,
            stride=stride_dw,
            padding=1,
            groups=in_channels,
            bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=stride_pw,
            padding=0,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = F.relu(x)
        return x
    

class MobileNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        """
        Build the MobileNet architecture
        For the first standard convolutional layer (use padding=1 and bias=False for the convolution)
        For the AvgPool layer, use nn.AdaptiveAvgPool2d.

        Inputs:
            num_classes: number of classes for classification
        """
        # input         filter
        # 224x224x3     3x3x3x32  
        self.conv0 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        # 112x112x32    3x3x32 dw
        # 112x112x32    1x1x32x64
        self.dw_sep_conv0 = DepthwiseSeparableConv(32, 64, 1, 1)
        # 112x112x64    3x3x64 dw
        # 56x56x64      1x1x64x128
        self.dw_sep_conv1 = DepthwiseSeparableConv(64, 128, 2, 1)
        # 56x56x128     3x3x128 dw
        # 56x56x128     1x1x128x128
        self.dw_sep_conv2 = DepthwiseSeparableConv(128, 128, 1, 1)
        # 56x56x128     3x3x128 dw
        # 28x28x128     1x1x128x256
        self.dw_sep_conv3 = DepthwiseSeparableConv(128, 256, 2, 1)
        # 28x28x256     3x3x256 dw
        # 28x28x256     1x1x256x256
        self.dw_sep_conv4 = DepthwiseSeparableConv(256, 256, 1, 1)
        # 28x28x256     3x3x256 dw
        # 14x14x256     1x1x256x512
        self.dw_sep_conv5 = DepthwiseSeparableConv(256, 512, 2, 1)
        # 14x14x512     3x3x512 dw
        # 14x14x512     1x1x512x512
        def create_dw_sep_6(): return DepthwiseSeparableConv(512, 512, 1, 1)
        self.dw_sep_conv61 = create_dw_sep_6()
        self.dw_sep_conv62 = create_dw_sep_6()
        self.dw_sep_conv63 = create_dw_sep_6()
        self.dw_sep_conv64 = create_dw_sep_6()
        self.dw_sep_conv65 = create_dw_sep_6()
        # 14x14x512     3x3x512 dw
        # 7x7xx512      1x1x512x1024
        self.dw_sep_conv7 = DepthwiseSeparableConv(512, 1024, 2, 1)
        # 7x7x1024      3x3x1024 dw
        # 7x7x1024      1x1x1024x1024
        self.dw_sep_conv8 = DepthwiseSeparableConv(1024, 1024, 1, 1)
        # 7x7x1024      7x7 pool
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 1x1x1024      1024xnum_classes
        self.fc = nn.Linear(1024, num_classes)
        # 1x1xnum_classes
        self.softmax = nn.Softmax(dim=1) # dim 0 is the batch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv0(x)
        x = self.dw_sep_conv0(x)
        x = self.dw_sep_conv1(x)
        x = self.dw_sep_conv2(x)
        x = self.dw_sep_conv3(x)
        x = self.dw_sep_conv4(x)
        x = self.dw_sep_conv5(x)
        x = self.dw_sep_conv61(x)
        x = self.dw_sep_conv62(x)
        x = self.dw_sep_conv63(x)
        x = self.dw_sep_conv64(x)
        x = self.dw_sep_conv65(x)
        x = self.dw_sep_conv7(x)
        x = self.dw_sep_conv8(x)
        x = self.avgpool(x)
        x = x.flatten(1) # don't flatten dim 0 (batch)
        x = self.fc(x)
        # x = self.softmax(x) # disable for use with CrossEntropyLoss
        return x