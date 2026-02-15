import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride_dw, stride_pw):
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
        self.depthwise = NotImplemented
        self.bn1 = NotImplemented
        self.pointwise = NotImplemented
        self.bn2 = NotImplemented
        raise NotImplementedError
    
    def forward(self, x):
        raise NotImplementedError
    

class MobileNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        """
        Build the MobileNet architecture
        For the first standard convolutional layer (use padding=1 and bias=False for the convolution)
        For the AvgPool layer, use nn.AdaptiveAvgPool2d.

        Inputs:
            num_classes: number of classes for classification
        """
        self.conv0 = NotImplemented
        self.dw_sep_conv0 = NotImplemented 
        self.dw_sep_conv1 = NotImplemented
        self.dw_sep_conv2 = NotImplemented
        self.dw_sep_conv3 = NotImplemented
        self.dw_sep_conv4 = NotImplemented
        self.dw_sep_conv5 = NotImplemented
        self.dw_sep_conv61 = NotImplemented
        self.dw_sep_conv62 = NotImplemented
        self.dw_sep_conv63 = NotImplemented
        self.dw_sep_conv64 = NotImplemented
        self.dw_sep_conv65 = NotImplemented
        self.dw_sep_conv7 = NotImplemented
        self.dw_sep_conv8 = NotImplemented
        self.avgpool = NotImplemented
        self.fc = NotImplemented
        self.softmax = NotImplemented

        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError