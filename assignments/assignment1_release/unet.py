import torch
from torch import nn


def double_conv_block(in_channels, out_channels):
    """
    This double conv block are the blocks used in the encoder part of UNet.
    It uses a padding of 1 to preserve spatial dimensions.
    
    :param in_channels: Description
    :param out_channels: Description
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )


class DecoderBlock(nn.Module):
    """
    Decoder block of UNet. It consists of an upconvolution layer followed by a double conv block.
    Use the double_conv_block defined above.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upconv = NotImplemented
        self.conv = NotImplemented
        raise NotImplementedError

    def forward(self, x, skip):
        raise NotImplementedError


class UNet(nn.Module):
    def __init__(self, input_shape, num_classes):
        super().__init__()
        self.encoder_block1 = NotImplemented
        self.encoder_block2 = NotImplemented
        self.encoder_block3 = NotImplemented
        self.encoder_block4 = NotImplemented
        self.encoder_block5 = NotImplemented
        self.pool = NotImplemented
        self.decoder_block1 = NotImplemented
        self.decoder_block2 = NotImplemented
        self.decoder_block3 = NotImplemented
        self.decoder_block4 = NotImplemented
        self.outconv = NotImplemented
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError