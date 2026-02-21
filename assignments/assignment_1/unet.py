import torch
from torch import nn


def double_conv_block(in_channels: int, out_channels: int) -> nn.Module:
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
    def __init__(self, in_channels: int, out_channels: int, skip_connections = True):
        super().__init__()
        # The stride doesn't mean "move the kernel 2px over the input." 
        # It means space the input pixels 2 apart in the output grid, then convolve.
        # Input:        Sparse grid (stride=2):
        # a b           a 0 b
        # c d           0 0 0
        #               c 0 d
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # with skip_connections:    conv expects in_channels (upconv out + skip)
        # without:                  conv expects out_channels (upconv out only)
        self.conv = double_conv_block(in_channels if skip_connections else out_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None) -> torch.Tensor:
        x = self.upconv(x)
        if skip is not None:
            # concatenate along channel axis
            x = torch.cat([x, skip], dim=1)
        x = self.conv(x) 
        return x


class UNet(nn.Module):

    skip_connections: bool

    def __init__(self, in_channels: int, num_classes: int, skip_connections = True):
        super().__init__()
        self.skip_connections = skip_connections
        self.encoder_block1 = double_conv_block(in_channels, 64)
        self.encoder_block2 = double_conv_block(64, 128)
        self.encoder_block3 = double_conv_block(128, 256)
        self.encoder_block4 = double_conv_block(256, 512)
        self.encoder_block5 = double_conv_block(512, 1024)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.decoder_block1 = DecoderBlock(1024, 512, skip_connections=skip_connections)
        self.decoder_block2 = DecoderBlock(512, 256, skip_connections=skip_connections)
        self.decoder_block3 = DecoderBlock(256, 128, skip_connections=skip_connections)
        self.decoder_block4 = DecoderBlock(128, 64, skip_connections=skip_connections)
        self.outconv = nn.Conv2d(
            64,
            num_classes,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.encoder_block1(x)
        x = self.pool(e1)
        e2 = self.encoder_block2(x)
        x = self.pool(e2)
        e3 = self.encoder_block3(x)
        x = self.pool(e3)
        e4 = self.encoder_block4(x)
        x = self.pool(e4)
        x = self.encoder_block5(x) # bottleneck
        x = self.decoder_block1(x, e4 if self.skip_connections else None)
        x = self.decoder_block2(x, e3 if self.skip_connections else None)
        x = self.decoder_block3(x, e2 if self.skip_connections else None)
        x = self.decoder_block4(x, e1 if self.skip_connections else None)
        x = self.outconv(x)
        return x