import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
 

def discrete_2d_convolution(image, kernel):
    # 1. Convert to float to avoid uint8 values
    image = image.astype(np.float64)
    kernel = np.array(kernel, dtype=np.float64)

    # We do not flip the kernel since we are implementing cross-correlation

    # Extract the dimensions
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    
    # Because we want the output image to have the same size as the input image, we need to pad the input image
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    # TODO: Pad the image with zeros on all sides

    # TODO: perform the convolution operation

    raise NotImplementedError
    

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, logits, targets, smooth=1):
        """
        Compute the Dice Loss between the logits and the targets.
        In this implementation, we use smoothing to avoid division by zero: it is added to both the numerator and the denominator
        """
        inputs = torch.sigmoid(logits)

        # TODO: compute the Dice coefficient

        return 1. - dice
    

class BinaryCELoss(nn.Module):
    def __init__(self):
        super(BinaryCELoss, self).__init__()

        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        if targets.shape != logits.shape:
            targets = targets.view(logits.shape)
        return self.bce(logits, targets.float())


class DiceCELoss(nn.Module):
    def __init__(self):
        super(DiceCELoss, self).__init__()
        self.diceLoss = DiceLoss()
        # In this case, we use the binary cross entropy loss instead of the cross entropy loss since we have a binary segmentation task.
        self.ceLoss = BinaryCELoss()

    def forward(self, logits, targets):
        raise NotImplementedError