import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
 

def discrete_2d_convolution(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
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

    # Pad the image with zeros on all sides
    pad_dims = ((pad_height, pad_height), (pad_width, pad_width)) # (before_dim1, after_dim1), ...
    padded_image = np.pad(image, pad_dims, mode='constant', constant_values=0)

    # TODO: perform the convolution operation
    convolved_image = np.zeros((image_height, image_width))
    for i in range(image_height):
        for j in range(image_width):
            src_data = padded_image[i:i + kernel_height, j:j + kernel_width]
            convolved_image[i, j] = np.sum(src_data * kernel)
    
    # Alternative approach
    # - compute all convolution windows at one time instead of iterating
    # - uses much more memory and fails locally on a large image
    # windows = np.lib.stride_tricks.sliding_window_view(padded_image, (kernel_height, kernel_width))
    # convolved_image = np.sum(windows * kernel, axis=(2, 3))

    return convolved_image
    

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, smooth=1):
        """
        Compute the Dice Loss between the logits and the targets.
        In this implementation, we use smoothing to avoid division by zero: it is added 
        to both the numerator and the denominator
        """
        # 2 * card(X intersection Y) + smooth
        # -----------------------------------
        #      card(X) + card(Y) + smooth
        inputs = torch.sigmoid(logits).flatten()
        targets = targets.flatten()
        intersection = (inputs * targets).sum() # soft intersection with probabilities
        num = 2 * intersection + smooth
        denom = inputs.sum() + targets.sum() + smooth
        dice = num / denom
        return 1 - dice
    

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
        # In this case, we use the binary cross entropy loss 
        # instead of the cross entropy loss since we have a binary segmentation task.
        self.ceLoss = BinaryCELoss()

    def forward(self, logits, targets):
        raise NotImplementedError