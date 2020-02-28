import torch
import numpy as np
from typing import Tuple
import random
import torch.nn.functional as F


def pad_to_square(img: torch.Tensor, pad_value) -> Tuple[torch.Tensor, Tuple]:
    """
    Squares the image by padding to zero (instead of cropping)
    :param img: image to pad
    :param pad_value: padding to add to the image
    :return: a tuple containing the padded image and the pad tuple
    """
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    # noinspection PyTypeChecker
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image: torch.Tensor, size: int) -> torch.Tensor:
    """
    Resizes the image to the desired size
    :param image: image tensor
    :param size: new size
    :return: resized image tensor
    """
    #  The image tensor is unsqueezed because, according to the pytorch documentation
    #       The input dimensions are interpreted in the form:
    #       mini-batch x channels x [optional depth] x [optional height] x width.
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def random_resize(images: torch.Tensor, min_size: int = 288, max_size: int = 448) -> torch.Tensor:
    """
    Randomly resize between the specified range. Options are divisible by 32. All
    images in the mini-batch are resized by the same randomly chosen size.
    :param images: images tensor, including the mini-batch dimension
    :param min_size: minimum size. Must be divisible by 32.
    :param max_size: maximum size. Must be divisible by 32
    :return: randomly resized images tensor.
    """
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest")
    return images
