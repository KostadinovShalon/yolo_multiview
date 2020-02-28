import torch.nn as nn
import torch
from typing import Tuple


def horizontal_flip(images: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Do a flip in the last dimension. The targets
    are flipped by subtracting the y coordinate from 1

    :param images: images tensor
    :param targets: targets tensor
    :return: images and targets flipped
    """
    images = torch.flip(images, [-1])
    # noinspection PyTypeChecker
    targets[:, 2] = 1 - targets[:, 2]
    return images, targets


def to_cpu(tensor):
    """
    Sets a tensor to the CPU
    :param tensor: a tensor
    :return: the tensor in cpu device
    """
    return tensor.detach().cpu()


def weights_init_normal(m: nn.Module):
    """
    Initializes the weights of a module. If the module is convolutional, it is randomly
    initialized with a normal distribution with 0 mean and 0.02 std. If the module is
    a BatchNorm2d, it is initialized with a normal distribution with 1 mean and 0.02 std, and the bias
    is set to 0.
    :param m: PyTorch module
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)
