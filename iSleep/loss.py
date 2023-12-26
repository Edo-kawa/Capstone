import torch
import torch.nn.functional as F
from functions import generate_framewise_target

# TODO: Pad target to be frame-wise
def clip_bce(framewise_output, target):
    """Binary crossentropy loss.
    """

    frames_num = framewise_output.shape[1]

    return F.binary_cross_entropy(
        framewise_output, generate_framewise_target(target, frames_num))