import numpy as np
from PIL import Image 

import torch
import torchvision.transforms as transforms

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

# put statistics of proprioception and actions here 
proprio_mean = torch.zeros(6) # for cartesian proprios
proprio_std = torch.ones(6) # for cartesian proprios

action_mean = torch.zeros(6) # for cartesian actions
action_std = torch.ones(6) # for cartesian actions

def to_pil(img : torch.Tensor):
    img = np.moveaxis(img.numpy()*255, 0, -1)
    return Image.fromarray(img.astype(np.uint8))

def unnormalize_fn(mean : tuple, std : tuple) -> transforms.Compose:
    """
    returns a transformation that turns torch tensor to PIL Image
    """
    return transforms.Compose(
        [
            transforms.Normalize(
                mean=tuple(-m / s for m, s in zip(mean, std)),
                std=tuple(1.0 / s for s in std),
            ),
            transforms.Lambda(lambda x: torch.clamp(x, 0., 1.)), 
            transforms.ToPILImage(),
        ]
    )

def get_default_vision_aug(type : str, size : int = 224):
    """
    returns the default augmentation for the given type
    type in ["imagenet", "clip"]
    """
    assert type in ["imagenet", "clip"], "type must be one of imagenet, clip"
    if type == "imagenet": 
        return transforms.Compose([
            transforms.Resize(size=size), 
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_DEFAULT_MEAN,
                std=IMAGENET_DEFAULT_STD,
            ),
        ])
    elif type == "clip":
        return transforms.Compose([
            transforms.Resize(size=size), 
            transforms.ToTensor(),
            transforms.Normalize(
                mean=OPENAI_CLIP_MEAN,
                std=OPENAI_CLIP_STD,
            ),
        ])

# normalize proprioception and action data 
@torch.jit.script
def normalize_data(
    x : torch.tensor, 
    mean : torch.tensor,
    std : torch.tensor,
):
    """
    x : N, k
    proprioception data (x) is normalized by mean and std, which are tensors of shape k
    """
    assert x.shape[1] == mean.shape[0] == std.shape[0], "x, mean, std must have the same dimension"
    return (x - mean) / std

# unnormalize proprioception and action data 
@torch.jit.script
def unnormalize_data(
    x : torch.tensor, 
    mean : torch.tensor,
    std : torch.tensor,
):
    """
    x : N, k
    proprioception data (x) is normalized by mean and std, which are tensors of shape k
    """
    assert x.shape[1] == mean.shape[0] == std.shape[0], "x, mean, std must have the same dimension"
    return x * std + mean

