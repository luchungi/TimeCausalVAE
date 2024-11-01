import torch.nn as nn


class BaseEncoder(nn.Module):
    """This is a base class for Encoders neural networks."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x):
        raise NotImplementedError()
