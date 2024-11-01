from torch import Tensor, nn


class BasePrior(nn.Module):
    r"""
    The BasePrior here not only need to sample, but also should have log_prob. This is for calculation of DKL in a tractable fashion.

    I did not use ABC class to force this because it would be quite clear
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def sample(self, n_sample: int, device) -> Tensor:
        raise NotImplementedError

    def log_prob(self, x: Tensor) -> Tensor:
        raise NotImplementedError
