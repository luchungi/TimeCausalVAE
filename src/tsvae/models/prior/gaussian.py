import torch
import numpy as np
from torch import Tensor

from tsvae.models.prior.base_prior import BasePrior

PI = torch.tensor(np.pi)


def log_standard_normal(x):
    log_p = -0.5 * torch.log(2.0 * PI) - 0.5 * x**2.0
    log_p = torch.sum(log_p, dim=-1)
    return log_p


def entropy_normal(log_var):
    entropy = -0.5 * (1 + torch.log(2.0 * PI) + log_var)
    entropy = torch.sum(entropy, dim=-1)
    return entropy


class GaussianPrior(BasePrior):
    r"""
    Standard Gaussian Prior
    """

    def __init__(self, dim=2):
        super(GaussianPrior, self).__init__()
        self.dim = dim
        # params weights

    def sample(self, n_sample: int, device) -> Tensor:
        return torch.randn(n_sample, self.dim, device=device)

    def log_prob(self, x: Tensor) -> Tensor:
        return log_standard_normal(x)
