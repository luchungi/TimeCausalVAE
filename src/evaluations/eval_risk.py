import numpy as np
import torch


def value_at_risk(x, q):
    r"""
    x should be dim1 torch tensor
    """
    # TODO: check input shape
    x0 = torch.quantile(x, q)
    return x0


def expected_shortfall(x, q):
    r"""
    Also called average value at risk
    x should be dim1 torch tensor
    """
    # TODO: check input shape
    x0 = value_at_risk(x, q)
    return x[x < x0].mean()
