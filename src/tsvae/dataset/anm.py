from torch.utils.data import Dataset
import torch


class ANMDataset(Dataset):
    def __init__(self, n_sample, x_lower=0, x_upper=2, noise_std=1):
        true_function = lambda x: torch.nn.Softplus()(x)
        x = torch.rand(n_sample, 1) * (x_upper - x_lower) + x_lower
        eps = torch.randn(n_sample, 1) * noise_std
        xn = x + eps
        y = true_function(xn)
        self.labels = x
        self.data = y.unsqueeze(1)
