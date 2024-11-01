import numpy as np
import torch
from ot import sliced_wasserstein_distance
from torch import Tensor, nn

from evaluations.awd.pathstodist import paths_to_dist_parallel

from evaluations.esig_dist import compute_exp_sig


class SWD(nn.Module):
    r"""
    Sliced Wasserstein distance
    """

    def __init__(self, n_projections=100, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_projections = n_projections
        # gen = torch.Generator(device=self.device)
        # gen.manual_seed(42)

    def forward(self, X: Tensor, Y: Tensor) -> Tensor:
        X = X.flatten(start_dim=1)
        Y = Y.flatten(start_dim=1)
        swd_dist = sliced_wasserstein_distance(X, Y, n_projections=self.n_projections, seed=0)
        return swd_dist


def l2_dist(x, y: Tensor) -> Tensor:
    return (x - y).pow(2).sum().sqrt()


class SignatureMMD(nn.Module):
    r"""
    Signature MMD or Expected Signature Moment Matching
    """

    def __init__(self, trunc=3, augmented=True, normalise=True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.trunc = 3
        self.augmented = True
        self.normalise = True

    def forward(self, X: Tensor, Y: Tensor) -> Tensor:
        esig_X = compute_exp_sig(x=X, trunc=self.trunc, augmented=self.augmented, normalise=self.normalise)
        esig_Y = compute_exp_sig(x=Y, trunc=self.trunc, augmented=self.augmented, normalise=self.normalise)
        sigmmd_dist = l2_dist(esig_X, esig_Y)
        return sigmmd_dist


class MomentMMD(nn.Module):
    r"""
    Moment MMD
    """

    def __init__(self, p=1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.p = p

    def forward(self, X: Tensor, Y: Tensor) -> Tensor:
        moment_X = X.abs().pow(self.p).mean(axis=0)
        moment_Y = Y.abs().pow(self.p).mean(axis=0)
        momentmmd_dist = l2_dist(moment_X, moment_Y)
        return momentmmd_dist


class SAWD(nn.Module):
    r"""
    Sliced adapted Wasserstein distance
    """

    def __init__(self, n_compute_awd, n_slices, len_slices, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_compute_awd, self.n_slices, self.len_slices = (
            n_compute_awd,
            n_slices,
            len_slices,
        )

    def forward(self, X, Y):
        data1 = np.array(X[..., 0])
        data2 = np.array(Y[..., 0])

        N_DATA = self.n_compute_awd  # number of samples to compute the distance
        T_max = data1.shape[1]  # length of path

        path1 = data1[:N_DATA, :T_max]
        path2 = data2[:N_DATA, :T_max]

        klist = [1] + [
            int(np.round(N_DATA ** (1 / 2))) for t in range(1, T_max)
        ]  # determines how many points the adapted empirical map uses for each time step
        markov = 0  # 1 if markovian adapted empirical map should be used, 0 else

        dist = paths_to_dist_parallel(
            path1,
            path2,
            n_slices=self.n_slices,
            len_slices=self.len_slices,
            use_klist=1,
            k_list=klist,
            markov=markov,
            verbose=0,
            max_workers=4,
        )
        return dist
