import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from tsvae.dataset.base import DatasetOutput
from tsvae.utils.visualization_utils import visualize_data


# Compare (visualize) paths
def compare_path(x_real, x_fake, titles=None, file_path=None, return_figax=False, dim=0, plot_size=1000):
    fig, ax = plt.subplots(1, 2, figsize=[12, 4], sharex=True, sharey=True)
    ax[0].plot(
        x_real[:plot_size][..., dim].numpy().T,
        alpha=0.3,
        marker="o",
        linewidth=1,
        markersize=1,
    )
    ax[1].plot(
        x_fake[:plot_size][..., dim].numpy().T,
        alpha=0.3,
        marker="o",
        linewidth=1,
        markersize=1,
    )
    if titles:
        ax[0].set_title(titles[0])
        ax[1].set_title(titles[1])

    for i in range(2):
        ax[i].set_xlabel("Time")
        ax[i].set_ylabel("Prices")
    if return_figax:
        return fig, ax
    if file_path is None:
        plt.show()
    else:
        plt.savefig(file_path, bbox_inches="tight")
    plt.close()


# Compare Marginal
def compare_marginal_hist(real_data, fake_data, n_time_slice=4, file_path=None):
    time_range = np.linspace(11, real_data.shape[1] - 1, n_time_slice, dtype=int)
    n_cols = n_time_slice
    fig, ax = plt.subplots(1, n_cols, figsize=[12, 4], sharey=True, sharex=True)
    bins = np.linspace(0, 5, 50)
    for i in range(n_cols):
        axi = ax.flat[i]
        n = time_range[i]
        axi.hist(real_data[:, n, 0], bins=bins, alpha=0.5, label="real")
        axi.hist(fake_data[:, n, 0], bins=bins, alpha=0.3, label="fake")
        axi.set_title(f"Time: {n}")
        axi.legend()
        # axi.set_xlabel("Marginal prices")
    if file_path is None:
        plt.show()
    else:
        plt.savefig(file_path, bbox_inches="tight")
        plt.close(fig)


# Marginal Correlation
def compare_corr_marginal(z_real, z_fake, file_path=None):
    fig, ax = plt.subplots(1, 2, figsize=[12, 4], sharex=True, sharey=True)
    df_real = pd.DataFrame(z_real[..., 0])
    df_fake = pd.DataFrame(z_fake[..., 0])

    im = ax[0].matshow(df_real.corr(), cmap="Spectral")
    ax[0].set_title("real")
    im = ax[1].matshow(df_fake.corr(), cmap="Spectral")
    ax[1].set_title("fake")
    fig.colorbar(im, ax=ax.ravel().tolist())

    if file_path is None:
        plt.show()
    else:
        plt.savefig(file_path, bbox_inches="tight")
        plt.close(fig)


def compare_corr(real_data, fake_data, verbose=False, file_path=None):

    real_logr = real_data.log().diff(dim=1)
    fake_logr = fake_data.log().diff(dim=1)
    d = real_logr.shape[-1]
    real_corr = real_logr.view(-1, d).T.corrcoef()
    fake_corr = fake_logr.view(-1, d).T.corrcoef()
    if verbose:
        print("Real data:")
        print(real_corr)
        print("Fake data:")
        print(fake_corr)
    return real_corr, fake_corr


def generate_noised_paths(path, model):
    paths = path.repeat(100, 1, 1)
    dataset_output = DatasetOutput(data=paths, labels=torch.zeros_like(paths[:, 0, :1]))
    with torch.no_grad():
        single_model_output = model(dataset_output)
        noised_paths = single_model_output["recon_x"]
    return paths, noised_paths


def compare_noised_paths(path, model, file_path=None):
    paths, noised_paths = generate_noised_paths(path, model)
    compare_path(paths, noised_paths, file_path=file_path)
