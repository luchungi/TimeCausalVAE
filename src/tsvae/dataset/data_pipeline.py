import matplotlib.pyplot as plt
import torch

from tsvae.base import BasePipeline
from tsvae.dataset.anm import ANMDataset
from tsvae.dataset.base import BaseDataset
from tsvae.dataset.blackscholes import BlackScholes2Dataset, BlackScholesDataset
from tsvae.dataset.pdv import PDVPriceFeatureDataset
from tsvae.dataset.stochasticvol import HestonDataset
from tsvae.dataset.stock import LogrDataset, SP500VIXDataset
from tsvae.utils.logger_utils import get_console_logger
from tsvae.dataset.toy2d import MixMultiVariateNormal, CheckerBoard, Spiral
from tsvae.utils.visualization_utils import visualize_data_2d

logger = get_console_logger(__name__)


class DataPipeline(BasePipeline):
    r"""
    Input:
        exp_config
    Return:
        train_dataset
        eval_dataset

    """

    def __init__(
        self,
    ):
        self.base_dataset = None

    def __call__(self, exp_config, **kwargs):
        train_data, train_labels = self._get_data_label(exp_config, use="train", **kwargs)
        train_dataset = BaseDataset(train_data, train_labels)
        eval_data, eval_labels = self._get_data_label(exp_config, use="eval", **kwargs)
        eval_dataset = BaseDataset(eval_data, eval_labels)
        return train_dataset, eval_dataset

    def _visualize_path(self, paths, dim=0):
        plt.plot(
            paths[:1000, :, dim].numpy().T,
            alpha=0.3,
            marker="o",
            linewidth=1,
            markersize=1,
        )
        plt.show()

    def _get_data_label(self, exp_config, use=None, **kwargs):
        if exp_config.dataset == "BSprice":
            dataset = BlackScholesDataset(exp_config.n_sample, exp_config.n_timestep - 1, **kwargs)
            data = dataset.data
            labels = torch.ones([data.shape[0], 1])
        elif exp_config.dataset == "BS2price":
            dataset = BlackScholes2Dataset(exp_config.n_sample, exp_config.n_timestep - 1, rho=exp_config.rho, **kwargs)
            data = dataset.data
            labels = torch.ones([data.shape[0], 1])
        elif exp_config.dataset == "Hestonprice":
            dataset = HestonDataset(exp_config.n_sample, exp_config.n_timestep - 1, **kwargs)
            data = dataset.data
            labels = torch.ones([data.shape[0], 1])
        elif exp_config.dataset == "PDVPriceConFeature":
            dataset = PDVPriceFeatureDataset(exp_config.n_sample, exp_config.n_timestep, **kwargs)
            data = dataset.data
            labels = dataset.labels
        elif exp_config.dataset == "SP500VIX":
            dataset = SP500VIXDataset(
                exp_config.n_sample,
                exp_config.n_timestep,
                base_data_dir=exp_config.base_data_dir,
            )
            data = dataset.data
            labels = dataset.labels
        elif exp_config.dataset == "Logr":
            dataset = LogrDataset(
                exp_config.n_sample,
                base_data_dir=exp_config.base_data_dir,
            )
            data = dataset.data
            labels = dataset.labels
        elif exp_config.dataset == "ANM":
            dataset = ANMDataset(exp_config.n_sample, **kwargs)
            data = dataset.data
            labels = dataset.labels
        elif exp_config.dataset == "GM":
            dataset = MixMultiVariateNormal(exp_config.n_sample, **kwargs)
            data = dataset.sample().view(-1, 1, 2)
            labels = torch.ones([data.shape[0], 1])
        elif exp_config.dataset == "Board":
            dataset = CheckerBoard(exp_config.n_sample, **kwargs)
            data = dataset.sample().view(-1, 1, 2)
            labels = torch.ones([data.shape[0], 1])
        elif exp_config.dataset == "Spiral":
            dataset = Spiral(exp_config.n_sample, **kwargs)
            data = dataset.sample().view(-1, 1, 2)
            labels = torch.ones([data.shape[0], 1])
        else:
            raise ValueError("No such dataset name")
        if self.base_dataset is None:
            self.base_dataset = dataset
            logger.info("Base dataset initialized")
        return data, labels
