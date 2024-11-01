import os
from copy import deepcopy
from os import path as pt
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml
from tqdm import tqdm

from evaluations.eval_distances import SWD, SignatureMMD
from tsvae.models.utils.distances import GaussianMMD, GaussianMMD2
from tsvae.dataset.base import DatasetOutput
from tsvae.dataset.data_pipeline import DataPipeline
from tsvae.models import network_pipeline
from tsvae.utils.load_save_utils import load_obj, save_obj
from tsvae.utils.random_utils import set_seed
from tsvae.utils.visualization_utils import visualize_real_recon_fake


def base2final_model_dirs(base_model_dir):
    hyper_model_dirs = [f.path for f in os.scandir(base_model_dir) if f.is_dir()]
    final_model_dirs = [
        checkpoint_dir.path
        for model_dir in hyper_model_dirs
        for checkpoint_dir in os.scandir(model_dir)
        if checkpoint_dir.is_dir() and "final_model" in checkpoint_dir.path
    ]
    return final_model_dirs


def load_hyper_metric_from_folders(model_dirs, compute=False, plot_path=None, n_sample_test=5000, *args, **kwargs):
    base_data_dir = kwargs.get("base_data_dir")
    hyper_metric_dict = {}
    for model_dir in tqdm(model_dirs):
        model_evaluator = ModelEvaluator(model_dir, base_data_dir)
        if plot_path is not None or compute:
            real_data, fake_data, recon_data = model_evaluator.load_data(n_sample_test)

        if plot_path is not None:
            fig = visualize_real_recon_fake(real_data, recon_data, fake_data)
            file_path = pt.join(plot_path, Path(model_dir).parent.name + ".png")
            fig.suptitle(
                model_evaluator.hyper_label,
            )
            plt.savefig(file_path, bbox_inches="tight")
            plt.close(fig)

        if compute:
            hyper_metric = model_evaluator.compute_hyper_metric(real_data, fake_data)
            model_evaluator.save_hyper_metric(hyper_metric)
        else:
            hyper_metric = model_evaluator.load_hyper_metric()

        hyper_metric_dict[model_dir] = {
            "hyper_label": model_evaluator.hyper_label,
            "hyper_metric": hyper_metric,
        }

    return hyper_metric_dict


class ModelEvaluator:
    r"""
    A class with all information to evaluate a model
    """

    def __init__(self, model_dir=None, exp_config=None, model=None, base_data_dir=None, *args, **kwargs) -> None:
        if model_dir:
            self.exp_config = self.load_from_folder(model_dir, base_data_dir)
            self.load_model()
        else:
            self.exp_config = deepcopy(exp_config)
            self.model = model

        self.data_ppl = DataPipeline()
        self.autoload_hyper_label()

    def load_from_folder(self, model_dir, base_data_dir):
        self.model_dir = model_dir  # the folder contains weights
        self.hyper_model_dir = os.path.dirname(model_dir)  # the folder containing exp_config
        # self.base_model_dir = os.path.dirname(self.hyper_model_dir)
        self.exp_config_path = pt.join(self.hyper_model_dir, "exp_config.yaml")
        with open(self.exp_config_path) as file:
            exp_config = yaml.load(file, Loader=yaml.UnsafeLoader)
        # we need this to load custom data
        exp_config.base_data_dir = base_data_dir
        return exp_config

    def load_model(self):
        self.network_ppl = network_pipeline.NetworkPipeline()
        self.model = self.network_ppl(self.exp_config)
        self.model.load_from_folder(self.model_dir)

    def load_data(self, n_sample_test=5000, seed=0):
        if seed > 0:
            set_seed(seed)
        exp_config = deepcopy(self.exp_config)
        exp_config.n_sample = n_sample_test
        _, test_dataset = self.data_ppl(exp_config)

        dataset_output = DatasetOutput(data=test_dataset.data, labels=test_dataset.labels)

        with torch.no_grad():
            model_output = self.model(dataset_output)
            test_data = dataset_output["data"]
            recon_data = model_output["recon_x"]
            gen_data = self.model.generation(n_sample_test, c=dataset_output["labels"][:n_sample_test])

        return test_data, gen_data, recon_data

    def compute_hyper_metric(self, real_data, fake_data):
        hyper_metric = {}
        hyper_metric["mmd"] = GaussianMMD()(real_data, fake_data)
        hyper_metric["swd"] = SWD()(real_data, fake_data)
        # hyper_metric["esig"] = SignatureMMD()(real_data, fake_data)
        return hyper_metric

    def save_hyper_metric(self, hyper_metric):
        self.hyper_metric_path = pt.join(self.model_dir, "hyper_metric.pkl")
        save_obj(hyper_metric, self.hyper_metric_path)

    def load_hyper_metric(self):
        self.hyper_metric_path = pt.join(self.model_dir, "hyper_metric.pkl")
        hyper_metric = load_obj(self.hyper_metric_path)
        return hyper_metric

    def autoload_hyper_label(self):
        if self.exp_config.model == "BetaCVAE":
            self.hyper_label = f"{self.model.beta:.4f}"
        elif self.exp_config.model == "InfoCVAE":
            self.hyper_label = f"({self.model.alpha:.4f},{self.model.beta:.4f})"
