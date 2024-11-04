from abc import ABC, abstractmethod
import os
from copy import deepcopy

import torch
from torch.nn import Module

from tsvae.base import BaseConfig
from tsvae.dataset.base import DatasetOutput
from tsvae.models.utils.output import ModelOutput


class BaseModel(Module):
    r"""
    What is important is that is communicating with trainer. Forward get data from trainer and output a dict which contains loss. The loss will be used in optimizer in the trainer.
    """
    model_config: BaseConfig

    def __init__(self, model_config: BaseConfig):
        super().__init__()
        self.model_config = model_config
        self.model_name = "BaseModel"
        self.device = None  # Do we want to enforce that all parameter in the same device?

    def forward(self, inputs: DatasetOutput, **kwargs) -> ModelOutput:
        r"""
        By using the forward of nn.Module we leverage their error instead of NotImplementedError
        """
        return super().forward()
    

    def generation(self, n_sample: int, device, **kwargs):
        raise NotImplementedError()

    def load_from_folder(self, dir_path):
        model_weights = BaseModel._load_model_weights_from_folder(dir_path)
        self.load_state_dict(model_weights)

    def save(self, dir_path: str):
        model_dict = {"model_state_dict": deepcopy(self.state_dict())}
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        # TODO: unify this two saving to saving obj
        self.model_config.save_json(dir_path, "model_config")
        torch.save(model_dict, os.path.join(dir_path, "model.pt"))

    def update(self):
        r"""Method that allows model update during the training (at the end of a training epoch)

        If needed, this method must be implemented in a child class.

        By default, it does nothing.
        """
        pass

    @classmethod
    def _load_model_weights_from_folder(cls, dir_path):
        file_list = os.listdir(dir_path)

        if "model.pt" not in file_list:
            raise FileNotFoundError(f"Missing model weights file ('model.pt') file in" f"{dir_path}... Cannot perform model building.")

        path_to_model_weights = os.path.join(dir_path, "model.pt")

        try:
            model_weights = torch.load(path_to_model_weights, map_location="cpu",weights_only=True)

        except RuntimeError:
            RuntimeError("Enable to load model weights. Ensure they are saves in a '.pt' format.")

        if "model_state_dict" not in model_weights.keys():
            raise KeyError("Model state dict is not available in 'model.pt' file. Got keys:" f"{model_weights.keys()}")

        model_weights = model_weights["model_state_dict"]

        return model_weights
