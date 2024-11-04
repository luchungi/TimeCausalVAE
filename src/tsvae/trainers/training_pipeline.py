from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch

from tsvae.base import BasePipeline
from tsvae.models.vae import VAE
from tsvae.trainers.base_trainer import BaseTrainer
from tsvae.trainers.base_trainer_config import BaseTrainerConfig
from tsvae.utils.logger_utils import get_console_logger

logger = get_console_logger(__name__)


class TrainingPipeline(BasePipeline):

    def __init__(
        self,
        model: Optional[VAE],
        training_config: Optional[BaseTrainerConfig] = None,
        exp_config=None,
    ):

        if not isinstance(training_config, BaseTrainerConfig):
            raise AssertionError("A 'BaseTrainerConfig' " "is expected for the pipeline")

        self.model = model
        self.training_config = training_config
        self.exp_config = exp_config

    def __call__(
        self,
        train_dataset: Union[np.ndarray, torch.Tensor, torch.utils.data.Dataset],
        eval_dataset: Union[np.ndarray, torch.Tensor, torch.utils.data.Dataset] = None,
        device_name=None,
    ):

        if self.training_config.wandb_callback:
            from tsvae.trainers.training_callbacks import WandbCallback

            callbacks = []  # the TrainingPipeline expects a list of callbacks
            wandb_cb = WandbCallback()  # Build the callback
            # SetUp the callback
            name = Path(self.training_config.output_dir).name
            wandb_cb.setup(
                exp_config=self.exp_config,
                training_config=self.training_config,  # training config
                model_config=self.model.model_config,  # model config
                project_name="time-causal-vae",  # specify your wandb project
                entity_name="my_wandb_entity",  # specify your wandb entity  
            )
            callbacks.append(wandb_cb)  # Add it to the callbacks list
        else:
            callbacks = None

        trainer = BaseTrainer(
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            training_config=self.training_config,
            exp_config=self.exp_config,
            callbacks=callbacks,
            device_name=device_name,
        )

        self.trainer = trainer
        return trainer

    def train(self, log_output):
        self.trainer.train(log_output=log_output)
