import logging
import os
import sys
from os import path as pt
from pathlib import Path

import ml_collections
import yaml

from tsvae.dataset.data_pipeline import DataPipeline
from tsvae.models.network_pipeline import NetworkPipeline
from tsvae.trainers.base_trainer_config import BaseTrainerConfig
from tsvae.trainers.training_pipeline import TrainingPipeline
from tsvae.utils.random_utils import set_seed


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Create handlers
c_handler = logging.StreamHandler()
c_handler.setLevel(logging.INFO)
logger.addHandler(c_handler)


def get_output_dir(config):
    output_dir = (
        config.base_output_dir
        + f"/results/{config.dataset}_timestep_{config.n_timestep}/model_{config.model}_De_{config.decoder}_En_{config.encoder}_Prior_{config.prior}_Con_{config.conditioner}_Dis_{config.discriminator}_comment_{config.comment}"
    )
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def update_config(exp_config, new_config):
    exp_config.update([(k, v) for k, v in new_config.items() if v is not None])
    exp_config = ml_collections.ConfigDict(exp_config)
    return exp_config


class ExperimentPipeline:
    def __init__(self, exp_config, base_output_dir: str = None, new_config: dict = {}) -> None:
        r"""
        exp_config: experiment configuration
        base_output_dir: the folder that you put all experiment result in. In this folder, the experiment pipeline will create a folder called result to save results and a folder called wandb to save the wandb result, if wandb = True.
        """

        exp_config = update_config(exp_config, new_config)

        exp_config.base_output_dir = base_output_dir
        exp_config.output_dir = get_output_dir(exp_config)
        logger.info(f"Experiment results saved to {exp_config.output_dir}")

        logger.info(f"Saving experiment config to {exp_config.output_dir}")
        config_file_path = pt.join(exp_config.output_dir, "exp_config.yaml")
        with open(config_file_path, "w") as outfile:
            yaml.dump(exp_config, outfile, default_flow_style=False)
        logger.info(exp_config)
        self.exp_config = exp_config

        # Generating data
        logger.info(f"Setting ramdom seed: {exp_config.seed}")
        set_seed(exp_config.seed)

        logger.info(f"Loading dataset: {exp_config.dataset}")
        data_pipeline = DataPipeline()
        train_dataset, eval_dataset = data_pipeline(exp_config)
        self.train_dataset = train_dataset

        # Loading network
        logger.info("Load networks:")
        network_pipeline = NetworkPipeline()
        model = network_pipeline(exp_config)
        logger.info(f"{model}")
        self.model = model

        # Loading trainer
        training_config = BaseTrainerConfig(
            output_dir=exp_config.output_dir,
            learning_rate=exp_config.lr,
            per_device_train_batch_size=exp_config.train_batch_size,
            per_device_eval_batch_size=exp_config.eval_batch_size,
            optimizer_cls=exp_config.optimizer,
            optimizer_params=None,
            scheduler_cls=None,
            scheduler_params=None,
            steps_saving=exp_config.steps_saving,
            steps_predict=exp_config.steps_predict,
            seed=exp_config.seed,
            num_epochs=exp_config.epochs,
            wandb_callback=exp_config.wandb,
            wandb_output_dir=exp_config.base_output_dir + "/wandb",
            ploter=exp_config.get("ploter", "path"),
        )

        self.train_pipeline = TrainingPipeline(model=model, training_config=training_config, exp_config=exp_config)

        # log_output_dir is not accessible through pipeline
        self.trainer = self.train_pipeline(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            device_name=exp_config.device_name,
        )

    def train(self):
        self.trainer.train(log_output=True)
