# import argparse
# import logging
# import os
# import sys
# from os import path as pt
# from pathlib import Path

# import ml_collections
# import yaml

# from experiments.experiment_utils import get_output_dir
# from tsvae.dataset.data_pipeline import DataPipeline
# from tsvae.models.network_pipeline import NetworkPipeline
# from tsvae.trainers.base_trainer_config import BaseTrainerConfig
# from tsvae.trainers.training_pipeline import TrainingPipeline
# from tsvae.utils.random_utils import set_seed

# base_dir = str(Path(pt.abspath(__file__)).parent.parent)
# src_dir = base_dir + "/src"
# sys.path.append(src_dir)


# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
# # Create handlers
# c_handler = logging.StreamHandler()
# c_handler.setLevel(logging.INFO)
# logger.addHandler(c_handler)


# def update_config(exp_config, new_config):
#     exp_config.update([(k, v) for k, v in new_config.items() if v is not None])
#     return exp_config


# def main(exp_config_path: str = None, base_output_dir: str = None, args=None):
#     # Setting up experiment configuration
#     logger.info(f"experiment config loaded from {exp_config_path}")
#     with open(exp_config_path) as file:
#         exp_config = ml_collections.ConfigDict(yaml.safe_load(file))
#     exp_config = update_config(exp_config, vars(args))


# def experiment_pipeline(exp_config, base_output_dir: str = None):
#     r"""
#     exp_config: experiment configuration
#     base_output_dir: the folder that you put all experiment result in. In this folder, the experiment pipeline will create a folder called result to save results and a folder called wandb to save the wandb result, if wandb = True.
#     """
#     exp_config.base_output_dir = base_output_dir
#     exp_config.output_dir = get_output_dir(exp_config)
#     logger.info(f"Experiment results saved to {exp_config.output_dir}")

#     logger.info(f"Saving experiment config to {exp_config.output_dir}")
#     config_file_path = pt.join(exp_config.output_dir, "exp_config.yaml")
#     with open(config_file_path, "w") as outfile:
#         yaml.dump(exp_config, outfile, default_flow_style=False)
#     logger.info(exp_config)

#     # Generating data
#     logger.info(f"Setting ramdom seed: {exp_config.seed}")
#     set_seed(exp_config.seed)

#     logger.info(f"Loading dataset: {exp_config.dataset}")
#     data_pipeline = DataPipeline()
#     train_dataset, eval_dataset = data_pipeline(exp_config)

#     # Loading network
#     logger.info("Load networks:")
#     network_pipeline = NetworkPipeline()
#     model = network_pipeline(exp_config)
#     logger.info(f"{model}")

#     # Loading trainer
#     training_config = BaseTrainerConfig(
#         output_dir=exp_config.output_dir,
#         learning_rate=exp_config.lr,
#         per_device_train_batch_size=exp_config.train_batch_size,
#         per_device_eval_batch_size=exp_config.eval_batch_size,
#         optimizer_cls=exp_config.optimizer,
#         optimizer_params=None,
#         scheduler_cls=None,
#         scheduler_params=None,
#         steps_saving=exp_config.steps_saving,
#         steps_predict=exp_config.steps_predict,
#         seed=exp_config.seed,
#         num_epochs=exp_config.epochs,
#         wandb_callback=exp_config.wandb,
#         wandb_output_dir=exp_config.base_output_dir + "/wandb",
#     )

#     train_pipeline = TrainingPipeline(model=model, training_config=training_config, exp_config=exp_config)

#     # log_output_dir is not accessible through pipeline
#     trainer = train_pipeline(
#         train_dataset=train_dataset,
#         eval_dataset=eval_dataset,
#         device_name=exp_config.device_name,
#     )

#     # train_pipeline.train(log_output=True)
