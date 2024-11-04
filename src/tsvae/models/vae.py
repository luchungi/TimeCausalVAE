from dataclasses import dataclass
from typing import Optional, Dict

import torch
from torch import Tensor

from tsvae.dataset.base import DatasetOutput
from tsvae.models.transform import get_transform
from tsvae.models.base import BaseConfig, BaseModel
from tsvae.models.decoder.base_decoder import BaseDecoder
from tsvae.models.encoder.base_encoder import BaseEncoder
from tsvae.models.prior.base_prior import BasePrior
from tsvae.models.prior.gaussian import GaussianPrior, entropy_normal
from tsvae.models.utils.loss import get_loss
from tsvae.models.utils.output import ModelOutput


@dataclass
class VAEConfig(BaseConfig):
    data_dim: int = 1
    data_length: int = 1
    latent_length: int = 1
    latent_dim: int = 1
    reconstruction_loss: str = "l1"

    transform: str = ""
    inv_transform: str = ""

    uses_default_encoder: bool = False
    uses_default_decoder: bool = False


class VAE(BaseModel):

    model_config: VAEConfig

    def __init__(
        self,
        model_config: VAEConfig,
        encoder: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None,
        prior: Optional[BasePrior] = None,
    ):
        super().__init__(model_config)
        self.model_config
        self.model_name = "VAE"
        self.recon_loss_func = get_loss(model_config.reconstruction_loss)

        self.encoder = encoder
        self.decoder = decoder
        self.prior = prior

        self.transform = get_transform(self.model_config.transform)
        self.inv_transform = get_transform(self.model_config.inv_transform)

    def forward(self, inputs: DatasetOutput, **kwargs):
        r"""
        return a ModelOutput which have both what you need for training (loss, metric) and what you need for sampling
        """
        x0 = inputs["data"]
        x = self.transform(x0)
        encoder_output = self.encoder(x)
        mu, log_var = encoder_output.embedding, encoder_output.log_covariance

        std = torch.exp(0.5 * log_var)
        z, eps = self._sample_gauss(mu, std)
        recon_x = self.decoder(z)["reconstruction"]
        recon_x0 = self.inv_transform(recon_x)

        loss_dict = self._loss_function(recon_x0, x0, mu, log_var, z)
        data_dict = {"recon_x": recon_x0, "z": z}
        output = ModelOutput({**loss_dict, **data_dict})
        return output

    def _loss_function(self, recon_x0: Tensor, x0: Tensor, mu: Tensor, log_var: Tensor, z: Tensor) -> Dict:
        r"""
        Recon + DKL
        DKL = E[log(posterior)] - E[log(prior] (expectation under posterior)
        """
        recon_loss = self.recon_loss_func(x0, recon_x0)
        posterior_term = entropy_normal(log_var)
        prior_term = self.prior.log_prob(z)
        kld_loss = posterior_term.mean() - prior_term.mean()
        total_loss = recon_loss + kld_loss

        loss_dict = {"recon_loss": recon_loss, "reg_loss": kld_loss, "loss": total_loss}
        return loss_dict

    def _sample_gauss(self, mu, std):
        # Reparametrization trick
        # Sample N(0, I)
        eps = torch.randn_like(std)
        return mu + eps * std, eps

    def generation(self, n_sample: int, **kwargs):
        z = self.prior.sample(n_sample, device=self.device)
        recon_x = self.decoder(z)["reconstruction"]
        recon_x0 = self.inv_transform(recon_x)
        return recon_x0


@dataclass
class CVAEConfig(VAEConfig):
    pass


class CVAE(VAE):
    model_config: CVAEConfig

    def __init__(
        self,
        model_config: CVAEConfig,
        encoder: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None,
        prior: Optional[BasePrior] = None,
    ):
        super().__init__(model_config, encoder, decoder, prior)
        self.model_name = "CVAE"

    def forward(self, inputs: DatasetOutput, **kwargs):
        r"""This is a conditional version of forward now"""
        x0 = inputs["data"]
        x = self.transform(x0)
        c = inputs["labels"]
        encoder_output = self.encoder(x, c)
        mu, log_var = encoder_output.embedding, encoder_output.log_covariance
        std = torch.exp(0.5 * log_var)
        z, eps = self._sample_gauss(mu, std)
        recon_x = self.decoder(z, c)["reconstruction"]
        recon_x0 = self.inv_transform(recon_x)
        loss_dict = self._loss_function(recon_x0, x0, mu, log_var, z)
        data_dict = {"recon_x": recon_x0, "z": z}
        output = ModelOutput({**loss_dict, **data_dict})
        return output

    def generation(self, n_sample: int, **kwargs):
        c = kwargs.pop("c")
        z = self.prior.sample(n_sample, device=self.device)
        recon_x = self.decoder(z, c)["reconstruction"]
        recon_x0 = self.inv_transform(recon_x)
        return recon_x0

    def _loss_function(self, recon_x0: Tensor, x0: Tensor, mu: Tensor, log_var: Tensor, z: Tensor) -> Dict:
        return super()._loss_function(recon_x0, x0, mu, log_var, z)
