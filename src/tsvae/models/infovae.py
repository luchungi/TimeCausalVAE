from dataclasses import dataclass
from typing import Optional

from torch import Tensor

from tsvae.models.decoder.base_decoder import BaseDecoder
from tsvae.models.encoder.base_encoder import BaseEncoder
from tsvae.models.prior.base_prior import BasePrior
from tsvae.models.prior.gaussian import entropy_normal
from tsvae.models.utils.distances import GaussianMMD2
from tsvae.models.vae import CVAE, VAE, CVAEConfig, VAEConfig


@dataclass
class InfoVAEConfig(VAEConfig):
    beta: float = 1.0
    alpha: float = 1.5


class InfoVAE(VAE):
    r"""
    InfoVAE includes WAE, AE, VAE, Beta-VAE
    """

    model_config: InfoVAEConfig

    def __init__(
        self,
        model_config: InfoVAEConfig,
        encoder: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None,
        prior: Optional[BasePrior] = None,
    ):
        super().__init__(model_config, encoder, decoder, prior)
        self.model_name = "InfoVAE"
        self.beta = model_config.beta
        self.alpha = model_config.alpha
        self.mmd2 = GaussianMMD2()

    def _loss_function(self, recon_x: Tensor, x: Tensor, mu: Tensor, log_var: Tensor, z: Tensor):
        recon_loss = self.recon_loss_func(x, recon_x)
        posterior_term = entropy_normal(log_var)
        prior_term = self.prior.log_prob(z.flatten(start_dim=1))
        kld_loss = (posterior_term - prior_term).mean()

        z_prior = self.prior.sample(len(z), device=z.device)
        mmd_loss = self.mmd2(z.flatten(start_dim=1), z_prior.flatten(start_dim=1))
        # alpha >= beta
        total_loss = recon_loss + self.beta * kld_loss + (self.alpha - self.beta) * mmd_loss

        loss_dict = {
            "recon_loss": recon_loss,
            "reg_loss": kld_loss,
            "loss": total_loss,
            "mmd_loss": mmd_loss,
        }
        return loss_dict


@dataclass
class InfoCVAEConfig(CVAEConfig):
    beta: float = 1.0
    alpha: float = 1.5


class InfoCVAE(CVAE):
    model_config: InfoCVAEConfig

    def __init__(
        self,
        model_config: InfoCVAEConfig,
        encoder: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None,
        prior: Optional[BasePrior] = None,
    ):
        super().__init__(model_config, encoder, decoder, prior)
        self.model_name = "InfoCVAE"
        self.beta = model_config.beta
        self.alpha = model_config.alpha
        self.mmd2 = GaussianMMD2()

    def _loss_function(self, recon_x: Tensor, x: Tensor, mu: Tensor, log_var: Tensor, z: Tensor):
        recon_loss = self.recon_loss_func(x, recon_x)
        posterior_term = entropy_normal(log_var)
        prior_term = self.prior.log_prob(z.flatten(start_dim=1))
        kld_loss = (posterior_term - prior_term).mean()

        z_prior = self.prior.sample(len(z), device=z.device)
        mmd_loss = self.mmd2(z.flatten(start_dim=1), z_prior.flatten(start_dim=1))
        # alpha >= beta
        total_loss = recon_loss + self.beta * kld_loss + (self.alpha - self.beta) * mmd_loss

        loss_dict = {
            "recon_loss": recon_loss,
            "reg_loss": kld_loss,
            "loss": total_loss,
            "mmd_loss": mmd_loss,
        }
        return loss_dict
