from dataclasses import dataclass
from typing import Dict, Optional

from torch import Tensor

from tsvae.models.decoder.base_decoder import BaseDecoder
from tsvae.models.encoder.base_encoder import BaseEncoder
from tsvae.models.prior.base_prior import BasePrior
from tsvae.models.prior.gaussian import entropy_normal
from tsvae.models.vae import CVAE, VAE, CVAEConfig, VAEConfig


@dataclass
class BetaVAEConfig(VAEConfig):
    beta: float = 1.0


class BetaVAE(VAE):
    model_config: BetaVAEConfig

    def __init__(
        self,
        model_config: BetaVAEConfig,
        encoder: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None,
        prior: Optional[BasePrior] = None,
    ):
        super().__init__(model_config, encoder, decoder, prior)
        self.model_name = "BetaVAE"
        self.beta = model_config.beta

    def _loss_function(self, recon_x0: Tensor, x0: Tensor, mu: Tensor, log_var: Tensor, z: Tensor) -> Dict:
        recon_loss = self.recon_loss_func(x0, recon_x0)
        posterior_term = entropy_normal(log_var)
        prior_term = self.prior.log_prob(z.flatten(start_dim=1))
        kld_loss = (posterior_term - prior_term).mean()

        total_loss = recon_loss + self.beta * kld_loss  # The only line different from VAE

        loss_dict = {"recon_loss": recon_loss, "reg_loss": kld_loss, "loss": total_loss}
        return loss_dict


@dataclass
class BetaCVAEConfig(CVAEConfig):
    beta: float = 1.0


class BetaCVAE(CVAE):
    model_config: BetaCVAEConfig

    def __init__(
        self,
        model_config: BetaCVAEConfig,
        encoder: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None,
        prior: Optional[BasePrior] = None,
    ):
        super().__init__(model_config, encoder, decoder, prior)
        self.model_name = "BetaCVAE"
        self.beta = model_config.beta

    def _loss_function(self, recon_x0: Tensor, x0: Tensor, mu: Tensor, log_var: Tensor, z: Tensor) -> Dict:
        recon_loss = self.recon_loss_func(x0, recon_x0)
        posterior_term = entropy_normal(log_var)
        prior_term = self.prior.log_prob(z.flatten(start_dim=1))
        kld_loss = (posterior_term - prior_term).mean()

        total_loss = recon_loss + self.beta * kld_loss  # The only line different from VAE

        loss_dict = {"recon_loss": recon_loss, "reg_loss": kld_loss, "loss": total_loss}
        return loss_dict
