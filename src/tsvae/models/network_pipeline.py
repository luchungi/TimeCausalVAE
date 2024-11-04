from torch import relu

from tsvae.base import BasePipeline
from tsvae.models.conditioner.id_conditioner import IdentityConditioner
from tsvae.models.decoder.mlp_decoder import IdDecoder
from tsvae.models.decoder.neuralsde_decoder import CRSigDecoder
from tsvae.models.encoder.mlp_encoder import IdEncoder
from tsvae.models.prior.gaussian import GaussianPrior
from tsvae.models.prior.realnvp import FlowPrior
from tsvae.utils.logger_utils import get_console_logger

logger = get_console_logger(__name__)


class NetworkPipeline(BasePipeline):

    def __init__(
        self,
    ):
        pass

    def __call__(self, exp_config, **kwargs):
        conditioner = self._get_conditioner(exp_config, **kwargs)
        encoder = self._get_encoder(exp_config, conditioner, **kwargs)
        decoder = self._get_decoder(exp_config, conditioner, **kwargs)
        prior = self._get_prior(exp_config, **kwargs)
        model = self._get_model(encoder, decoder, prior, exp_config, **kwargs)
        return model

    def _get_conditioner(self, exp_config, **kwargs):
        if exp_config.conditioner == "Id":
            conditioner = IdentityConditioner()
        else:
            raise Exception("No such conditioner")
        return conditioner

    def _get_encoder(self, exp_config, conditioner=None, **kwargs):
        if exp_config.encoder == "MLP":
            from tsvae.models.encoder.mlp_encoder import MLPEncoder

            encoder = MLPEncoder(
                exp_config.data_dim,
                exp_config.data_length,
                exp_config.latent_dim,
                exp_config.latent_length,
                exp_config.E_hidden_dim,
            )

        elif exp_config.encoder == "CMLP":
            from tsvae.models.encoder.mlp_encoder import CMLPEncoder

            encoder = CMLPEncoder(
                exp_config.data_dim + exp_config.condition_dim,
                exp_config.data_length,
                exp_config.latent_dim,
                exp_config.latent_length,
                exp_config.E_hidden_dim,
                exp_config.condition_dim,
                conditioner,
            )

        elif exp_config.encoder == "LSTM":
            from tsvae.models.encoder.lstm_encoder import LSTMEncoder

            encoder = LSTMEncoder(
                exp_config.data_dim,
                exp_config.data_length,
                exp_config.latent_dim,
                exp_config.latent_length,
                exp_config.E_hidden_dim,
                exp_config.E_num_layers,
            )
        elif exp_config.encoder == "CLSTM":
            from tsvae.models.encoder.lstm_encoder import CLSTMEncoder

            encoder = CLSTMEncoder(
                exp_config.data_dim + exp_config.condition_dim,
                exp_config.data_length,
                exp_config.latent_dim,
                exp_config.latent_length,
                exp_config.E_hidden_dim,
                exp_config.E_num_layers,
                exp_config.condition_dim,
                conditioner,
            )
        elif exp_config.encoder == "CLSTMRes":
            from tsvae.models.encoder.lstm_encoder import CLSTMResEncoder

            encoder = CLSTMResEncoder(
                exp_config.data_dim + exp_config.condition_dim,
                exp_config.data_length,
                exp_config.latent_dim,
                exp_config.latent_length,
                exp_config.E_hidden_dim,
                exp_config.E_num_layers,
                exp_config.condition_dim,
                conditioner,
            )

        elif exp_config.encoder == "IdEncoder":
            encoder = IdEncoder()
        else:
            raise Exception("No such encoder")
        return encoder

    def _get_decoder(self, exp_config, conditioner=None, **kwargs):
        # Decoder
        if exp_config.decoder == "MLP":
            from tsvae.models.decoder.mlp_decoder import MLPDecoder

            decoder = MLPDecoder(
                exp_config.data_dim,
                exp_config.data_length,
                exp_config.latent_dim,
                exp_config.latent_length,
                exp_config.D_hidden_dim,
            )
        elif exp_config.decoder == "CMLP":
            from tsvae.models.decoder.mlp_decoder import CMLPDecoder

            decoder = CMLPDecoder(
                exp_config.data_dim,
                exp_config.data_length,
                exp_config.latent_dim + exp_config.condition_dim,
                exp_config.latent_length,
                exp_config.D_hidden_dim,
                exp_config.condition_dim,
                conditioner,
            )
        elif exp_config.decoder == "CAddMLP":
            from tsvae.models.decoder.mlp_decoder import CAddMLPDecoder

            decoder = CAddMLPDecoder(
                exp_config.data_dim,
                exp_config.data_length,
                exp_config.latent_dim,
                exp_config.latent_length,
                exp_config.D_hidden_dim,
                exp_config.condition_dim,
                conditioner,
            )

        elif exp_config.decoder == "LSTM":
            from tsvae.models.decoder.lstm_decoder import LSTMDecoder

            decoder = LSTMDecoder(
                exp_config.data_dim,
                exp_config.data_length,
                exp_config.latent_dim,
                exp_config.latent_length,
                exp_config.D_hidden_dim,
                exp_config.D_num_layers,
            )
        elif exp_config.decoder == "LSTMRes":
            from tsvae.models.decoder.lstm_decoder import LSTMResDecoder

            decoder = LSTMResDecoder(
                exp_config.data_dim,
                exp_config.data_length,
                exp_config.latent_dim,
                exp_config.latent_length,
                exp_config.D_hidden_dim,
                exp_config.D_num_layers,
            )
        elif exp_config.decoder == "CLSTMRes":
            from tsvae.models.decoder.lstm_decoder import CLSTMResDecoder

            decoder = CLSTMResDecoder(
                exp_config.data_dim,
                exp_config.data_length,
                exp_config.latent_dim + exp_config.condition_dim,
                exp_config.latent_length,
                exp_config.D_hidden_dim,
                exp_config.D_num_layers,
                exp_config.condition_dim,
                conditioner,
            )
        elif exp_config.decoder == "CRSigDecoder":

            decoder = CRSigDecoder(
                n_lag=exp_config.latent_length,
                input_dim=5,
                output_dim=exp_config.data_dim,
                reservoir_dim=50,
                brownian_dim=exp_config.latent_dim + exp_config.condition_dim,
                activation=relu,
                conditioner=conditioner,
                condition_dim=exp_config.condition_dim,
            )
        elif exp_config.decoder == "IdDecoder":

            decoder = IdDecoder()
        else:
            raise Exception("No such decoder")
        return decoder

    def _get_prior(self, exp_config, **kwargs):
        # Prior
        if exp_config.prior == "RealNVP":
            prior = FlowPrior(
                num_flows=exp_config.P_num_flows,
                latent_dim=exp_config.latent_dim * exp_config.latent_length,
                hidden_dim=exp_config.P_hidden_dim,
            )
        elif exp_config.prior == "Gaussian":
            prior = GaussianPrior(dim=exp_config.latent_dim * exp_config.latent_length)
        else:
            raise Exception("No such prior")

        return prior

    def _get_model(self, encoder, decoder, prior, exp_config, **kwargs):
        if exp_config.model == "VAE":
            from tsvae.models.vae import VAEConfig

            model_config = VAEConfig(
                data_dim=exp_config.data_dim,
                data_length=exp_config.data_length,
                latent_length=exp_config.latent_length,
                latent_dim=exp_config.latent_dim,
                reconstruction_loss="l1",
                transform=exp_config.transform,
                inv_transform=exp_config.inv_transform,
            )
            from tsvae.models.vae import VAE

            model = VAE(model_config=model_config, encoder=encoder, decoder=decoder, prior=prior)
        elif exp_config.model == "BetaCVAE":
            from tsvae.models.betavae import BetaVAEConfig

            model_config = BetaVAEConfig(
                data_dim=exp_config.data_dim,
                data_length=exp_config.data_length,
                latent_length=exp_config.latent_length,
                latent_dim=exp_config.latent_dim,
                reconstruction_loss="l1",
                beta=exp_config.beta,
                transform=exp_config.transform,
                inv_transform=exp_config.inv_transform,
            )
            from tsvae.models.betavae import BetaCVAE

            model = BetaCVAE(model_config=model_config, encoder=encoder, decoder=decoder, prior=prior)
        elif exp_config.model == "InfoCVAE":
            from tsvae.models.infovae import InfoCVAE, InfoCVAEConfig

            model_config = InfoCVAEConfig(
                data_dim=exp_config.data_dim,
                data_length=exp_config.data_length,
                latent_length=exp_config.latent_length,
                latent_dim=exp_config.latent_dim,
                reconstruction_loss="l1",
                beta=exp_config.beta,
                alpha=exp_config.alpha,
                transform=exp_config.transform,
                inv_transform=exp_config.inv_transform,
            )
            model = InfoCVAE(model_config=model_config, encoder=encoder, decoder=decoder, prior=prior)
        else:
            raise Exception("No such model")

        return model
