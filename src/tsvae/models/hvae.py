from dataclasses import dataclass, field
from typing import Optional, List


from tsvae.models.utils.distances import GaussianMMD2
from tsvae.dataset.base import DatasetOutput
from tsvae.models.base import ModelOutput
from tsvae.models.betavae import BetaCVAE, BetaCVAEConfig
from tsvae.models.decoder.base_decoder import BaseDecoder
from tsvae.models.encoder.base_encoder import BaseEncoder
from tsvae.models.prior.base_prior import BasePrior
from tsvae.models.prior.gaussian import entropy_normal
from tsvae.models.vae import CVAE, VAE, CVAEConfig, VAEConfig
import torch
from torch import nn


class EncoderResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.seq = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))

    def forward(self, x):
        return x + 0.1 * self.seq(x)


class DecoderResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.seq = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))

    def forward(self, x):
        return x + 0.1 * self.seq(x)


class BatchNorm(nn.Module):
    def __init__(self, dim_hidden, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.norm = nn.BatchNorm1d(dim_hidden)

    def forward(self, x):
        return torch.swapaxes(self.norm(torch.swapaxes(x, 1, 2)), 1, 2)


class Swish(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sigmoid(x)


class MLPBlock(nn.Module):
    def __init__(self, dim_in, dim_out, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        dim_hidden = 512  # could make it hyper later
        self.seq = nn.Sequential(nn.Linear(dim_in, dim_hidden), Swish(), BatchNorm(dim_hidden), nn.Linear(dim_hidden, dim_out))

    def forward(self, x):
        return self.seq(x)


class EncoderBlock(nn.Module):
    def __init__(self, dims):
        super().__init__()
        modules = []
        for i in range(len(dims) - 1):
            modules.append(MLPBlock(dims[i], dims[i + 1]))
        self.modules_list = nn.ModuleList(modules)

    def forward(self, x):
        for module in self.modules_list:
            x = module(x)
        return x


class HEncoder(BaseEncoder):
    def __init__(self, x_dim, z_dims, n_level, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.x_dim = x_dim
        self.z_dims = z_dims
        self.n_level = n_level

        # z_dims[i] = dim of z_i
        self.enc_blocks = nn.ModuleList([EncoderBlock([x_dim, z_dims[-1]])])

    def forward(self, x):
        xs = []  # [x_h,...,x_0]
        for e in self.enc_blocks:
            x = e(x)
            xs.append(x)
        return xs[::-1]  # xs[:-1][::-1] is [x_0, x_1, ...,x_h]


class DecoderBlock(nn.Module):
    def __init__(self, dims):
        super().__init__()
        modules = []
        for i in range(len(dims) - 1):
            modules.append(MLPBlock(dims[i], dims[i + 1]))
        self.modules_list = nn.ModuleList(modules)

    def forward(self, x):
        for module in self.modules_list:
            x = module(x)
        return x


class HDecoder(BaseDecoder):
    def __init__(self, x_dim, z_dims, n_level, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.x_dim = x_dim
        self.z_dims = z_dims
        self.n_level = n_level

        # self.dec_res_blocks = nn.ModuleList([DecoderResBlock(h_dims[1]), DecoderResBlock(h_dims[2]), DecoderResBlock(h_dims[3])])

        # self.dec_blocks = nn.ModuleList(
        #     [
        #         DecoderBlock([z_dims[0] + h_dims[0], h_dims[1]]),
        #         DecoderBlock([z_dims[1] + h_dims[1], h_dims[2]]),
        #         DecoderBlock([z_dims[2] + h_dims[2], h_dims[3]]),
        #     ]
        # )

        self.con_q_x = DecoderBlock([z_dims[1], 2 * z_dims[0]])

        self.con_q_xz = nn.ModuleList([DecoderBlock([z_dims[1], 2 * z_dims[1]])])
        self.con_p_z = nn.ModuleList([DecoderBlock([z_dims[0], 2 * z_dims[1]])])

        self.rec_net = DecoderBlock([z_dims[1], x_dim])

    def forward(self, xs):

        mu0, log_var0 = self.con_q_x(xs[0]).chunk(2, dim=-1)
        z0 = reparameterize(mu0, log_var0)
        # h0 = torch.zeros_like(z0)
        kld0 = kl(mu0, log_var0)

        zs = [z0]  # will be length n_level
        # hs = [h0]  # will be length n_level
        klds = [kld0]  # will be length n_level

        for i in range(self.n_level - 1):
            # compute next h
            # zh = torch.cat([zs[i], hs[i]], dim=-1)
            # h = self.dec_res_blocks[i](self.dec_blocks[i](zh))
            # hs.append(h)

            # p(z_l|z_<l)
            mu, log_var = self.con_p_z[i](zs[i]).chunk(2, dim=-1)

            # q(z_l|z_<l,x)
            # xh = torch.cat([h, xs[i + 1]], dim=-1)
            delta_mu, delta_log_var = self.con_q_xz[i](xs[i]).chunk(2, dim=-1)
            z = reparameterize(mu + delta_mu, log_var + delta_log_var)
            zs.append(z)

            # DKL(q(z_l|z_<l,x) | p(z_l|z_<l))
            klds.append(kl_2(delta_mu, delta_log_var, mu, log_var))
        # z_L -> h_L -> x_hat
        # zh = torch.cat([zs[-1], hs[-1]], dim=-1)
        # h = self.dec_res_blocks[-1](self.dec_blocks[-1](zh))
        recon_x = self.rec_net(zs[-1])
        return recon_x, klds, zs

    def sample(self, n_sample, device):
        mu0 = torch.zeros([n_sample, 1, self.z_dims[0]], device=device)
        log_var0 = torch.zeros([n_sample, 1, self.z_dims[0]], device=device)
        z0 = reparameterize(mu0, log_var0)
        zs = [z0]  # will be length n_level
        for i in range(self.n_level - 1):
            mu, log_var = self.con_p_z[i](zs[i]).chunk(2, dim=-1)  # p(z_l|z_<l)
            z = reparameterize(mu, log_var)
            zs.append(z)
        recon_x = self.rec_net(zs[-1])
        return recon_x, zs

    # def sample(self, n_sample):
    #     mu0 = torch.zeros([n_sample, self.z_dims[0]])
    #     log_var0 = torch.zeros([n_sample, self.z_dims[0]])
    #     z0 = reparameterize(mu0, log_var0)
    #     h0 = torch.zeros_like(z0)

    #     zs = [z0]  # will be length n_level
    #     hs = [h0]  # will be length n_level

    #     for i in range(self.n_level - 1):
    #         # compute next h
    #         zh = torch.cat([zs[i], hs[i]], dim=-1)
    #         h = self.dec_res_blocks[i](self.dec_blocks[i](zh))
    #         hs.append(h)

    #         # p(z_l|z_<l)
    #         mu, log_var = self.con_p_z[i](h).chunk(2, dim=-1)
    #         z = reparameterize(mu, log_var)
    #         zs.append(z)

    #     # z_L -> h_L -> x_hat
    #     zh = torch.cat([zs[-1], hs[-1]], dim=-1)
    #     h = self.dec_res_blocks[-1](self.dec_blocks[-1](zh))
    #     recon_x = self.rec_net(h)

    #     return recon_x, zs


@dataclass
class HVAEConfig(VAEConfig):
    x_dim: int = 1
    z_dims: List[int] = field(default_factory=list)


class HVAE(VAE):
    r"""
    InfoVAE includes WAE, AE, VAE, Beta-VAE
    """

    model_config: HVAEConfig

    def __init__(
        self,
        model_config: HVAEConfig,
        encoder: HEncoder,
        decoder: HDecoder,
        prior: BasePrior,
    ):
        super().__init__(model_config, encoder, decoder, prior)
        self.model_name = "HVAE"

    def forward(self, inputs: DatasetOutput, **kwargs):
        x_raw = inputs["data"]
        x = self.transform(x_raw)
        xs = self.encoder(x)
        recon_x, klds, zs = self.decoder(xs)
        loss = self.recon_loss_func(x, recon_x) + sum(klds)

        loss_dict = {"loss": loss, "klds": klds}
        data_dict = {"xs": xs, "zs": zs, "recon_x": recon_x}
        output = ModelOutput({**loss_dict, **data_dict})
        return output

    def sample(self, n_sample, device):
        return self.decoder.sample(n_sample, device)


def reparameterize(mu, log_sigma):
    return torch.randn_like(mu) * log_sigma.exp() + mu


def kl(mu, log_var):
    """
    kl loss with standard norm distribute
    :param mu:
    :param log_var:
    :return:
    """
    loss = -0.5 * torch.sum(torch.flatten(1 + log_var - mu**2 - torch.exp(log_var), start_dim=1), dim=1)
    return torch.mean(loss, dim=0)


def kl_2(delta_mu, delta_log_var, mu, log_var):
    var = torch.exp(log_var)
    delta_var = torch.exp(delta_log_var)

    loss = -0.5 * torch.sum(torch.flatten(1 + delta_log_var - delta_mu**2 / var - delta_var, start_dim=1), dim=1)
    return torch.mean(loss, dim=0)
