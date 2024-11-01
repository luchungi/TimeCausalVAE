from tsvae.models.conditioner.base_conditioner import BaseConditioner
import torch


class IdentityConditioner(BaseConditioner):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.net = torch.nn.Identity()

    def forward(self, x):
        return self.net(x)
