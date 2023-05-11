from torch import nn


class ReluMLP(nn.Module):
    def __init__(
        self, in_ch: int, out_ch: int, num_hidden_layers: int = 1, latent_dim: int = 256
    ):
        super().__init__()

        self._fc = nn.ModuleList(
            [
                nn.Linear(in_ch, latent_dim),
                nn.ReLU(inplace=True),
                *(
                    [nn.Linear(latent_dim, latent_dim), nn.ReLU(inplace=True)]
                    * num_hidden_layers
                ),
                nn.Linear(latent_dim, out_ch),
            ]
        )

    def forward(self, x):
        for layer in self._fc:
            x = layer(x)
        return x
