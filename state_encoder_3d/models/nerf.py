import torch
from torch import nn
import torch.nn.functional as F


class PositionalEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int = 3,
        num_frequencies: int = 10,
        periodic_functions: list = [torch.sin, torch.cos],
        include_input: bool = True,
    ):
        super().__init__()

        embed_fns = []
        out_dim = 0

        if include_input:
            embed_fns.append(lambda x: x)
            out_dim += in_dim

        freq_bands = 2.0 ** torch.linspace(
            0.0, num_frequencies - 1, steps=num_frequencies
        )
        for freq in freq_bands:
            for p_fn in periodic_functions:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += in_dim

        self._embed_fns = embed_fns
        self.out_dim = out_dim

    def forward(self, inputs):
        return torch.cat([fn(inputs) for fn in self._embed_fns], dim=-1)


class LatentNeRF(nn.Module):
    """
    Adopted from '3D Scene Representation for Visuomotor Control'.
    """

    def __init__(self, D=8, W=256, input_dim=3, latent_ch=256, output_ch=4):
        super().__init__()

        self._positional_encoder = PositionalEncoder(in_dim=input_dim)

        mlp_input_dim = self._positional_encoder.out_dim + latent_ch
        self._layers = nn.ModuleList(
            [nn.Linear(input_dim, W)]
            + [
                nn.Linear(W, W) if i != 4 else nn.Linear(W + mlp_input_dim, W)
                for i in range(D - 1)
            ]
            + [nn.Linear(W, output_ch)]
        )

    def forward(self, latent: torch.Tensor, coordinate: torch.Tensor) -> torch.Tensor:
        encoded_coord = self._positional_encoder(coordinate)
        x = torch.cat([latent, encoded_coord], dim=-1)
        for i, layer in enumerate(self._layers):
            x = layer(x)
            x = F.relu(x, inplace=True)
            if i == 4:  # Skip connection
                x = torch.cat([torch.cat([latent, encoded_coord], dim=-1), x], dim=-1)
        return x
