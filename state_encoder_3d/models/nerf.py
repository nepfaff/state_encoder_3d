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

    def __init__(
        self,
        D: int = 8,
        W: int = 256,
        input_dim: int = 3,
        latent_ch: int = 256,
        output_ch: int = 4,
    ):
        super().__init__()

        self._positional_encoder = PositionalEncoder(in_dim=input_dim)

        mlp_input_dim = self._positional_encoder.out_dim + latent_ch
        self._layers = nn.ModuleList(
            [nn.Linear(mlp_input_dim, W)]
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
            # TODO: The final output shouldn't be activated by ReLU
            x = F.relu(x, inplace=True)
            if i == 4:  # Skip connection
                x = torch.cat([torch.cat([latent, encoded_coord], dim=-1), x], dim=-1)
        rad = F.sigmoid(x[..., :3])
        sigma = F.relu(x[..., 3])
        return rad, sigma


# class LatentNeRFNerfaccWrapper:
#     """A wrapper around LatentNeRF to make it work with Nerfacc."""

#     def __init__(self, ray_origins: torch.Tensor, ray_directions: torch.Tensor):
#         self._ray_origins = ray_origins
#         self._ray_directions = ray_directions

#     def simga_fn(
#         self,
#         latent: torch.Tensor,
#         t_starts: torch.Tensor,
#         t_ends: torch.Tensor,
#         ray_indices: torch.Tensor,
#     ) -> torch.Tensor:
#         t_origins = rays_o[ray_indices]  # (n_samples, 3)
#         t_dirs = rays_d[ray_indices]  # (n_samples, 3)
#         positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
#         sigmas = radiance_field.query_density(positions)

#         x = self.forward(latent, coordinate)
#         return x[..., -1]

#     def rgb_sigma_fn(
#         self, latent: torch.Tensor, coordinate: torch.Tensor
#     ) -> torch.Tensor:
#         x = self.forward(latent, coordinate)
#         return x
