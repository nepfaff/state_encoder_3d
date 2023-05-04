from typing import Tuple, Optional

import torch
from torch import nn
import einops

from state_encoder_3d.utils import get_world_rays


def sample_points_along_rays(
    near_depth: float,
    far_depth: float,
    num_samples: int,
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    device: torch.device,
    randomize_sampling: bool = False,
):
    """
    Args:
        randomize_sampling: If true, inject uniform noise into the sample space to make
            the samples correspond to a continous distribution.
    """
    # Compute a linspace of num_samples depth values beetween near_depth and far_depth.
    z_vals = torch.linspace(near_depth, far_depth, num_samples, device=device)

    if randomize_sampling:
        noise = (
            torch.rand(z_vals.shape, device=device)
            * (far_depth - near_depth)
            / num_samples
        )
        z_vals += noise

    # Using the ray_origins, ray_directions, generate 3D points along
    # the camera rays according to the z_vals.
    pts = (
        ray_origins[..., None, :] + ray_directions[..., None, :] * z_vals[..., :, None]
    )

    return pts, z_vals


def volume_integral(
    z_vals: torch.Tensor, sigmas: torch.Tensor, radiances: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Compute the deltas in depth between the points.
    dists = torch.cat(
        [
            z_vals[..., 1:] - z_vals[..., :-1],
            torch.broadcast_to(
                torch.Tensor([1e10]).to(z_vals.device), z_vals[..., :1].shape
            ),
        ],
        dim=-1,
    )

    # Compute the alpha values from the densities and the dists.
    alpha = 1.0 - torch.exp(-torch.einsum("brzs, z -> brzs", sigmas, dists))

    # Compute the Ts from the alpha values. Use torch.cumprod.
    Ts = torch.cumprod(1.0 - alpha + 1e-10, -2)

    # Compute the weights from the Ts and the alphas.
    weights = alpha * Ts

    # Compute the pixel color as the weighted sum of the radiance values.
    rgb = torch.einsum("brzs, brzs -> brs", weights, radiances)

    # Compute the depths as the weighted sum of z_vals.
    depth_map = torch.einsum("brzs, z -> brs", weights, z_vals)

    return rgb, depth_map, weights


class VolumeRenderer(nn.Module):
    def __init__(
        self, near: float, far: float, n_samples: int = 64, white_back: bool = False
    ):
        super().__init__()
        self._near = near
        self._far = far
        self._n_samples = n_samples
        self._white_back = white_back

    def forward(
        self,
        cam2world,
        intrinsics,
        xy_pix,
        radiance_field: nn.Module,
        radiance_field_input: Optional[torch.Tensor] = None,
        randomize_sampling: bool = False,
        ray_batch_chunk: int = 1024 * 32,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Takes as inputs ray origins and directions - samples points along the
        rays and then calculates the volume rendering integral.

        Params:
            input_dict: Dictionary with keys 'cam2world', 'intrinsics', and 'xy_pix'
            radiance_field: nn.Module instance of the radiance field we want to render.
            radiance_field_input: An optional first input to pass into 'radiance_field'.
            randomize_sampling: If true, inject uniform noise into the sample space to make
                the samples correspond to a continous distribution.
            ray_batch_chunk: The batch size for ray batching.

        Returns:
            Tuple of rgb, depth_map
            rgb: for each pixel coordinate x_pix, the color of the respective ray.
            depth_map: for each pixel coordinate x_pix, the depth of the respective ray.

        """
        batch_size, num_rays = xy_pix.shape[0], xy_pix.shape[1]

        # Compute the ray directions in world coordinates.
        ros, rds = get_world_rays(xy_pix, intrinsics, cam2world)

        # Generate the points along rays and their depth values
        pts, z_vals = sample_points_along_rays(
            self._near,
            self._far,
            self._n_samples,
            ros,
            rds,
            device=xy_pix.device,
            randomize_sampling=randomize_sampling,
        )

        if radiance_field_input is not None:
            radiance_field_input_reshaped = einops.repeat(
                radiance_field_input,
                "b d -> b (num_rays num_samples) d",
                num_rays=num_rays,
                num_samples=self._n_samples,
            ).reshape(
                -1, radiance_field_input.shape[-1]
            )  # Shape (b*num_rays, d)

        # Flatten for ray batching
        pts = pts.reshape(-1, 3)

        # Render rays in batches to avoid OOM
        rads = []
        sigmas = []
        for i in range(0, pts.shape[0], ray_batch_chunk):
            if radiance_field_input is None:
                rad, sigma = radiance_field(pts[i : i + ray_batch_chunk])
            else:
                rad, sigma = radiance_field(
                    radiance_field_input_reshaped[i : i + ray_batch_chunk],
                    pts[i : i + ray_batch_chunk],
                )
            rads.append(rad)
            sigmas.append(sigma)
        rad = torch.cat(rads, dim=0)
        sigma = torch.cat(sigmas, dim=0)

        # Reshape sigma and rad back to (batch_size, num_rays, self.n_samples, -1)
        sigma = sigma.view(batch_size, num_rays, self._n_samples, 1)
        rad = rad.view(batch_size, num_rays, self._n_samples, 3)

        # Compute pixel colors, depths, and weights via the volume integral.
        rgb, depth_map, weights = volume_integral(z_vals, sigma, rad)

        if self._white_back:
            # Encourage learning zero density in background regions (zero loss if
            # rgb = accum = 0.0)
            accum = weights.sum(dim=-2)
            rgb = rgb + (1.0 - accum)

        return rgb, depth_map
