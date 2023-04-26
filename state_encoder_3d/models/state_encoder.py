from typing import Dict

import torch
from torch import nn

from .image_encoder import CompNeRFImageEncoder


class CompNeRFStateEncoder(nn.Module):
    def __init__(self, out_ch: int, resnet_out_dim: int, in_ch: int = 3):
        super().__init__()

        self._image_encoder = CompNeRFImageEncoder(
            in_ch=in_ch, out_ch=out_ch, resnet_out_dim=resnet_out_dim
        )

        self._mlp1 = nn.Sequential(
            nn.Linear(out_ch + 16, out_ch),
            nn.ReLU(),
            nn.Linear(out_ch, out_ch),
            nn.ReLU(),
        )

        self._mlp2 = nn.Sequential(
            nn.Linear(out_ch, out_ch),
            nn.ReLU(),
            nn.Linear(out_ch, out_ch),
            nn.ReLU(),
            nn.Linear(out_ch, out_ch),
        )

    def forward(self, input: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        input: A dict with the keys:
            - "images": Images of shape (B,N,3,H,W).
            - "extrinsics": Extrinsics cam2world matrices of shape (B,N,4,4).
        return: Latent of shape (B,D).
        """
        images = input["images"]
        extrinsics = input["extrinsics"].view(*images.shape[:-2], -1)

        image_embeddings = self._image_encoder(images)  # Shape (B,N,D)

        embeddings = torch.cat(
            [image_embeddings, extrinsics], dim=-1
        )  # Shape (B,N,D+16)
        embeddings1 = self._mlp1(embeddings)  # Shape (B,N,D)

        # Average across camera viewpoints
        embedding = torch.mean(embeddings1, dim=1)  # Shape (B,D)
        embedding1 = self._mlp2(embedding)  # Shape (B,D)

        # Normalize to have unit L2 norm
        normalized_embedding = embedding1 / (embedding1.norm(dim=-1) + 1e-9)

        return normalized_embedding
