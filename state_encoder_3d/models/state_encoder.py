from typing import Dict, Union, Tuple

import torch
from torch import nn

from .image_encoder import CompNeRFImageEncoder


def state_contrastive_loss(
    anchor: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor, margin: float = 2.0
):
    # Use mean squared distance rather than L2 norm for gradient smoothness
    d_pos = torch.mean((anchor - pos) ** 2, dim=-1)
    d_neg = torch.mean((anchor - neg) ** 2, dim=-1)
    loss = torch.clamp(d_pos - d_neg + margin, min=0.0).mean()
    return loss


class CompNeRFStateEncoder(nn.Module):
    def __init__(
        self,
        out_ch: int,
        resnet_out_dim: int,
        in_ch: int = 3,
        compute_state_contrastive_loss: bool = False,
    ):
        super().__init__()
        self._compute_state_contrastive_loss = compute_state_contrastive_loss

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

    def forward(
        self, input: Dict[str, torch.Tensor]
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        input: A dict with the keys:
            - "images": Images of shape (B,N,3,H,W).
            - "extrinsics": Extrinsics cam2world matrices of shape (B,N,4,4).
            - "neg_image": The negative image if 'compute_state_contrastive_loss' is true.
        return: Latent of shape (B,D). Also returns the state contrastive loss if
            'compute_state_contrastive_loss' is true.
        """
        images = input["images"]
        assert len(images.shape) == 5
        B, N = images.shape[:2]
        extrinsics = input["extrinsics"].view(B, N, 16)

        images = images.reshape(-1, *images.shape[-3:])
        image_embeddings = self._image_encoder(images)
        image_embeddings = image_embeddings.view(B, N, -1)  # Shape (B,N,D)

        if self._compute_state_contrastive_loss:
            neg_image = input["neg_image"]
            neg_image = neg_image.reshape(-1, *neg_image.shape[-3:])
            neg_image_embedding = self._image_encoder(neg_image)
            neg_image_embedding = neg_image_embedding.view(B, 1, -1)

            loss_ct = state_contrastive_loss(
                anchor=image_embeddings[:, 0],
                pos=image_embeddings[:, 1],
                neg=neg_image_embedding,
            )

        embeddings = torch.cat(
            [image_embeddings, extrinsics], dim=-1
        )  # Shape (B,N,D+16)
        embeddings1 = self._mlp1(embeddings)  # Shape (B,N,D)

        # Average across camera viewpoints
        embedding = torch.mean(embeddings1, dim=1)  # Shape (B,D)
        embedding1 = self._mlp2(embedding)  # Shape (B,D)

        # Normalize to have unit L2 norm
        normalized_embedding = embedding1 / (
            embedding1.norm(dim=-1, keepdim=True) + 1e-9
        )

        if self._compute_state_contrastive_loss:
            return normalized_embedding, loss_ct
        return normalized_embedding
