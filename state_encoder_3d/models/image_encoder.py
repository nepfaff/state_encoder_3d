from torch import nn

from .resnet import resnet18


class CompNeRFImageEncoder(nn.Module):
    """
    Image encoder as proposed in '3D Neural Scene Representations for Visomotor Control'.
    """

    def __init__(
        self, out_ch: int, resnet_out_dim: int, in_ch: int = 3, normalize: bool = False
    ):
        super().__init__()

        self._normalize = normalize

        self._encoder = resnet18(in_ch=in_ch)

        self._fc = nn.Sequential(
            nn.Linear(resnet_out_dim, out_ch), nn.ReLU(), nn.Linear(out_ch, out_ch)
        )

    def forward(self, images):
        x = self._encoder(images)
        x = self._fc(x)

        if self._normalize:
            # Normalize to have unit L2 norm
            x = x / (x.norm(dim=-1, keepdim=True) + 1e-9)

        return x


class DiffusionPolicyImageEncoder(nn.Module):
    """
    Image encoder as proposed in 'Diffusion Policy'.
    """

    # TODO
