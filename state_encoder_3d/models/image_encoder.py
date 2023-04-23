from torch import nn

from .resnet import resnet18


class CompNeRFImageEncoder(nn.Module):
    """
    Image encoder as proposed in '3D Neural Scene Representations for Visomotor Control'.
    """

    def __init__(self, out_ch: int, resnet_out_dim: int, in_ch: int = 3):
        super().__init__()

        self._encoder = resnet18(in_ch=in_ch)

        self._fc = nn.Sequential(
            nn.Linear(resnet_out_dim, out_ch), nn.ReLU(), nn.Linear(out_ch, out_ch)
        )

    def forward(self, images):
        x = self._encoder(images)
        x = self._fc(x)
        return x


class DiffusionPolicyImageEncoder(nn.Module):
    """
    Image encoder as proposed in 'Diffusion Policy'.
    """

    # TODO
