import torch
from torch import nn


class CNNImageDecoder(nn.Module):
    def __init__(self, in_ch, hidden_ch, num_up, out_ch):
        super().__init__()

        self._in_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_ch,
                out_channels=hidden_ch,
                kernel_size=1,
                stride=1,
                padding="same",
                padding_mode="zeros",
            ),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self._hidden_conv = nn.Sequential(
            *(
                [
                    nn.ConvTranspose2d(
                        in_channels=hidden_ch,
                        out_channels=hidden_ch,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                        padding_mode="zeros",
                    ),
                    nn.LeakyReLU(),
                ]
                * num_up
            )
        )

        self._out_conv = nn.Conv2d(
            in_channels=hidden_ch,
            out_channels=out_ch,
            kernel_size=3,
            padding="same",
            padding_mode="zeros",
        )

    def forward(self, input: torch.Tensor):
        """
        input: Latent of shape (B, N) where B is the batch dimension and N the latent
            dimension.
        """
        x = input.unsqueeze(-1).unsqueeze(-1)
        x = self._in_conv(x)
        x = self._hidden_conv(x)
        x = self._out_conv(x)
        return x


class CoordCat(nn.Module):
    """
    This class takes an input of shape (B, L, ...) and concatenates normalied
    coordinates along the latent dimension. This is followed when we need
    positional information, such as in the case of CoordConv.
    """

    def __init__(self):
        super().__init__()

    def forward(self, input):
        x = torch.linspace(-1, 1, input.shape[2])
        y = torch.linspace(-1, 1, input.shape[3])
        xy = torch.meshgrid(x, y)
        xy = torch.stack(xy, dim=0)
        out = xy.repeat(input.shape[0], 1, 1, 1)

        return torch.cat([input, out.to(input.device)], dim=1)


class CoordCatCNNImageDecoder(nn.Module):
    def __init__(self, in_ch, hidden_ch, num_up, out_ch):
        super().__init__()

        self._in_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_ch,
                out_channels=hidden_ch,
                kernel_size=1,
                stride=1,
                padding="same",
                padding_mode="zeros",
            ),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        hidden_conv = [
            CoordCat(),
            nn.ConvTranspose2d(
                in_channels=hidden_ch + 2,
                out_channels=hidden_ch,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                padding_mode="zeros",
            ),
            nn.LeakyReLU(),
        ] * num_up

        self._hidden_conv = nn.Sequential(*hidden_conv)

        self.out_conv = nn.Conv2d(
            in_channels=hidden_ch,
            out_channels=out_ch,
            kernel_size=3,
            padding="same",
            padding_mode="zeros",
        )

    def forward(self, input: torch.Tensor):
        x = input.unsqueeze(-1).unsqueeze(-1)
        x = self._in_conv(x)
        x = self._hidden_conv(x)
        x = self.out_conv(x)
        return x
