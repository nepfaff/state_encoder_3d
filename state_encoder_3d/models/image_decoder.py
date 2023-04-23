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
