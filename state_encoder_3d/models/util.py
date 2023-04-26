from torch import nn


def init_weights_normal(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity="relu", mode="fan_in")

        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
