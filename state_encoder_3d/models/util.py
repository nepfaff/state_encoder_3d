from torch import nn


def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
