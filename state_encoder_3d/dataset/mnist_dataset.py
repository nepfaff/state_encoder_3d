import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import datasets, transforms


class MNISTDataset(Dataset):
    def __init__(self):
        super().__init__()

        self._mnist = datasets.MNIST(
            "mnist_data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    torchvision.transforms.Pad(18, fill=0, padding_mode="constant"),
                    transforms.ToTensor(),
                ]
            ),
        )

    def __len__(self):
        return len(self._mnist)

    def __getitem__(self, idx) -> torch.Tensor:
        image, _ = self._mnist[idx]
        return image
