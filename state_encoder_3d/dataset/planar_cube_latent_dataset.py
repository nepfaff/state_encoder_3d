import numpy as np
from torch.utils.data import IterableDataset
import zarr


class PlanarCubeLatentDataset(IterableDataset):
    def __init__(self, zarr_path: str):
        data = zarr.open(zarr_path)
        self._states = np.asarray(data.states)
        self._latents = np.asarray(data.latents)

    def __len__(self):
        return len(self._states)

    def __iter__(self, override_idx=None):
        while True:
            idx = (
                override_idx
                if override_idx is not None
                else np.random.randint(0, len(self._states))
            )
            yield self._latents[idx], self._states[idx]
