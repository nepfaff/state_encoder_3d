import numpy as np
import torch
from torch.utils.data import IterableDataset
import einops
import zarr
import skimage

from state_encoder_3d.utils import get_opencv_pixel_coordinates


class PlanarCubeDataset(IterableDataset):
    def __init__(
        self,
        data_store_path: str,
        num_views: int,
        max_num_instances=None,
        rand_views: bool = True,
        sample_neg_image: bool = False,
    ):
        """
        Args:
            max_num_instances (_type_, optional): Maximum number of cube state instances.
            num_views (int, optional): Number of views to load at once per instance.
            rand_views (bool, optional): If true, sample random views. Otherwise, always
                return the first 'num_views' for the instance.
            sample_neg_image (bool, optional): Whether to sample a negative image for
                state-contrastive learning.
        """
        self._data_store = zarr.open(data_store_path)
        self._num_views = num_views
        self._rand_views = rand_views
        self._sample_neg_image = sample_neg_image

        self._num_instances = len(self._data_store.images)

        assert num_views > 0

        if max_num_instances is not None and max_num_instances < self._num_instances:
            self._num_instances = max_num_instances

    def __len__(self):
        return len(self._num_instances)

    def __iter__(self, override_idx=None):
        while True:
            if override_idx is not None:
                idx = override_idx
            else:
                idx = (
                    0
                    if self._num_instances == 1
                    else np.random.randint(0, self._num_instances - 1)
                )

            rgbs = np.asarray(self._data_store.images[idx])
            w2cs = np.asarray(self._data_store.world2cams, dtype=np.float32)
            intrinsics = np.asarray(self._data_store.intrinsics, dtype=np.float32)

            observation_idx = (
                np.random.randint(0, len(rgbs), size=self._num_views)
                if self._rand_views
                else list(range(self._num_views))
            )
            if self._num_views == 1:
                rgb = skimage.img_as_float32(rgbs[observation_idx[0]])
            else:
                rgb = []
                for i in observation_idx:
                    rgb.append(skimage.img_as_float32(rgbs[i]))
                rgb = np.stack(rgb, axis=0)

            x_pix = get_opencv_pixel_coordinates(
                *(rgb.shape[:2] if self._num_views == 1 else rgb.shape[1:3])
            )
            x_pix = einops.rearrange(x_pix, "i j c -> (i j) c")
            rgb = einops.rearrange(rgb, "... i j c -> ... (i j) c")

            if self._num_views == 1:
                c2w = np.linalg.inv(w2cs[observation_idx])
            else:
                c2w = []
                for i in observation_idx:
                    c2w.append(np.linalg.inv(w2cs[i]))
                c2w = np.stack(c2w, axis=0)

            if self._sample_neg_image:
                # Sample a negative image from a different state but same view-point
                # as the first observation index.
                while True:
                    neg_idx = np.random.randint(0, self._num_instances - 1)
                    if idx != neg_idx:
                        break
                neg_rgb = np.asarray(self._data_store.images[neg_idx])[observation_idx[0]]
                neg_rgb = skimage.img_as_float32(neg_rgb)
                neg_rgb = einops.rearrange(neg_rgb, "... i j c -> ... (i j) c")

            model_input = {
                "cam2world": torch.from_numpy(
                    c2w
                ),  # Shape (num_views,4,4)
                "intrinsics": torch.from_numpy(intrinsics),  # Shape (4,4)
                "x_pix": x_pix,  # Shape (i*j, c)
                "idx": torch.tensor([idx]),
            }
            # rgb of shape (w*h,3) if num_views=1 else (num_views,w*h,3)

            if self._sample_neg_image:
                yield model_input, rgb, neg_rgb
            else:
                yield model_input, rgb
