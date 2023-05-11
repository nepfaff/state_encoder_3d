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
        num_neg_views: int = -1,
        return_depth: bool = False,
    ):
        """
        Args:
            max_num_instances (_type_, optional): Maximum number of cube state instances.
            num_views (int, optional): Number of views to load at once per instance.
            rand_views (bool, optional): If true, sample random views. Otherwise, always
                return the first 'num_views' for the instance.
            sample_neg_image (bool, optional): Whether to sample a negative image for
                state-contrastive learning.
            return_depth (bool, optional): Whether to return the depth images that
                correspond to the ruturned RGB images.
        """
        self._data_store = zarr.open(data_store_path)
        self._num_views = num_views
        self._rand_views = rand_views
        self._sample_neg_image = sample_neg_image
        self._num_neg_views = num_neg_views
        self._return_depth = return_depth

        self._num_instances = len(self._data_store.images)

        assert num_views > 0
        if sample_neg_image:
            assert num_neg_views > 0

        if max_num_instances is not None and max_num_instances < self._num_instances:
            self._num_instances = max_num_instances

        if self._num_instances == 1:
            self._fixed_idx = np.random.randint(0, len(self._data_store.images) - 1)

    def __len__(self):
        return self._num_instances
    
    def __getitem__(self, idx):
        rgbs = np.asarray(self._data_store.images[idx])
        if self._return_depth:
            depths = np.asarray(self._data_store.depths[idx])
        w2cs = np.asarray(self._data_store.world2cams, dtype=np.float32)
        intrinsics = np.asarray(self._data_store.intrinsics, dtype=np.float32)
        finger_positions = np.asarray(
            self._data_store.finger_positions[idx], dtype=np.float32
        )
        box_positions = np.asarray(
            self._data_store.box_positions[idx], dtype=np.float32
        )
        env_state = np.concatenate((finger_positions, box_positions), axis=-1)

        observation_idx = (
            np.random.randint(0, len(rgbs), size=self._num_views)
            if self._rand_views
            else list(range(self._num_views))
        )
        if self._num_views == 1:
            rgb = skimage.img_as_float32(rgbs[observation_idx[0]])
            if self._return_depth:
                depth = depths[observation_idx[0]]
        else:
            rgb = []
            for i in observation_idx:
                rgb.append(skimage.img_as_float32(rgbs[i]))
            rgb = np.stack(rgb, axis=0)

            if self._return_depth:
                depth = []
                for i in observation_idx:
                    depth.append(depths[i])
                depth = np.stack(depth, axis=0)

        x_pix = get_opencv_pixel_coordinates(
            *(rgb.shape[:2] if self._num_views == 1 else rgb.shape[1:3])
        )
        x_pix = einops.rearrange(x_pix, "i j c -> (i j) c")
        rgb = einops.rearrange(rgb, "... i j c -> ... (i j) c")
        if self._return_depth:
            depth = einops.rearrange(depth, "... i j -> ... (i j)")

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
            neg_indices = []
            while len(neg_indices) < self._num_neg_views:
                neg_idx = np.random.randint(0, self._num_instances - 1)
                if neg_idx == idx or neg_idx in neg_indices:
                    continue
                neg_indices.append(neg_idx)
            neg_indices = neg_indices
            neg_rgbs = []
            for neg_idx in neg_indices:
                neg_rgb = np.asarray(self._data_store.images[neg_idx])[
                    observation_idx[0]
                ]
                neg_rgb = skimage.img_as_float32(neg_rgb)
                neg_rgb = einops.rearrange(neg_rgb, "... i j c -> ... (i j) c")
                neg_rgbs.append(neg_rgb)
            neg_rgb = np.asarray(neg_rgbs)

            if self._num_neg_views == 1:
                neg_rgb.squeeze(0)

        if not self._return_depth:
            depth = torch.tensor([])
        if not self._sample_neg_image:
            neg_rgb = torch.tensor([])
            
        model_input = {
            "cam2world": torch.from_numpy(c2w),  # Shape (num_views,4,4)
            "intrinsics": torch.from_numpy(intrinsics),  # Shape (4,4)
            "x_pix": x_pix,  # Shape (i*j, c)
            "idx": torch.tensor([idx]),
            "rgb": torch.from_numpy(rgb),  # Shape (w*h,3) if num_views=1 else (num_views,w*h,3)
            "neg_rgb": neg_rgb,  # Shape (w*h,3) if num_neg_views=1 else (num_neg_views,w*h,3)
            "depth": depth,  # Shape (w*h) if num_views=1 else (num_views,w*h)
            "env_state": torch.from_numpy(env_state),  # Shape (4,)
        }
        return model_input

    def __iter__(self, override_idx=None):
        while True:
            if override_idx is not None:
                idx = override_idx
            else:
                idx = (
                    self._fixed_idx
                    if self._num_instances == 1
                    else np.random.randint(0, self._num_instances - 1)
                )
            yield self.__getitem__(idx)
