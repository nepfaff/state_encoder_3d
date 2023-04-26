import io
import imageio

import numpy as np
import torch
from torch.utils.data import IterableDataset
import einops
import h5py
import skimage
from skimage.transform import resize

from state_encoder_3d.utils import get_opencv_pixel_coordinates


def parse_rgb(hdf5_dataset):
    s = hdf5_dataset[...].tobytes()
    f = io.BytesIO(s)

    img = imageio.imread(f)[:, :, :3]
    img = skimage.img_as_float32(img)
    return img


def parse_intrinsics(hdf5_dataset):
    s = hdf5_dataset[...].tobytes()
    s = s.decode("utf-8")

    lines = s.split("\n")
    f, cx, cy, _ = map(float, lines[0].split())
    full_intrinsic = torch.tensor([[f, 0.0, cx], [0.0, f, cy], [0.0, 0, 1]])

    return full_intrinsic


def parse_pose(hdf5_dataset):
    raw = hdf5_dataset[...]
    ba = bytearray(raw)
    s = ba.decode("ascii")

    lines = s.splitlines()
    pose = np.zeros((4, 4), dtype=np.float32)

    for i in range(16):
        pose[i // 4, i % 4] = lines[0].split(" ")[i]

    pose = torch.from_numpy(pose.squeeze())
    return pose


class SRNsCarsDataset(IterableDataset):
    def __init__(
        self,
        max_num_instances=None,
        img_sidelength=None,
        num_views: int = 1,
        rand_views: bool = True,
        cars_path: str = "cars_train.hdf5",
    ):
        """
        Args:
            max_num_instances (_type_, optional): Number of car intances.
            img_sidelength (_type_, optional): Image sidelength to downsample to.
            num_views (int, optional): Number of views to load at once
                per instance.
            rand_views (bool, optional): If true, sample random views. Otherwise, always
                return the first 'num_views' for the instance.
        """
        self._file = h5py.File(cars_path, "r")
        self._instances = sorted(list(self._file.keys()))

        self._img_sidelength = img_sidelength
        self._num_views = num_views
        self._rand_views = rand_views

        assert num_views > 0
        assert img_sidelength > 0 and img_sidelength <= 128

        if max_num_instances is not None:
            self._instances = self._instances[:max_num_instances]

    def __len__(self):
        return len(self._instances)

    def __iter__(self, override_idx=None):
        while True:
            if override_idx is not None:
                idx = override_idx
            else:
                idx = (
                    0
                    if len(self._instances) == 1
                    else np.random.randint(0, len(self._instances) - 1)
                )

            key = self._instances[idx]

            instance = self._file[key]
            rgbs_ds = instance["rgb"]
            c2ws_ds = instance["pose"]

            rgb_keys = list(rgbs_ds.keys())
            c2w_keys = list(c2ws_ds.keys())

            observation_idx = (
                np.random.randint(0, len(rgb_keys), size=self._num_views)
                if self._rand_views
                else list(range(self._num_views))
            )
            if self._num_views == 1:
                observation_idx = observation_idx[0]
                rgb = parse_rgb(rgbs_ds[rgb_keys[observation_idx]])
            else:
                rgb = []
                for i in observation_idx:
                    rgb.append(parse_rgb(rgbs_ds[rgb_keys[i]]))
                rgb = np.stack(rgb, axis=0)

            x_pix = get_opencv_pixel_coordinates(
                *(rgb.shape[:2] if self._num_views == 1 else rgb.shape[1:3])
            )

            # There is a lot of white-space around the cars - we'll thus crop the images a bit:
            rgb = rgb[..., 32:-32, 32:-32, :]
            x_pix = x_pix[32:-32, 32:-32]

            # Nearest-neighbor downsampling of *both* the
            # RGB image and the pixel coordinates. This is better than down-
            # sampling RGB only and then generating corresponding pixel coordinates,
            # which generates "fake rays", i.e., rays that the camera
            # didn't actually capture with wrong colors. Instead, this simply picks a
            # subset of the "true" camera rays.
            if (
                self._img_sidelength is not None
                and rgb.shape[0] != self._img_sidelength
            ):
                if self._num_views == 1:
                    rgb = resize(
                        rgb,
                        (self._img_sidelength, self._img_sidelength),
                        anti_aliasing=False,
                        order=0,
                    )
                    rgb = torch.from_numpy(rgb)
                else:
                    rgbs = []
                    for i in range(self._num_views):
                        resized_rgb = resize(
                            rgb[i],
                            (self._img_sidelength, self._img_sidelength),
                            anti_aliasing=False,
                            order=0,
                        )
                        rgbs.append(torch.from_numpy(resized_rgb))
                    rgb = torch.stack(rgbs, dim=0)
                x_pix = resize(
                    x_pix,
                    (self._img_sidelength, self._img_sidelength),
                    anti_aliasing=False,
                    order=0,
                )

            x_pix = einops.rearrange(x_pix, "i j c -> (i j) c")
            rgb = einops.rearrange(rgb, "... i j c -> ... (i j) c")

            if self._num_views == 1:
                c2w = parse_pose(c2ws_ds[c2w_keys[observation_idx]])
            else:
                c2w = []
                for i in observation_idx:
                    c2w.append(parse_pose(c2ws_ds[c2w_keys[i]]))
                c2w = np.stack(c2w, axis=0)

            intrinsics = parse_intrinsics(instance["intrinsics.txt"])
            # Normalize intrinsics from resolution-specific intrinsics for 128x128
            intrinsics[:2, :3] /= 128.0

            model_input = {
                "cam2world": c2w,  # Shape (4,4) if num_views=1 else (num_views,4,4)
                "intrinsics": intrinsics,  # Shape (4,4)
                "x_pix": x_pix,  # Shape (i*j, c)
                "idx": torch.tensor([idx]),
            }
            # rgb of shape (w*h,3) if num_views=1 else (num_views,w*h,3)

            yield model_input, rgb
