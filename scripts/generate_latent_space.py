import argparse
import os
import shutil

import zarr
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from state_encoder_3d.models import (
    CompNeRFStateEncoder,
    CompNeRFImageEncoder,
)
from state_encoder_3d.dataset import PlanarCubeDataset


def load(model: nn.Module, ckpt_path: str, device: torch.device) -> None:
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["trainer_state_dict"])


def set_requires_grad(model: nn.Module, requires_grad: bool) -> None:
    for param in model.parameters():
        param.requires_grad = requires_grad


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to the planar cube data zarr file.",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        required=True,
        choices=["state", "image"],
        help="The encoder model to use for generating the latents.",
    )
    parser.add_argument(
        "--ckpt_path", type=str, required=True, help="Path to the encoder checkpoint."
    )
    parser.add_argument(
        "--out", type=str, required=True, help="Path to output zarr file."
    )
    parser.add_argument(
        "--latent_dim",
        default=256,
        type=int,
    )
    parser.add_argument(
        "--resnet_out_dim",
        default=2048,
        type=int,
    )
    parser.add_argument(
        "--num_state_encoder_views",
        default=10,
        type=int,
    )
    parser.add_argument(
        "--num_views",
        default=24,
        type=int,
    )
    args = parser.parse_args()
    data_path = args.data
    encoder_name = args.encoder
    ckpt_path = args.ckpt_path
    out_path = args.out
    latent_dim = args.latent_dim
    resnet_out_dim = args.resnet_out_dim
    num_views = args.num_views
    num_state_encoder_views = args.num_state_encoder_views

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    print(f"Using device {device}")

    if encoder_name == "state":
        encoder = CompNeRFStateEncoder(
            out_ch=latent_dim,
            in_ch=3,
            resnet_out_dim=resnet_out_dim,
        ).to(device)
    elif encoder_name == "image":
        encoder = CompNeRFImageEncoder(
            out_ch=latent_dim,
            in_ch=3,
            resnet_out_dim=resnet_out_dim,
            normalize=True,
        ).to(device)

    load(encoder, ckpt_path, device)
    set_requires_grad(encoder, requires_grad=False)

    dataset = PlanarCubeDataset(
        data_store_path=data_path,
        num_views=num_views,
    )

    # States with corresponding latents
    states = []  # Shape (N, 4)
    # Shape (N, num_views, latent_dim) if image encoder or (N, latent_dim) if state
    # encoder
    latents = []
    for i in tqdm(range(len(dataset))):
        data = dataset[i]
        c2w = data["cam2world"].to(device)
        rgb = data["rgb"].to(device)
        state = data["env_state"].to(device)

        if encoder_name == "state":
            # Sample such that use 'num_state_encoder_views' unique views per state
            # encoding and encode each state 'num_views' times
            view_indices = []
            for _ in range(num_views):
                view_indices.append(
                    np.random.choice(
                        np.arange(num_views),
                        size=num_state_encoder_views,
                        replace=False,
                    )
                )
            view_indices = np.concatenate(view_indices, axis=0).flatten()

            encoded_rgbs = rgb[view_indices].reshape(
                num_views, num_state_encoder_views, 64, 64, 3
            )
            encoded_c2ws = c2w[view_indices].reshape(
                num_views, num_state_encoder_views, 4, 4
            )

            view_latent = []
            for rgb, c2w in zip(encoded_rgbs, encoded_c2ws):
                encoder_input = rgb.view(1, num_state_encoder_views, 64, 64, 3).permute(
                    0, 1, 4, 2, 3
                )
                encoder_input_dict = {
                    "images": encoder_input,
                    "extrinsics": c2w,
                }
                latent = encoder(encoder_input_dict)  # Shape (1, latent_dim)
                view_latent.append(latent)
            latent = torch.concat(view_latent, dim=0)  # Shape (num_views, latent_dim)

        elif encoder_name == "image":
            encoder_input = rgb.view(num_views, 64, 64, 3).permute(0, 3, 1, 2)
            latent = encoder(encoder_input)  # Shape (num_views, latent_dim)

        states.append(state.detach().cpu().numpy())
        latents.append(latent.detach().cpu().numpy())

    states = np.asarray(states)
    latents = np.asarray(latents)

    if os.path.exists(out_path):
        print(
            f"Dataset storage path {out_path} already exists. Deleting the old dataset."
        )
        shutil.rmtree(out_path)
    store = zarr.DirectoryStore(out_path)
    root = zarr.group(store=store)
    states_store = root.zeros_like("states", states)
    states_store[:] = states
    latents_store = root.zeros_like("latents", latents)
    latents_store[:] = latents


if __name__ == "__main__":
    main()
