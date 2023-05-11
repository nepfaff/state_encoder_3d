import torch
from torch import nn
from tqdm import tqdm
import time
import os
import shutil
from argparse import Namespace
import json

import wandb

from state_encoder_3d.models import (
    LatentNeRF,
    ReluMLP,
    CompNeRFStateEncoder,
)
from state_encoder_3d.dataset import PlanarCubeDataset

config = Namespace(
    log_path=f"outputs/planar_cube_ae_state_decoder_{time.strftime('%Y-%b-%d-%H-%M-%S')}",
    checkpoint_path=f"outputs/planar_cube_ae_state_decoder_{time.strftime('%Y-%b-%d-%H-%M-%S')}/checkpoints",
    data_path="data/planar_cube_grid_blue_floor_depth.zarr",
    encoder_ckpt_path="outputs/nerf_ae_ct_and_depth/encoder_177000",
    nerf_ckpt_path="outputs/nerf_ae_ct_and_depth/nerf_177000",
    batch_size=100,
    latent_dim=256,
    resnet_out_dim=2048,
    env_state_decoder_latent_dim=512,
    env_state_decoder_num_hidden_layers=5,
    lr=1e-3,
    img_res=(64, 64),
    num_steps=5000,
    steps_til_summary=100,
    wandb_mode="offline",
)


def save(name, step, model, optim):
    save_name = f"{name}_{step}"
    path = os.path.join(config.checkpoint_path, save_name)
    torch.save(
        {
            "trainer_state_dict": model.state_dict(),
            "optim_primitive_state_dict": optim.state_dict(),
            "step": step,
        },
        path,
    )


def load(model: nn.Module, ckpt_path: str, device: torch.device) -> None:
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["trainer_state_dict"])


def set_requires_grad(model: nn.Module, requires_grad: bool) -> None:
    for param in model.parameters():
        param.requires_grad = requires_grad


def main():
    current_time = time.strftime("%Y-%b-%d-%H-%M-%S")
    wandb.init(
        project="state_encoder_3d",
        name=f"train_planar_cube_ae_state_decoder_{current_time}",
        mode=config.wandb_mode,
        config=vars(config),
    )
    print(f"Config:\n{config}")

    if not os.path.exists(config.checkpoint_path):
        os.makedirs(config.checkpoint_path)

    with open(os.path.join(config.log_path, "config.json"), "w") as fp:
        json.dump(vars(config), fp)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    print(f"Using device {device}")

    dataset = PlanarCubeDataset(
        data_store_path=config.data_path,
        num_views=1,
    )
    dataloader = iter(
        torch.utils.data.DataLoader(dataset, batch_size=config.batch_size)
    )

    encoder = CompNeRFStateEncoder(
        out_ch=config.latent_dim,
        in_ch=3,
        resnet_out_dim=config.resnet_out_dim,
    ).to(device)
    set_requires_grad(encoder, requires_grad=False)
    nerf = LatentNeRF(latent_ch=config.latent_dim).to(device)
    set_requires_grad(nerf, requires_grad=False)

    # Load model weights
    load(encoder, config.encoder_ckpt_path, device)
    load(nerf, config.nerf_ckpt_path, device)

    env_state_decoder = ReluMLP(
        in_ch=config.latent_dim,
        out_ch=4,
        num_hidden_layers=config.env_state_decoder_num_hidden_layers,
        latent_dim=config.env_state_decoder_latent_dim,
    ).to(device)
    decoder_optim = torch.optim.Adam(
        env_state_decoder.parameters(), lr=config.lr, betas=(0.9, 0.999)
    )

    mse = lambda x, y: torch.mean((x - y) ** 2)

    for step in tqdm(range(config.num_steps)):
        model_input = next(dataloader)
        c2w = model_input["cam2world"].to(device)
        gt_image = model_input["rgb"].to(device)
        gt_env_state = model_input["env_state"].to(device)

        encoder_input = gt_image.view(
            config.batch_size, 1, config.img_res[0], config.img_res[1], 3
        ).permute(0, 1, 4, 2, 3)
        encoder_input_dict = {
            "images": encoder_input,
            "extrinsics": c2w,
        }
        latents = encoder(encoder_input_dict)  # Shape (B,D)

        env_state = env_state_decoder(latents)

        loss: torch.Tensor = mse(env_state, gt_env_state)
        wandb.log({"loss": loss.item()})

        decoder_optim.zero_grad()
        loss.backward()
        decoder_optim.step()

        if not step % config.steps_til_summary:
            print(
                f"Step {step}: loss = {loss.item():.5f}; "
                + f"predicted_env_state:\n{env_state.detach().cpu()[0]}; gt_env_state:\n"
                + f"{gt_env_state.detach().cpu()[0]}"
            )

            # Remove old checkpoints
            shutil.rmtree(config.checkpoint_path)
            os.mkdir(config.checkpoint_path)

            # Save new weights
            save(
                name="env_state_decoder",
                step=step,
                model=env_state_decoder,
                optim=decoder_optim,
            )


if __name__ == "__main__":
    main()
