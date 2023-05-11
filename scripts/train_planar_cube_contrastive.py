import torch
from tqdm import tqdm
import time
import os
import shutil
from argparse import Namespace
import json

import wandb

from state_encoder_3d.models import (
    CompNeRFImageEncoder,
    state_contrastive_loss,
)
from state_encoder_3d.dataset import PlanarCubeDataset

config = Namespace(
    log_path=f"outputs/planar_cube_contrastive_{time.strftime('%Y-%b-%d-%H-%M-%S')}",
    checkpoint_path=f"outputs/planar_cube_contrastive_{time.strftime('%Y-%b-%d-%H-%M-%S')}/checkpoints",
    data_path="data/planar_cube_grid_blue_floor_depth.zarr",
    batch_size=100,
    latent_dim=256,
    resnet_out_dim=2048,
    lr=1e-3,
    triplet_margin=0.5,
    img_res=(64, 64),
    num_steps=50001,
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


def main():
    current_time = time.strftime("%Y-%b-%d-%H-%M-%S")
    wandb.init(
        project="state_encoder_3d",
        name=f"train_planar_cube_contrastive_{current_time}",
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
        num_views=2,
        sample_neg_image=True,
        num_neg_views=1,
    )
    dataloader = iter(
        torch.utils.data.DataLoader(dataset, batch_size=config.batch_size)
    )

    encoder = CompNeRFImageEncoder(
        out_ch=config.latent_dim,
        in_ch=3,
        resnet_out_dim=config.resnet_out_dim,
        normalize=True,
    ).to(device)

    encoder_optim = torch.optim.Adam(
        encoder.parameters(), lr=config.lr, betas=(0.9, 0.999)
    )

    for step in tqdm(range(config.num_steps)):
        model_input = next(dataloader)
        gt_image = model_input["rgb"].to(device)
        neg_image = model_input["neg_rgb"].to(device)

        encoder_input = gt_image.view(
            config.batch_size, 2, config.img_res[0], config.img_res[1], 3
        ).permute(0, 1, 4, 2, 3)
        neg_encoder_input = neg_image.view(
            config.batch_size, config.img_res[0], config.img_res[1], 3
        ).permute(0, 3, 1, 2)

        latents = encoder(encoder_input.reshape(-1, *encoder_input.shape[-3:]))
        latents = latents.reshape(config.batch_size, 2, config.latent_dim)
        anchor_latent = latents[:, 0].squeeze(1)  # Shape (B, D)
        pos_latent = latents[:, 1].squeeze(1)  # Shape (B, D)
        neg_latent = encoder(neg_encoder_input).reshape(
            config.batch_size, config.latent_dim
        )  # Shape (B, D)

        loss_ct = state_contrastive_loss(
            anchor_latent, pos_latent, neg_latent, margin=config.triplet_margin
        )
        wandb.log(
            {
                "loss": loss_ct.item(),
            }
        )

        encoder_optim.zero_grad()
        loss_ct.backward()
        encoder_optim.step()

        if not step % config.steps_til_summary:
            print(f"Step {step}: loss = {loss_ct.item():.5f}")

            # Remove old checkpoints
            shutil.rmtree(config.checkpoint_path)
            os.mkdir(config.checkpoint_path)

            # Save new weights
            save(name="encoder", step=step, model=encoder, optim=encoder_optim)


if __name__ == "__main__":
    main()
