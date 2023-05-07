import torch
from tqdm import tqdm
import einops
import time
import os
import shutil
from argparse import Namespace
import json

import wandb

from state_encoder_3d.models import (
    LatentNeRF,
    VolumeRenderer,
    init_weights_normal,
)
from state_encoder_3d.dataset import PlanarCubeDataset
from state_encoder_3d.utils import plot_output_ground_truth

config = Namespace(
    log_path=f"outputs/planar_cube_ae_{time.strftime('%Y-%b-%d-%H-%M-%S')}",
    checkpoint_path=f"outputs/planar_cube_ae_{time.strftime('%Y-%b-%d-%H-%M-%S')}/checkpoints",
    data_path="data/planar_cube_grid_blue_floor.zarr",
    batch_size=2,
    latent_dim=256,
    resnet_out_dim=2048,
    lr=5e-4,
    img_res=(64, 64),
    near=4.0,
    far=13.0,
    num_samples_per_ray=250,
    num_steps=500001,
    steps_til_summary=1000,
    steps_til_plot=5000,
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
        name=f"train_cube_nerf_{current_time}",
        mode="offline",
    )

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
        max_num_instances=1,
        num_views=1,
    )
    dataloader = iter(
        torch.utils.data.DataLoader(dataset, batch_size=config.batch_size)
    )

    nerf = LatentNeRF(latent_ch=config.latent_dim).to(device)
    nerf.apply(init_weights_normal)
    # Near and far based on z_distances in PlanarCubeEnvironment
    renderer = VolumeRenderer(
        near=config.near,
        far=config.far,
        n_samples=config.num_samples_per_ray,
        white_back=True,
    ).to(device)

    optim = torch.optim.Adam(nerf.parameters(), lr=config.lr, betas=(0.9, 0.999))

    img2mse = lambda x, y: torch.mean((x - y) ** 2)

    # Constant latent as we have a single scene
    latent = 0.1 * torch.rand((1, config.latent_dim), device=device)
    latent = einops.repeat(latent, "b ... -> (repeat b) ...", repeat=config.batch_size)

    for step in tqdm(range(config.num_steps)):
        model_input, gt_image = next(dataloader)
        xy_pix = model_input["x_pix"].to(device)
        intrinsics = model_input["intrinsics"].to(device)
        c2w = model_input["cam2world"].reshape(config.batch_size, 4, 4).to(device)

        rgb, depth = renderer(c2w, intrinsics, xy_pix, nerf, latent)

        loss = img2mse(rgb, gt_image.to(device))
        wandb.log({"loss": loss.item()})

        optim.zero_grad()
        loss.backward()
        optim.step()

        if not step % config.steps_til_summary:
            print(f"Step {step}: loss = {float(loss.detach().cpu()):.5f}")

            # Remove old checkpoints
            shutil.rmtree(config.checkpoint_path)
            os.mkdir(config.checkpoint_path)

            # Save new weights
            save(name="nerf", step=step, model=nerf, optim=optim)

        if not step % config.steps_til_plot:
            fig = plot_output_ground_truth(
                rgb[0],
                depth[0],
                gt_image[0],
                resolution=(config.img_res[0], config.img_res[1], 3),
            )
            wandb.log({f"step_{step}": fig})


if __name__ == "__main__":
    main()
