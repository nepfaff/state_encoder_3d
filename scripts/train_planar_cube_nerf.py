import torch
from tqdm import tqdm
import einops
import time
import os
import shutil

import wandb

from state_encoder_3d.models import (
    LatentNeRF,
    VolumeRenderer,
    init_weights_normal,
)
from state_encoder_3d.dataset import PlanarCubeDataset
from state_encoder_3d.utils import plot_output_ground_truth

OUT_PATH = f"outputs/planar_cube_nerf_{time.strftime('%Y-%b-%d-%H-%M-%S')}/checkpoints"


def save(name, step, model, optim):
    save_name = f"{name}_{step}"
    path = os.path.join(OUT_PATH, save_name)
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
        project="state_encoder_3d", name=f"train_srn_ae_{current_time}", mode="offline"
    )

    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    print(f"Using device {device}")

    batch_size = 2
    dataset = PlanarCubeDataset(
        data_store_path="data/planar_cube_grid.zarr",
        max_num_instances=1,
        num_views=1,
    )
    dataloader = iter(torch.utils.data.DataLoader(dataset, batch_size=batch_size))

    latent_dim = 256
    nerf = LatentNeRF(latent_ch=latent_dim).to(device)
    nerf.apply(init_weights_normal)
    # Near and far based on z_distances in PlanarCubeEnvironment
    renderer = VolumeRenderer(near=4, far=13, n_samples=100, white_back=False).to(
        device
    )

    optim = torch.optim.Adam(nerf.parameters(), lr=5e-4, betas=(0.9, 0.999))

    img2mse = lambda x, y: torch.mean((x - y) ** 2)

    # Constant latent as we have a single scene
    latent = 0.1 * torch.rand((1, latent_dim), device=device)
    latent = einops.repeat(latent, "b ... -> (repeat b) ...", repeat=batch_size)

    num_steps = 100001
    steps_til_summary = 1000
    steps_til_plot = 5000
    for step in tqdm(range(num_steps)):
        model_input, gt_image = next(dataloader)
        xy_pix = model_input["x_pix"].to(device)
        intrinsics = model_input["intrinsics"].to(device)
        c2w = model_input["cam2world"].to(device)

        rgb, depth = renderer(c2w, intrinsics, xy_pix, nerf, latent)

        loss = img2mse(rgb, gt_image.to(device))
        wandb.log({"loss": loss.item()})

        optim.zero_grad()
        loss.backward()
        optim.step()

        if not step % steps_til_summary:
            print(f"Step {step}: loss = {float(loss.detach().cpu()):.5f}")

            # Remove old checkpoints
            shutil.rmtree(OUT_PATH)
            os.mkdir(OUT_PATH)

            # Save new weights
            save(name="nerf", step=step, model=nerf, optim=optim)

        if not step % steps_til_plot:
            fig = plot_output_ground_truth(
                rgb[0], depth[0], gt_image[0], resolution=(64, 64, 3)
            )
            wandb.log({f"step_{step}": fig})


if __name__ == "__main__":
    main()
