import torch
from tqdm import tqdm
import einops
import time
import os
import shutil
from argparse import Namespace

import wandb

from state_encoder_3d.models import (
    LatentNeRF,
    VolumeRenderer,
    CompNeRFStateEncoder,
    init_weights_normal,
)
from state_encoder_3d.dataset import PlanarCubeDataset
from state_encoder_3d.utils import plot_output_ground_truth

config = Namespace(
    checkpoint_path=f"outputs/planar_cube_ae_{time.strftime('%Y-%b-%d-%H-%M-%S')}/checkpoints",
    batch_size=2,
    num_views=10,
    latent_dim=256,
    resnet_out_dim=2048,
    lr=1e-4,
    img_res=(64, 64),
    near=4.0,
    far=13.0,
    num_samples_per_ray=100,
    weight_ct=0.1,
    num_img_encoded=8,
    num_img_decoded=2,
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
        name=f"train_cube_ae_{current_time}",
        mode=config.wandb_mode,
        config=vars(config),
    )
    print(f"Config:\n{config}")

    if not os.path.exists(config.checkpoint_path):
        os.makedirs(config.checkpoint_path)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    print(f"Using device {device}")

    dataset = PlanarCubeDataset(
        data_store_path="data/planar_cube_grid.zarr",
        num_views=config.num_views,
        sample_neg_image=True,
    )
    dataloader = iter(
        torch.utils.data.DataLoader(dataset, batch_size=config.batch_size)
    )

    encoder = CompNeRFStateEncoder(
        out_ch=config.latent_dim,
        in_ch=3,
        resnet_out_dim=config.resnet_out_dim,
        compute_state_contrastive_loss=True,
    ).to(device)
    nerf = LatentNeRF(latent_ch=config.latent_dim).to(device)
    nerf.apply(init_weights_normal)
    renderer = VolumeRenderer(
        near=config.near,
        far=config.far,
        n_samples=config.num_samples_per_ray,
        white_back=True,
    ).to(device)

    encoder_optim = torch.optim.Adam(
        encoder.parameters(), lr=config.lr, betas=(0.9, 0.999)
    )
    nerf_optim = torch.optim.Adam(nerf.parameters(), lr=config.lr, betas=(0.9, 0.999))

    img2mse = lambda x, y: torch.mean((x - y) ** 2)

    for step in tqdm(range(config.num_steps)):
        model_input, gt_image, neg_image = next(dataloader)
        xy_pix = model_input["x_pix"].to(device)
        intrinsics = model_input["intrinsics"].to(device)
        c2w = model_input["cam2world"].to(device)
        gt_image = gt_image.to(device)
        neg_image = neg_image.to(device)

        encoder_input = gt_image.view(
            config.batch_size, config.num_views, config.img_res[0], config.img_res[1], 3
        ).permute(0, 1, 4, 2, 3)[:, : config.num_img_encoded]
        neg_image_encoder_input = neg_image.view(
            config.batch_size, config.img_res[0], config.img_res[1], 3
        ).permute(0, 3, 1, 2)
        encoder_input_dict = {
            "images": encoder_input,
            "neg_image": neg_image_encoder_input,
            "extrinsics": c2w[:, : config.num_img_encoded],
        }
        loss_ct: torch.Tensor
        latents, loss_ct = encoder(encoder_input_dict)

        xy_pix = einops.repeat(
            xy_pix, "B N c -> B num_decoded N c", num_decoded=config.num_img_decoded
        ).reshape(config.batch_size * config.num_img_decoded, *xy_pix.shape[-2:])
        c2w_decoded = c2w[:, -config.num_img_decoded :].reshape(
            config.batch_size * config.num_img_decoded, 4, 4
        )
        intrinsics = einops.repeat(
            intrinsics, "B x y -> B num_decoded x y", num_decoded=config.num_img_decoded
        ).reshape(config.batch_size * config.num_img_decoded, 3, 3)
        latents = einops.repeat(
            latents, "B D -> B num_decoded D", num_decoded=config.num_img_decoded
        ).reshape(config.batch_size * config.num_img_decoded, -1)
        rgb, depth = renderer(c2w_decoded, intrinsics, xy_pix, nerf, latents)

        gt_decoded_image = gt_image[:, -config.num_img_decoded :].reshape(
            config.batch_size * config.num_img_decoded, *gt_image.shape[-2:]
        )

        loss_rec = img2mse(rgb, gt_decoded_image)
        loss: torch.Tensor
        loss = loss_rec + config.weight_ct * loss_ct
        wandb.log(
            {
                "loss": loss.item(),
                "loss_ct": loss_ct.item(),
                "loss_rec": loss_rec.item(),
            }
        )

        encoder_optim.zero_grad()
        nerf_optim.zero_grad()
        loss.backward()
        encoder_optim.step()
        nerf_optim.step()

        if not step % config.steps_til_summary:
            print(f"Step {step}: loss = {float(loss.detach().cpu()):.5f}")

            # Remove old checkpoints
            shutil.rmtree(config.checkpoint_path)
            os.mkdir(config.checkpoint_path)

            # Save new weights
            save(name="encoder", step=step, model=encoder, optim=encoder_optim)
            save(name="nerf", step=step, model=nerf, optim=nerf_optim)

        if not step % config.steps_til_plot:
            fig = plot_output_ground_truth(
                rgb[0],
                depth[0],
                gt_decoded_image[0],
                resolution=(config.img_res[0], config.img_res[1], 3),
            )
            wandb.log({f"step_{step}": fig})


if __name__ == "__main__":
    main()
