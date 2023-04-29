import torch
from tqdm import tqdm
import einops
import time
import os
import shutil

import wandb
import matplotlib.pyplot as plt

from state_encoder_3d.models import (
    LatentNeRF,
    VolumeRenderer,
    CompNeRFStateEncoder,
    init_weights_normal,
)
from state_encoder_3d.dataset import SRNsCarsDataset

OUT_PATH = f"outputs/srn_ae_{time.strftime('%Y-%b-%d-%H-%M-%S')}/checkpoints"


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


def plot_output_ground_truth(img, depth, gt_img, resolution):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), squeeze=False)
    axes[0, 0].imshow(img.cpu().view(*resolution).detach().numpy())
    axes[0, 0].set_title("Trained MLP")
    axes[0, 1].imshow(gt_img.cpu().view(*resolution).detach().numpy())
    axes[0, 1].set_title("Ground Truth")

    depth = depth.cpu().view(*resolution[:2]).detach().numpy()
    axes[0, 2].imshow(depth, cmap="Greys")
    axes[0, 2].set_title("Depth")

    for i in range(3):
        axes[0, i].set_axis_off()

    return fig


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

    img_sl = 64
    batch_size = 2
    num_views = 10
    dataset = SRNsCarsDataset(
        max_num_instances=None,
        img_sidelength=img_sl,
        num_views=num_views,
        rand_views=True,
        cars_path="notebooks/cars_train.hdf5",
    )
    dataloader = iter(torch.utils.data.DataLoader(dataset, batch_size=batch_size))

    latent_dim = 256
    lr = 1e-4

    encoder = CompNeRFStateEncoder(out_ch=latent_dim, in_ch=3, resnet_out_dim=2048).to(
        device
    )
    nerf = LatentNeRF(latent_ch=latent_dim).to(device)
    nerf.apply(init_weights_normal)
    renderer = VolumeRenderer(near=1.0, far=2.5, n_samples=100, white_back=True).to(
        device
    )

    encoder_optim = torch.optim.Adam(encoder.parameters(), lr=lr, betas=(0.9, 0.999))
    nerf_optim = torch.optim.Adam(nerf.parameters(), lr=lr, betas=(0.9, 0.999))

    img2mse = lambda x, y: torch.mean((x - y) ** 2)

    num_img_encoded = 8
    num_img_decoded = 2

    num_steps = 100001
    steps_til_summary = 1000
    steps_til_plot = 5000
    for step in tqdm(range(num_steps)):
        model_input, gt_image = next(dataloader)
        xy_pix = model_input["x_pix"].to(device)
        intrinsics = model_input["intrinsics"].to(device)
        c2w = model_input["cam2world"].to(device)
        gt_image = gt_image.to(device)

        encoder_input = gt_image.view(batch_size, num_views, img_sl, img_sl, 3).permute(
            0, 1, 4, 2, 3
        )[:, :num_img_encoded]
        encoder_input_dict = {
            "images": encoder_input,
            "extrinsics": c2w[:, :num_img_encoded],
        }
        latent = encoder(encoder_input_dict)

        xy_pix = einops.repeat(
            xy_pix, "B N c -> B num_decoded N c", num_decoded=num_img_decoded
        ).reshape(batch_size * num_img_decoded, *xy_pix.shape[-2:])
        c2w_decoded = c2w[:, -num_img_decoded:].reshape(
            batch_size * num_img_decoded, 4, 4
        )
        intrinsics = einops.repeat(
            intrinsics, "B x y -> B num_decoded x y", num_decoded=num_img_decoded
        ).reshape(batch_size * num_img_decoded, 3, 3)
        latent = einops.repeat(
            latent, "B D -> B num_decoded D", num_decoded=num_img_decoded
        ).reshape(batch_size * num_img_decoded, -1)
        rgb, depth = renderer(c2w_decoded, intrinsics, xy_pix, nerf, latent)

        gt_decoded_image = gt_image[:, -num_img_decoded:].reshape(
            batch_size * num_img_decoded, *gt_image.shape[-2:]
        )

        loss = img2mse(rgb, gt_decoded_image)
        wandb.log({"loss": loss.item()})

        encoder_optim.zero_grad()
        nerf_optim.zero_grad()
        loss.backward()
        encoder_optim.step()
        nerf_optim.step()

        if not step % steps_til_summary:
            print(f"Step {step}: loss = {float(loss.detach().cpu()):.5f}")

            # Remove old checkpoints
            shutil.rmtree(OUT_PATH)
            os.mkdir(OUT_PATH)
            
            # Save new weights
            save(name="encoder", step=step, model=encoder, optim=encoder_optim)
            save(name="nerf", step=step, model=nerf, optim=nerf_optim)

        if not step % steps_til_plot:
            fig = plot_output_ground_truth(
                rgb[0], depth[0], gt_decoded_image[0], resolution=(img_sl, img_sl, 3)
            )
            wandb.log({f"step_{step}": fig})


if __name__ == "__main__":
    main()
