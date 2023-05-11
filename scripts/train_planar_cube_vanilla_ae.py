import torch
from tqdm import tqdm
import time
import os
import shutil
from argparse import Namespace
import json

import wandb
import matplotlib.pyplot as plt

from state_encoder_3d.models import (
    init_weights_normal,
    CNNImageDecoder,
    CoordCatCNNImageDecoder,
    CompNeRFImageEncoder,
    state_contrastive_loss,
)
from state_encoder_3d.dataset import PlanarCubeDataset

config = Namespace(
    log_path=f"outputs/planar_cube_vanilla_ae_{time.strftime('%Y-%b-%d-%H-%M-%S')}",
    checkpoint_path=f"outputs/planar_cube_vanilla_ae_{time.strftime('%Y-%b-%d-%H-%M-%S')}/checkpoints",
    data_path="data/planar_cube_grid_blue_floor.zarr",
    batch_size=100,
    latent_dim=256,
    w_ct=0.1,
    decoder_hidden_ch=256,
    resnet_out_dim=2048,
    lr=5e-4,
    img_res=(64, 64),
    num_steps=50001,
    steps_til_summary=100,
    steps_til_plot=100,
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


def plot_output_ground_truth(model_output, ground_truth, resolution):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), squeeze=False)
    axes[0, 0].imshow(model_output.cpu().view(*resolution).detach().numpy())
    axes[0, 0].set_title("Trained MLP")
    axes[0, 1].imshow(ground_truth.cpu().view(*resolution).detach().numpy())
    axes[0, 1].set_title("Ground Truth")

    for i in range(2):
        axes[0, i].set_axis_off()

    return fig


def main():
    current_time = time.strftime("%Y-%b-%d-%H-%M-%S")
    wandb.init(
        project="state_encoder_3d",
        name=f"train_cube_vanilla_ae_{current_time}",
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

    num_views = 1 if config.w_ct == 0 else 2
    dataset = PlanarCubeDataset(
        data_store_path=config.data_path,
        num_views=num_views,
        sample_neg_image=config.w_ct > 0,
        num_neg_views=1 if config.w_ct > 0 else 0,
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
    # NOTE: num_up is determined by the image resolution as we need to upsample to that resolution
    decoder = CNNImageDecoder(
        in_ch=config.latent_dim, hidden_ch=config.decoder_hidden_ch, out_ch=3, num_up=6
    ).to(device)
    # decoder = CoordCatCNNImageDecoder(
    #     in_ch=config.latent_dim, hidden_ch=config.decoder_hidden_ch, out_ch=3, num_up=6
    # ).to(device)
    decoder.apply(init_weights_normal)

    encoder_optim = torch.optim.Adam(
        encoder.parameters(), lr=config.lr, betas=(0.9, 0.999)
    )
    decoder_optim = torch.optim.Adam(
        decoder.parameters(), lr=config.lr, betas=(0.9, 0.999)
    )

    img2mse = lambda x, y: torch.mean((x - y) ** 2)

    for step in tqdm(range(config.num_steps)):
        model_input = next(dataloader)
        gt_image = model_input["rgb"].to(device)
        gt_image = gt_image.view(
            config.batch_size, num_views, config.img_res[0], config.img_res[1], 3
        )
        neg_image = model_input["neg_rgb"].to(device)

        encoder_input = gt_image.permute(0, 1, 4, 2, 3)
        latent = encoder(encoder_input.view(-1, *encoder_input.shape[-3:]))

        decoded_image = decoder(latent)
        decoded_image = decoded_image.view(
            config.batch_size, num_views, 3, config.img_res[0], config.img_res[1]
        ).permute(0, 1, 3, 4, 2)

        if config.w_ct > 0:
            latent = latent.reshape(config.batch_size, num_views, config.latent_dim)
            anchor_latent = latent[:, 0].squeeze(1)  # Shape (B, D)
            pos_latent = latent[:, 1].squeeze(1)  # Shape (B, D)

            neg_encoder_input = neg_image.view(
                config.batch_size, config.img_res[0], config.img_res[1], 3
            ).permute(0, 3, 1, 2)
            neg_latent = encoder(neg_encoder_input).reshape(
                config.batch_size, config.latent_dim
            )  # Shape (B, D)

            loss_ct = state_contrastive_loss(anchor_latent, pos_latent, neg_latent)
        else:
            loss_ct = torch.tensor([0.0])

        loss_rec = img2mse(decoded_image, gt_image)

        loss = loss_rec + config.w_ct * loss_ct.item()
        wandb.log(
            {
                "loss": loss.item(),
                "loss_rec": loss_rec.item(),
                "loss_ct": loss_ct.item(),
            }
        )

        encoder_optim.zero_grad()
        decoder_optim.zero_grad()
        loss.backward()
        encoder_optim.step()
        decoder_optim.step()

        if not step % config.steps_til_summary:
            print(f"Step {step}: loss = {float(loss.detach().cpu()):.5f}")

            # Remove old checkpoints
            shutil.rmtree(config.checkpoint_path)
            os.mkdir(config.checkpoint_path)

            # Save new weights
            save(name="encoder", step=step, model=encoder, optim=encoder_optim)
            save(name="decoder", step=step, model=decoder, optim=decoder_optim)

        if not step % config.steps_til_plot:
            fig = plot_output_ground_truth(
                decoded_image[0, 0],
                gt_image[0, 0],
                resolution=(config.img_res[0], config.img_res[1], 3),
            )
            wandb.log({f"step_{step}": fig})
            plt.close(fig)


if __name__ == "__main__":
    main()
