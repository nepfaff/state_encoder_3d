import torch
import time
import os
import shutil
from argparse import Namespace
import json

import wandb
from tqdm import tqdm

from state_encoder_3d.models import ReluMLP
from state_encoder_3d.dataset import PlanarCubeLatentDataset

config = Namespace(
    log_path=f"outputs/planar_cube_state_decoder_{time.strftime('%Y-%b-%d-%H-%M-%S')}",
    checkpoint_path=f"outputs/planar_cube_state_decoder_{time.strftime('%Y-%b-%d-%H-%M-%S')}/checkpoints",
    data_path="data/checkpoints/ct_info_nce/encoder_latents.zarr",
    batch_size=1000,
    latent_dim=256,
    env_state_decoder_latent_dim=512,
    env_state_decoder_num_hidden_layers=5,
    lr=1e-3,
    img_res=(64, 64),
    num_steps=200000,
    steps_til_summary=10000,
    wandb_mode="online",
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
        name=f"train_planar_cube_state_decoder_{current_time}",
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
    
    dataset = PlanarCubeLatentDataset(
        zarr_path=config.data_path,
    )
    dataloader = iter(
        torch.utils.data.DataLoader(dataset, batch_size=config.batch_size)
    )

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
        latents, gt_states = next(dataloader)
        latents = latents.to(device)
        gt_states = gt_states.to(device)

        # Have multiple latents for each state
        latents = latents[
            :, torch.randint(low=0, high=latents.shape[1], size=(1,)).item(), :
        ].squeeze(1)

        states = env_state_decoder(latents)

        loss: torch.Tensor = mse(states, gt_states)
        wandb.log({"loss": loss.item()})

        decoder_optim.zero_grad()
        loss.backward()
        decoder_optim.step()

        if not step % config.steps_til_summary:
            print(
                f"Step {step}: loss = {loss.item():.5f}; "
                + f"predicted_state:\n{states.detach().cpu()[0]}; gt_states:\n"
                + f"{gt_states.detach().cpu()[0]}"
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
