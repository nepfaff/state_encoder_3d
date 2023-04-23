from distutils.core import setup

setup(
    name="3d_state_encoder",
    version="0.0.0",
    packages=["3d_state_encoder"],
    install_requires=[
        "torch",
        "hydra-core",
        "wandb",
        "omegaconf",
        "tqdm",
    ],
)
