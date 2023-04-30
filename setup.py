from distutils.core import setup

setup(
    name="state_encoder_3d",
    version="0.0.0",
    packages=["state_encoder_3d"],
    install_requires=[
        "torch",
        "hydra-core",
        "wandb",
        "omegaconf",
        "tqdm",
        "matplotlib",
        "einops",
        "scikit-image",
        "h5py",
        "pyvirtualdisplay",
        "manipulation",
        "zarr",
        "open3d",
        "pytorch3d",
    ],
)
