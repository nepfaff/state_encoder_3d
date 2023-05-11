from state_encoder_3d.dataset_generation.planar_cube_env import PlanarCubeEnvironment


def main():
    env = PlanarCubeEnvironment(
        time_step=1e-3,
        scene_directive_path="models/planar_cube.dmd.yaml",
    )
    # env.generate_sample_dataset("data/planar_cube_sample.zarr", 1000)
    env.generate_grid_dataset("data/planar_cube_grid_depth.zarr")


if __name__ == "__main__":
    main()
