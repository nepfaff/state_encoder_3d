from state_encoder_3d.dataset_generation.planar_cube_env import PlanarCubeEnvironment


def main():
    env = PlanarCubeEnvironment(
        time_step=1e-3,
        scene_directive_path="models/planar_pushing_no_rotations.dmd.yaml",
        num_cameras=1,
        initial_box_position=[0.0, 0.0],
        initial_finger_position=[1.0, 0.0],
    )
    env.generate_sample_dataset("data/planar_cube_sample.zarr", 1000)


if __name__ == "__main__":
    main()
