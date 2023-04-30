import os
from typing import List
import shutil

import numpy as np
from manipulation.utils import AddPackagePaths
from pydrake.all import (
    DiagramBuilder,
    AddMultibodyPlantSceneGraph,
    RigidTransform,
    Simulator,
    Role,
    MultibodyPlant,
    Parser,
    CameraInfo,
    DepthRenderCamera,
    RenderCameraCore,
    ClippingRange,
    DepthRange,
    RgbdSensor,
    MakeRenderEngineGl,
    RenderEngineGlParams,
)
import zarr
from tqdm import tqdm

from .images import ImageGenerator
from .camera_poses import generate_camera_poses


def get_parser(plant: MultibodyPlant) -> Parser:
    """Creates a parser for a plant and adds package paths to it."""
    parser = Parser(plant)
    AddPackagePaths(parser)
    parser.package_map().AddPackageXml(os.path.abspath("package.xml"))
    return parser


class PlanarCubeEnvironment:
    def __init__(
        self,
        time_step: float,
        scene_directive_path: str,
        min_pos: float = -4.5,
        max_pos: float = 4.5,
    ):
        self._time_step = time_step
        self._scene_directive_path = scene_directive_path
        self._min_pos = min_pos
        self._max_pos = max_pos

        # Camera intrinsics
        self._camera_info = CameraInfo(width=64, height=64, fov_y=np.pi / 3.0)
        self._intrinsics = np.array(
            [
                [self._camera_info.focal_x(), 0.0, self._camera_info.center_x()],
                [0.0, self._camera_info.focal_y(), self._camera_info.center_y()],
                [0.0, 0.0, 1.0],
            ]
        )

        self._setup()

    def _add_cameras(
        self, camera_poses: np.ndarray, camera_info: CameraInfo, starting_cam_idx: int
    ) -> None:
        """
        Adds depth cameras to the scene.
        :param camera_poses: Homogenous world2cam transforms of shape (n,4,4) where n is
            the number of camera poses. OpenCV convention.
        """
        parent_frame_id = self._scene_graph.world_frame_id()
        for i, X_CW in enumerate(camera_poses):
            i += starting_cam_idx
            depth_camera = DepthRenderCamera(
                RenderCameraCore(
                    self._renderer,
                    camera_info,
                    ClippingRange(near=0.1, far=150.0),
                    RigidTransform(),
                ),
                DepthRange(0.1, 10.0),
            )
            rgbd = self._builder.AddSystem(
                RgbdSensor(
                    parent_id=parent_frame_id,
                    X_PB=RigidTransform(np.linalg.inv(X_CW)),
                    depth_camera=depth_camera,
                    show_window=False,
                )
            )
            self._builder.Connect(
                self._scene_graph.get_query_output_port(),
                rgbd.query_object_input_port(),
            )

            # Export the camera outputs
            self._builder.ExportOutput(
                rgbd.color_image_output_port(), f"camera{i}_rgb_image"
            )
            self._builder.ExportOutput(
                rgbd.depth_image_32F_output_port(), f"camera{i}_depth_image"
            )
            self._builder.ExportOutput(
                rgbd.label_image_output_port(), f"camera{i}_label_image"
            )

    def _setup_cameras(self) -> None:
        # Add renderer
        self._renderer = "PlanarCubeEnvRenderer"
        if not self._scene_graph.HasRenderer(self._renderer):
            self._scene_graph.AddRenderer(
                self._renderer, MakeRenderEngineGl(RenderEngineGlParams())
            )

        X_CW = generate_camera_poses(
            z_distances=[6.0, 8.0, 10, 12.0],
            radii=[12.0, 8.0, 4.0, 0.0],
            num_poses=[10, 10, 5, 1],
        )
        # The planar cube env already contains one camera
        self._add_cameras(
            camera_poses=X_CW, camera_info=self._camera_info, starting_cam_idx=0
        )

        self._num_cameras = len(X_CW)
        self._world2cam_matrices = X_CW

    def _set_env_state(self, finger_pos: List[float], box_pos: List[float]) -> None:
        # Set box position
        context = self._simulator.get_mutable_context()
        plant_context = self._plant.GetMyMutableContextFromRoot(context)
        box = self._plant.GetBodyByName("box")
        box_model_instance = box.model_instance()
        if box.is_floating():
            self._plant.SetFreeBodyPose(
                plant_context,
                box,
                RigidTransform([*box_pos, 0.0]),
            )
        else:
            self._plant.SetPositions(plant_context, box_model_instance, box_pos)

        # Set finger position
        sphere = self._plant.GetBodyByName("sphere")
        sphere_model_instance = sphere.model_instance()
        if sphere.is_floating():
            self._plant.SetFreeBodyPose(
                plant_context,
                sphere,
                RigidTransform([*finger_pos, 0.0]),
            )
        else:
            self._plant.SetPositions(plant_context, sphere_model_instance, finger_pos)

    def _setup(self) -> None:
        # self._meshcat = StartMeshcat()

        # Setup environment
        self._builder = DiagramBuilder()
        self._plant, self._scene_graph = AddMultibodyPlantSceneGraph(
            self._builder, time_step=self._time_step
        )
        parser = get_parser(self._plant)
        parser.AddAllModelsFromFile(self._scene_directive_path)
        self._plant.Finalize()

        self._setup_cameras()

        # visualizer_params = MeshcatVisualizerParams()
        # visualizer_params.role = Role.kIllustration
        # self._visualizer = MeshcatVisualizer.AddToBuilder(
        #     builder,
        #     self._scene_graph,
        #     self._meshcat,
        #     visualizer_params,
        # )

        diagram = self._builder.Build()
        self._simulator = Simulator(diagram)

        # Set up image generator
        self._image_generator = ImageGenerator(
            max_depth_range=10.0, diagram=diagram, scene_graph=self._scene_graph
        )

    def _is_env_state_feasible(self) -> bool:
        """Returns false if the finger is inside the box and true otherwise."""
        context = self._simulator.get_mutable_context()
        scene_graph_context = self._scene_graph.GetMyMutableContextFromRoot(context)
        query_object = self._scene_graph.get_query_output_port().Eval(
            scene_graph_context
        )
        inspector = query_object.inspector()
        box_frame_id = self._plant.GetBodyFrameIdOrThrow(
            self._plant.GetBodyByName("box").index()
        )
        sphere_frame_id = self._plant.GetBodyFrameIdOrThrow(
            self._plant.GetBodyByName("sphere").index()
        )
        box_geometry_id = inspector.GetGeometries(box_frame_id, Role.kProximity)[0]
        sphere_geometry_id = inspector.GetGeometries(sphere_frame_id, Role.kProximity)[
            0
        ]
        distance = query_object.ComputeSignedDistancePairClosestPoints(
            box_geometry_id, sphere_geometry_id
        ).distance
        return distance >= 0.0

    def _save_dataset(
        self,
        path: str,
        images: np.ndarray,
        finger_positions: np.ndarray,
        box_positions: np.ndarray,
        intrinsics: np.ndarray,
        world2cams: np.ndarray,
    ) -> None:
        if os.path.exists(path):
            print(
                f"Dataset storage path {path} already exists. Deleting the old dataset."
            )
            shutil.rmtree(path)

        store = zarr.DirectoryStore(path)
        root = zarr.group(store=store)
        image_store = root.zeros_like("images", images)
        image_store[:] = images
        finger_pos_store = root.zeros_like("finger_positions", finger_positions)
        finger_pos_store[:] = finger_positions
        box_pos_store = root.zeros_like("box_positions", box_positions)
        box_pos_store[:] = box_positions
        intrinsics_store = root.zeros_like("intrinsics", intrinsics)
        intrinsics_store[:] = intrinsics
        world2cams_store = root.zeros_like("world2cams", world2cams)
        world2cams_store[:] = world2cams

    def generate_sample_dataset(self, dataset_path: str, num_samples: int) -> None:
        finger_positions = []  # Shape (N, 2)
        box_positions = []  # Shape (N, 2)
        images = []  # Shape (N, num_views, W, H, C)
        for _ in tqdm(range(num_samples)):
            # Set random scene state
            finger_pos = (self._max_pos - self._min_pos) * np.random.random_sample(
                2
            ) + self._min_pos
            box_pos = (self._max_pos - self._min_pos) * np.random.random_sample(
                2
            ) + self._min_pos
            self._set_env_state(finger_pos=finger_pos, box_pos=box_pos)

            if not self._is_env_state_feasible():
                continue

            views = []
            for cam_idx in range(self._num_cameras):
                image, _, _, _ = self._image_generator.get_camera_data(
                    camera_name=f"camera{cam_idx}",
                    context=self._simulator.get_context(),
                )
                views.append(image)

            finger_positions.append(finger_pos)
            box_positions.append(box_pos)
            images.append(views)

        finger_positions = np.asarray(finger_positions)
        box_positions = np.asarray(box_positions)
        images = np.asarray(images)

        self._save_dataset(
            dataset_path,
            images,
            finger_positions,
            box_positions,
            self._intrinsics,
            self._world2cam_matrices,
        )

    def generate_grid_dataset(self, dataset_path: str) -> None:
        # Discretization based on shape width of 1m
        shape_width = 1.0
        num_steps = int((self._max_pos - self._min_pos) / shape_width)
        samples_1d = np.linspace(self._min_pos, self._max_pos, num_steps)
        samples_4d = np.stack(
            np.meshgrid(samples_1d, samples_1d, samples_1d, samples_1d, indexing="ij"),
            axis=-1,
        ).reshape(-1, 4)

        finger_positions = []  # Shape (N, 2)
        box_positions = []  # Shape (N, 2)
        images = []  # Shape (N, num_views, W, H, C)
        for sample in tqdm(samples_4d):
            finger_pos = sample[:2]
            box_pos = sample[2:]
            self._set_env_state(finger_pos=finger_pos, box_pos=box_pos)

            if not self._is_env_state_feasible():
                continue

            views = []
            for cam_idx in range(self._num_cameras):
                image, _, _, _ = self._image_generator.get_camera_data(
                    camera_name=f"camera{cam_idx}",
                    context=self._simulator.get_context(),
                )
                views.append(image)

            finger_positions.append(finger_pos)
            box_positions.append(box_pos)
            images.append(views)

        finger_positions = np.asarray(finger_positions)
        box_positions = np.asarray(box_positions)
        images = np.asarray(images)

        self._save_dataset(
            dataset_path,
            images,
            finger_positions,
            box_positions,
            self._intrinsics,
            self._world2cam_matrices,
        )
